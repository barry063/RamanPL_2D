"""
preprocessing.py

Design goals:
- Single-spectrum first (1D arrays), with mapping/cube execution added in v0.3.5.
- Modality-aware presets can be built on top without forcing PL to over-process.
- Keep operations explicit, composable, and reproducible (serialisable step specs).
- Keep preprocessing steps scientifically 1D, while allowing mapping execution
  through a shared cube runner.

This module is designed to preserve existing single-spectrum behaviour while
providing a shared preprocessing path for mapping/batch integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from .schema import (
    AxisKind,
    Modality,
    PreprocessBackend,
    baseline_spec_to_runtime,
    normalise_axis_kind,
    normalise_baseline_spec,
    normalise_modality,
    normalise_preprocess_backend,
)
from .integration.ramanspy_bridge import resolve_preprocessing_backend
from .integration.ramanspy_translate import (
    pipeline_supports_ramanspy_translation,
    apply_pipeline_ramanspy_single,
    apply_pipeline_ramanspy_mapping,
)

# Local imports (package-safe)
try:
    from .baselineAPI import BaselineAPI
    from .dataImporter import DataImporter
except Exception:  # pragma: no cover (fallback for running as a script)
    from baselineAPI import BaselineAPI
    from dataImporter import DataImporter

try:
    from .integration.ramanspy_bridge import resolve_preprocessing_backend
    from .integration.ramanspy_translate import (
        pipeline_supports_ramanspy_translation,
        apply_pipeline_ramanspy_single,
        apply_pipeline_ramanspy_mapping,
    )
except Exception:  # pragma: no cover
    from ramanpl.integration.ramanspy_bridge import resolve_preprocessing_backend
    from ramanpl.integration.ramanspy_translate import (
        pipeline_supports_ramanspy_translation,
        apply_pipeline_ramanspy_single,
        apply_pipeline_ramanspy_mapping,
    )


ArrayLike1D = Union[np.ndarray, Sequence[float]]
Range2 = Optional[Tuple[float, float]]
BaselineSpec = Dict[str, Any]


@dataclass(frozen=True)
class SpectralDataset:
    """
    Canonical in-memory container for preprocessing.
    """
    x: np.ndarray
    y: np.ndarray
    modality: Modality
    axis_kind: AxisKind
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float).ravel())
        object.__setattr__(self, "y", np.asarray(self.y, dtype=float).ravel())
        object.__setattr__(self, "modality", normalise_modality(self.modality))
        object.__setattr__(self, "axis_kind", normalise_axis_kind(self.axis_kind))
        object.__setattr__(self, "meta", dict(self.meta))

    def copy_with(self, **kwargs) -> "SpectralDataset":
        d = {
            "x": self.x,
            "y": self.y,
            "modality": self.modality,
            "axis_kind": self.axis_kind,
            "meta": self.meta,
        }
        d.update(kwargs)
        if "meta" in kwargs:
            d["meta"] = dict(kwargs["meta"])
        return SpectralDataset(**d)


@dataclass(frozen=True)
class MappingPreprocessResult:
    """
    Result container for mapping/cube preprocessing.
    """
    x: np.ndarray
    cube: np.ndarray
    modality: Modality
    axis_kind: AxisKind
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float).ravel())
        object.__setattr__(self, "cube", np.asarray(self.cube, dtype=float))
        object.__setattr__(self, "modality", normalise_modality(self.modality))
        object.__setattr__(self, "axis_kind", normalise_axis_kind(self.axis_kind))
        object.__setattr__(self, "meta", dict(self.meta))

def _resolve_pipeline_backend_for_input(
    pipeline: "Pipeline | None",
    *,
    modality: str,
    axis_kind: str,
) -> Dict[str, Any]:
    """
    Resolve preprocessing backend selection for the current input.

    Step-3 policy
    -------------
    - native: always native
    - auto: use RamanSPy when available, supported for input, and all pipeline
            steps are currently translatable
    - ramanspy: require availability + supported input + translatable steps
    """
    requested_backend = "native" if pipeline is None else getattr(pipeline, "backend", "native")

    info = dict(resolve_preprocessing_backend(
        requested_backend=requested_backend,
        modality=modality,
        axis_kind=axis_kind,
    ))

    if pipeline is None:
        info["translation_supported"] = True
        info["unsupported_steps"] = []
        return info

    translation_supported, unsupported_steps = pipeline_supports_ramanspy_translation(pipeline)
    info["translation_supported"] = translation_supported
    info["unsupported_steps"] = list(unsupported_steps)

    # Explicit native always stays native
    if info["requested_backend"] == "native":
        info["resolved_backend"] = "native"
        info["execution_ready"] = True
        return info

    # Auto: promote to RamanSPy only when everything is ready
    if info["requested_backend"] == "auto":
        if (
            info["ramanspy_available"]
            and info["supported_for_input"]
            and translation_supported
        ):
            info["resolved_backend"] = "ramanspy"
            info["execution_ready"] = True
            info["reason"] = None
        else:
            info["resolved_backend"] = "native"
            info["execution_ready"] = True
            if (
                info["ramanspy_available"]
                and info["supported_for_input"]
                and not translation_supported
            ):
                info["reason"] = (
                    "Using native backend because RamanSPy translation is not yet "
                    f"implemented for: {unsupported_steps}"
                )
        return info

    # Forced RamanSPy: must be fully supported
    if info["requested_backend"] == "ramanspy":
        if not info["ramanspy_available"]:
            raise NotImplementedError(info["reason"])
        if not info["supported_for_input"]:
            raise NotImplementedError(info["reason"])
        if not translation_supported:
            raise NotImplementedError(
                "RamanSPy translation is currently implemented only for "
                f"CropByRange and SmoothSavGol. Unsupported steps: {unsupported_steps}"
            )

        info["resolved_backend"] = "ramanspy"
        info["execution_ready"] = True
        info["reason"] = None
        return info

    raise ValueError(f"Unsupported preprocessing backend '{info['requested_backend']}'.")

class PreprocessStep:
    """
    Base class for preprocessing steps.
    """

    name: str = "step"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class CropByRange(PreprocessStep):
    """
    Crop spectrum to a given x-range (inclusive bounds via mask_by_xrange).

    Notes
    -----
    - For v0.3.4, y is assumed 1D.
    - Respects meta['x_trimmed_on_load']: if True, this step becomes a no-op
      (but records provenance).
    """
    data_range: Range2
    name: str = "crop"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        if self.data_range is None:
            return ds

        meta = dict(ds.meta)

        if bool(meta.get("x_trimmed_on_load", False)):
            meta["crop"] = {"data_range": self.data_range, "applied": False, "reason": "x_trimmed_on_load"}
            return ds.copy_with(meta=meta)

        mask = DataImporter.mask_by_xrange(ds.x, self.data_range)
        if mask is None or np.sum(mask) < 3:
            raise ValueError(f"CropByRange produced too few points. data_range={self.data_range}")

        meta["crop"] = {"data_range": self.data_range, "applied": True}
        # keep mask for downstream use if needed (optional)
        meta["crop_mask"] = mask

        return ds.copy_with(x=ds.x[mask], y=ds.y[mask], meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "data_range": self.data_range}


@dataclass(frozen=True)
class SmoothSavGol(PreprocessStep):
    """
    Savitzky–Golay smoothing.

    Notes
    -----
    - Validates window_length odd and <= n_points.
    - This is a "fit-safe" step only if parameters are chosen conservatively,
      especially for PL.
    """
    window_length: int = 11
    polyorder: int = 3
    name: str = "savgol"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        y = np.asarray(ds.y, dtype=float).ravel()
        n = y.size

        w = int(self.window_length)
        p = int(self.polyorder)

        if w < 3:
            raise ValueError("SmoothSavGol.window_length must be >= 3.")
        if w % 2 == 0:
            w += 1  # minimal automatic correction
        if w > n:
            w = n if (n % 2 == 1) else (n - 1)
        if w < 3:
            raise ValueError("SmoothSavGol: spectrum too short after window adjustment.")
        if p >= w:
            raise ValueError("SmoothSavGol.polyorder must be < window_length.")

        y_s = savgol_filter(y, window_length=w, polyorder=p)

        meta = dict(ds.meta)
        meta["smoothing"] = {"method": "savgol", "window_length": w, "polyorder": p}
        # store intermediate for plotting/debug (v0.3.4 single-spectrum)
        meta["_smoothed_last"] = np.asarray(y_s, dtype=float).ravel()

        return ds.copy_with(y=y_s, meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "window_length": int(self.window_length), "polyorder": int(self.polyorder)}


@dataclass(frozen=True)
class BaselineSubtract(PreprocessStep):
    """
    Baseline subtraction using BaselineAPI.

    Canonical v0.3.8 spec:
        {"method": "poly", "poly_order": 3}
        {"method": "gaussian", "gaussian_sigma": 10}
        {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}
    """
    baseline_spec: Any = "poly"
    poly_degree: int = 3
    gaussian_sigma: int = 50
    clip_nonnegative: bool = True
    name: str = "baseline"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        x = np.asarray(ds.x, dtype=float).ravel()
        y = np.asarray(ds.y, dtype=float).ravel()

        spec = normalise_baseline_spec(
            self.baseline_spec,
            poly_degree=self.poly_degree,
            gaussian_sigma=self.gaussian_sigma,
        )
        method, bkwargs = baseline_spec_to_runtime(spec)

        result = BaselineAPI.subtract(
            x=x,
            y=y,
            method=method,
            clip_nonnegative=bool(self.clip_nonnegative),
            **bkwargs,
        )

        meta = dict(ds.meta)
        meta["baseline"] = {
            "spec": spec,
            "resolved_method": method,
            "clip_nonnegative": bool(self.clip_nonnegative),
            "kwargs": dict(bkwargs),
        }
        meta["_baseline_last"] = np.asarray(result.baseline, dtype=float).ravel()

        return ds.copy_with(
            y=np.asarray(result.y_corrected, dtype=float).ravel(),
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "baseline_spec": normalise_baseline_spec(
                self.baseline_spec,
                poly_degree=self.poly_degree,
                gaussian_sigma=self.gaussian_sigma,
            ),
            "clip_nonnegative": bool(self.clip_nonnegative),
        }


@dataclass(frozen=True)
class Pipeline:
    """
    Ordered list of preprocessing steps.
    """
    steps: List[PreprocessStep] = field(default_factory=list)
    backend: PreprocessBackend = "native"
    name: str = "pipeline"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        backend_info = _resolve_pipeline_backend_for_input(
            self,
            modality=ds.modality,
            axis_kind=ds.axis_kind,
        )

        if backend_info["resolved_backend"] == "ramanspy":
            x_out, y_out, meta_partial = apply_pipeline_ramanspy_single(ds, self)
            out = ds.copy_with(x=x_out, y=y_out, meta=meta_partial)
        else:
            out = ds
            for step in self.steps:
                out = step.apply(out)

        meta = dict(out.meta)
        meta["pipeline"] = self.to_dict()
        meta["preprocessing_backend"] = backend_info["resolved_backend"]
        meta["preprocessing_backend_info"] = backend_info
        return out.copy_with(meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": normalise_preprocess_backend(self.backend),
            "steps": [s.to_dict() for s in self.steps],
        }


def _apply_pipeline_steps(ds: SpectralDataset, steps: List[PreprocessStep]) -> SpectralDataset:
    """
    Apply an explicit list of preprocessing steps to a SpectralDataset.

    This is mainly used internally so mapping execution can:
    - apply CropByRange once at cube level, then
    - apply the remaining pointwise steps spectrum-by-spectrum.
    """
    out = ds
    for step in steps:
        out = step.apply(out)

    meta = dict(out.meta)
    meta["pipeline_steps_applied"] = [s.to_dict() for s in steps]
    return out.copy_with(meta=meta)


def _split_pipeline_for_mapping(pipeline: Optional[Pipeline]) -> Tuple[List[CropByRange], List[PreprocessStep]]:
    """
    Split a Pipeline into:
    1) axis-level crop steps (applied once to the shared x-axis / cube)
    2) pointwise steps (applied spectrum-by-spectrum)

    This keeps preprocessing steps scientifically 1D while avoiding repeated
    crop-mask construction for every pixel.
    """
    if pipeline is None:
        return [], []

    crop_steps: List[CropByRange] = []
    pointwise_steps: List[PreprocessStep] = []

    for step in pipeline.steps:
        if isinstance(step, CropByRange):
            crop_steps.append(step)
        else:
            pointwise_steps.append(step)

    return crop_steps, pointwise_steps

def _resolve_savgol_params_for_mapping(n_points: int, window_length: int, polyorder: int) -> Tuple[int, int]:
    """
    Resolve Savitzky–Golay parameters exactly as SmoothSavGol.apply() would,
    but once for a whole mapping cube.
    """
    w = int(window_length)
    p = int(polyorder)

    if w < 3:
        raise ValueError("SmoothSavGol.window_length must be >= 3.")
    if w % 2 == 0:
        w += 1
    if w > n_points:
        w = n_points if (n_points % 2 == 1) else (n_points - 1)
    if w < 3:
        raise ValueError("SmoothSavGol: spectrum too short after window adjustment.")
    if p >= w:
        raise ValueError("SmoothSavGol.polyorder must be < window_length.")

    return w, p

def _batched_polynomial_baseline_cube(
    *,
    x: np.ndarray,
    cube: np.ndarray,
    poly_degree: int,
) -> np.ndarray:
    """
    Compute polynomial baselines for an entire mapping cube in batch.

    Parameters
    ----------
    x : ndarray, shape [N]
        Shared spectral axis.
    cube : ndarray, shape [Y, X, N]
        Spectral cube.
    poly_degree : int
        Requested polynomial degree.

    Returns
    -------
    baseline : ndarray, shape [Y, X, N]
        Polynomial baseline evaluated for every spectrum.
    """
    x = np.asarray(x, dtype=float).ravel()
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("cube must be 3D with shape [Y, X, N].")
    if cube.shape[2] != x.size:
        raise ValueError("cube.shape[2] must match len(x).")
    if x.size == 0:
        raise ValueError("x must contain at least one point.")

    # Match BaselineAPI._baseline_poly graceful degree handling
    deg = int(poly_degree)
    deg = max(0, min(deg, x.size - 1))

    # Scale x to roughly [-1, 1] for numerical stability
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax > xmin:
        x_scaled = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    else:
        x_scaled = np.zeros_like(x, dtype=float)

    # Power-basis Vandermonde with ascending powers
    V = np.polynomial.polynomial.polyvander(x_scaled, deg)   # shape [N, deg+1]

    # Flatten cube to [P, N], where P = Y*X
    Ydim, Xdim, N = cube.shape
    Yflat = cube.reshape(-1, N)                              # shape [P, N]

    # Solve V @ C ≈ Y^T  -> C shape [deg+1, P]
    coeffs, _, _, _ = np.linalg.lstsq(V, Yflat.T, rcond=None)

    # Evaluate baseline on the same x grid
    baseline_flat = (V @ coeffs).T                           # shape [P, N]
    baseline = baseline_flat.reshape(Ydim, Xdim, N)

    return baseline

def _apply_vectorised_mapping_steps(
    *,
    x: np.ndarray,
    cube: np.ndarray,
    pointwise_steps: List[PreprocessStep],
) -> Tuple[np.ndarray, List[PreprocessStep], Dict[str, Any]]:
    """
    Apply mapping-safe vectorised steps over the full cube before any residual
    per-spectrum loop.

    Currently accelerated:
    - SmoothSavGol
    - BaselineSubtract with gaussian baseline
    - BaselineSubtract with polynomial baseline

    Returns
    -------
    cube_out
        Processed cube after vectorised steps.
    remaining_steps
        Steps that still need per-spectrum execution.
    shared_meta
        Shared provenance metadata for the accelerated steps.
    """
    cube_out = np.asarray(cube, dtype=float)
    remaining_steps: List[PreprocessStep] = []
    shared_meta: Dict[str, Any] = {}

    for step in pointwise_steps:
        # ------------------------------------------------------------
        # Fast path: Savitzky–Golay smoothing over axis=2
        # ------------------------------------------------------------
        if isinstance(step, SmoothSavGol):
            n_points = cube_out.shape[2]
            w, p = _resolve_savgol_params_for_mapping(
                n_points=n_points,
                window_length=step.window_length,
                polyorder=step.polyorder,
            )

            cube_out = savgol_filter(cube_out, window_length=w, polyorder=p, axis=2)

            shared_meta["smoothing"] = {
                "method": "savgol",
                "window_length": w,
                "polyorder": p,
            }
            shared_meta["_smoothed_last"] = np.asarray(cube_out[0, 0, :], dtype=float).ravel()
            continue

        # ------------------------------------------------------------
        # Fast path: baseline subtraction over axis=2 / whole cube
        # ------------------------------------------------------------
        if isinstance(step, BaselineSubtract):
            spec = normalise_baseline_spec(
                step.baseline_spec,
                poly_degree=step.poly_degree,
                gaussian_sigma=step.gaussian_sigma,
            )
            method, bkwargs = baseline_spec_to_runtime(spec)
            method_l = str(method).strip().lower()

            # ---- Gaussian baseline: fully vectorised
            if method_l == "gaussian":
                sigma = float(bkwargs.get("gaussian_sigma", step.gaussian_sigma))
                if sigma <= 0:
                    raise ValueError(f"gaussian_sigma must be > 0. Got {sigma}.")

                baseline = gaussian_filter1d(cube_out, sigma=sigma, axis=2)
                cube_out = cube_out - baseline

                if bool(step.clip_nonnegative):
                    cube_out = np.clip(cube_out, a_min=0.0, a_max=None)

                shared_meta["baseline"] = {
                    "spec": spec,
                    "resolved_method": method,
                    "clip_nonnegative": bool(step.clip_nonnegative),
                    "kwargs": dict(bkwargs),
                }
                shared_meta["_baseline_last"] = np.asarray(baseline[0, 0, :], dtype=float).ravel()
                continue

            # ---- Polynomial baseline: batched least-squares over all pixels
            if method_l == "poly":
                poly_order = int(bkwargs.get("poly_degree", step.poly_degree))
                baseline = _batched_polynomial_baseline_cube(
                    x=x,
                    cube=cube_out,
                    poly_degree=poly_order,
                )
                cube_out = cube_out - baseline

                if bool(step.clip_nonnegative):
                    cube_out = np.clip(cube_out, a_min=0.0, a_max=None)

                shared_meta["baseline"] = {
                    "spec": spec,
                    "resolved_method": method,
                    "clip_nonnegative": bool(step.clip_nonnegative),
                    "kwargs": dict(bkwargs),
                }
                shared_meta["_baseline_last"] = np.asarray(baseline[0, 0, :], dtype=float).ravel()
                continue

        # ------------------------------------------------------------
        # Fallback: keep step for the slower per-spectrum route
        # ------------------------------------------------------------
        remaining_steps.append(step)

    return cube_out, remaining_steps, shared_meta

def apply_pipeline_to_mapping_cube(
    *,
    x: ArrayLike1D,
    cube: np.ndarray,
    pipeline: Optional[Pipeline],
    modality: str,
    axis_kind: str,
    meta: Optional[Dict[str, Any]] = None,
) -> MappingPreprocessResult:
    """
    Apply a preprocessing pipeline to a spectral mapping cube.

    Behaviour
    ---------
    - CropByRange steps are applied once at the shared axis/cube level.
    - Fast vectorisable steps are applied once over the full cube:
        * SmoothSavGol
        * BaselineSubtract(method="gaussian")
    - Remaining steps are applied spectrum-by-spectrum.

    This preserves the scientific meaning of the current pipeline while
    substantially reducing Python-loop overhead for common mapping workflows.
    """
    x = np.asarray(x, dtype=float).ravel()
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("cube must be a 3D ndarray with shape [Y, X, N].")
    if cube.shape[2] != x.size:
        raise ValueError(
            f"cube.shape[2] ({cube.shape[2]}) must match len(x) ({x.size})."
        )

    meta_out: Dict[str, Any] = {} if meta is None else dict(meta)

    backend_info = _resolve_pipeline_backend_for_input(
        pipeline,
        modality=modality,
        axis_kind=axis_kind,
    )

    if pipeline is None:
        meta_out["preprocessing_backend"] = backend_info["resolved_backend"]
        meta_out["preprocessing_backend_info"] = backend_info
        return MappingPreprocessResult(
            x=x.copy(),
            cube=cube.copy(),
            modality=modality,
            axis_kind=axis_kind,
            meta=meta_out,
        )

    if backend_info["resolved_backend"] == "ramanspy":
        x_out, cube_out, meta_partial = apply_pipeline_ramanspy_mapping(
            x=x,
            cube_yxn=cube,
            pipeline=pipeline,
            modality=modality,
            axis_kind=axis_kind,
            meta=meta_out,
        )
        meta_partial = dict(meta_partial)
        meta_partial["pipeline"] = pipeline.to_dict()
        meta_partial["preprocessing_backend"] = backend_info["resolved_backend"]
        meta_partial["preprocessing_backend_info"] = backend_info

        return MappingPreprocessResult(
            x=x_out,
            cube=cube_out,
            modality=modality,
            axis_kind=axis_kind,
            meta=meta_partial,
        )

    crop_steps, pointwise_steps = _split_pipeline_for_mapping(pipeline)

    # ------------------------------------------------------------
    # 1) Apply crop steps once at cube level
    # ------------------------------------------------------------
    x_work = x.copy()
    cube_work = cube.copy()
    crop_history = []
    crop_mask_total = np.ones(x_work.shape, dtype=bool)

    for step in crop_steps:
        if step.data_range is None:
            continue

        if bool(meta_out.get("x_trimmed_on_load", False)):
            crop_history.append(
                {
                    "data_range": step.data_range,
                    "applied": False,
                    "reason": "x_trimmed_on_load",
                }
            )
            continue

        mask = DataImporter.mask_by_xrange(x_work, step.data_range)
        if mask is None or np.sum(mask) < 3:
            raise ValueError(
                f"CropByRange produced too few points for mapping cube. data_range={step.data_range}"
            )

        x_work = x_work[mask]
        cube_work = cube_work[:, :, mask]
        crop_mask_total = crop_mask_total[mask]

        crop_history.append(
            {
                "data_range": step.data_range,
                "applied": True,
            }
        )

    if crop_history:
        meta_out["crop"] = crop_history[0] if len(crop_history) == 1 else crop_history
        meta_out["crop_mask"] = crop_mask_total

    # ------------------------------------------------------------
    # 2) Apply fast vectorised steps over the whole cube
    # ------------------------------------------------------------
    cube_work, remaining_steps, fast_meta = _apply_vectorised_mapping_steps(
        x=x_work,
        cube=cube_work,
        pointwise_steps=pointwise_steps,
    )

    # ------------------------------------------------------------
    # 3) Apply any residual steps pointwise
    # ------------------------------------------------------------
    if remaining_steps:
        cube_processed = np.empty_like(cube_work, dtype=float)
        sample_meta = dict(fast_meta)

        base_meta_for_pointwise = dict(meta_out)
        base_meta_for_pointwise.update(fast_meta)

        for iy in range(cube_work.shape[0]):
            for ix in range(cube_work.shape[1]):
                ds0 = SpectralDataset(
                    x=x_work,
                    y=np.asarray(cube_work[iy, ix, :], dtype=float).ravel(),
                    modality=modality,
                    axis_kind=axis_kind,
                    meta=dict(base_meta_for_pointwise),
                )

                ds1 = _apply_pipeline_steps(ds0, remaining_steps)
                cube_processed[iy, ix, :] = np.asarray(ds1.y, dtype=float).ravel()

                if not sample_meta:
                    sample_meta = dict(ds1.meta)
                else:
                    for k, v in ds1.meta.items():
                        if k not in sample_meta:
                            sample_meta[k] = v
    else:
        cube_processed = cube_work
        sample_meta = dict(fast_meta)

    # ------------------------------------------------------------
    # 4) Shared provenance
    # ------------------------------------------------------------
    meta_final = dict(meta_out)
    meta_final["pipeline"] = pipeline.to_dict()
    meta_final["preprocessing_backend"] = backend_info["resolved_backend"]
    meta_final["preprocessing_backend_info"] = backend_info

    if sample_meta is not None:
        for key in (
            "smoothing",
            "baseline",
            "_smoothed_last",
            "_baseline_last",
            "pipeline_steps_applied",
        ):
            if key in sample_meta:
                meta_final[key] = sample_meta[key]

    return MappingPreprocessResult(
        x=x_work,
        cube=cube_processed,
        modality=modality,
        axis_kind=axis_kind,
        meta=meta_final,
    )

def build_legacy_single_spectrum_pipeline(
    *,
    data_range,
    smoothing,
    smooth_window,
    smooth_order,
    background_remove,
    baseline_method,
    poly_degree=None,
    gaussian_sigma=50,
    backend: str = "native",
) -> Pipeline:
    """
    Build a pipeline that reproduces the existing single-spectrum preprocessing order:
      1) smoothing (if enabled)
      2) baseline subtraction (if enabled)
      Cropping is intentionally NOT included here because you typically crop at import time
      for single spectra. (We can add it later if you standardise import->dataset flow.)
    """
    steps: List[PreprocessStep] = []

    # Note: data_range intentionally unused here (for single spectrum legacy behaviour).
    # Kept in signature so later refactor is low-friction.

    if smoothing:
        steps.append(SmoothSavGol(window_length=smooth_window, polyorder=smooth_order))

    if background_remove:
        steps.append(
            BaselineSubtract(
                baseline_spec=baseline_method,
                poly_degree=3 if poly_degree is None else poly_degree,
                gaussian_sigma=gaussian_sigma,
                clip_nonnegative=True,
            )
        )

    return Pipeline(steps=steps, backend=backend, name="legacy_single_spectrum")

def build_legacy_mapping_pipeline(
    *,
    data_range,
    smoothing,
    smooth_window,
    smooth_order,
    background_remove,
    baseline_method,
    poly_degree=None,
    gaussian_sigma=50,
    backend: str = "native",
) -> Pipeline:
    """
    Build a pipeline that reproduces the existing mapping preprocessing order:

      1) crop by data_range (if provided and not already trimmed on load)
      2) smoothing (if enabled)
      3) baseline subtraction (if enabled)

    Notes
    -----
    - Unlike the single-spectrum builder, mapping includes CropByRange because
      mapping datasets are commonly loaded as full cubes and then trimmed.
    - This function is intended to preserve legacy mapping behaviour while
      moving execution into the shared preprocessing framework.
    """
    steps: List[PreprocessStep] = []

    if data_range is not None:
        steps.append(CropByRange(data_range=data_range))

    if smoothing:
        steps.append(SmoothSavGol(window_length=smooth_window, polyorder=smooth_order))

    if background_remove:
        steps.append(
            BaselineSubtract(
                baseline_spec=baseline_method,
                poly_degree=3 if poly_degree is None else poly_degree,
                gaussian_sigma=gaussian_sigma,
                clip_nonnegative=True,
            )
        )

    return Pipeline(steps=steps, backend=backend, name="legacy_mapping")