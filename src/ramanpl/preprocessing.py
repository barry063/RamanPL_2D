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

from .schema import (
    AxisKind,
    Modality,
    baseline_spec_to_runtime,
    normalise_axis_kind,
    normalise_baseline_spec,
    normalise_modality,
)

# Local imports (package-safe)
try:
    from .baselineAPI import BaselineAPI
    from .dataImporter import DataImporter
except Exception:  # pragma: no cover (fallback for running as a script)
    from baselineAPI import BaselineAPI
    from dataImporter import DataImporter


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
    name: str = "pipeline"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        out = ds
        for step in self.steps:
            out = step.apply(out)
        meta = dict(out.meta)
        meta["pipeline"] = self.to_dict()
        return out.copy_with(meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "steps": [s.to_dict() for s in self.steps]}


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
    - All remaining steps are applied pointwise to each spectrum in the cube.
    - This preserves the current scientific meaning of smoothing/baseline steps
      while enabling mapping support in v0.3.5.

    Parameters
    ----------
    x
        Shared spectral axis, shape [N].
    cube
        Spectral cube, shape [Y, X, N].
    pipeline
        Preprocessing Pipeline. May be None.
    modality
        "Raman" or "PL".
    axis_kind
        "raman_shift_cm-1" | "energy_eV" | "wavelength_nm".
    meta
        Optional shared metadata/provenance.

    Returns
    -------
    MappingPreprocessResult
        Processed axis, processed cube, and shared metadata.
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

    if pipeline is None:
        return MappingPreprocessResult(
            x=x.copy(),
            cube=cube.copy(),
            modality=modality,
            axis_kind=axis_kind,
            meta=meta_out,
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
    # 2) Apply remaining steps pointwise
    # ------------------------------------------------------------
    cube_processed = np.empty_like(cube_work, dtype=float)
    sample_meta = None

    for iy in range(cube_work.shape[0]):
        for ix in range(cube_work.shape[1]):
            ds0 = SpectralDataset(
                x=x_work,
                y=np.asarray(cube_work[iy, ix, :], dtype=float).ravel(),
                modality=modality,
                axis_kind=axis_kind,
                meta=dict(meta_out),
            )

            ds1 = _apply_pipeline_steps(ds0, pointwise_steps)
            cube_processed[iy, ix, :] = np.asarray(ds1.y, dtype=float).ravel()

            # Store one representative metadata record for shared provenance.
            if sample_meta is None:
                sample_meta = dict(ds1.meta)

    # Shared provenance
    meta_final = dict(meta_out)
    meta_final["pipeline"] = pipeline.to_dict()

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

    return Pipeline(steps=steps, name="legacy_single_spectrum")

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

    return Pipeline(steps=steps, name="legacy_mapping")