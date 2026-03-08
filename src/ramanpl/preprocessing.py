"""
preprocessing.py

Minimal preprocessing framework for RamanPL_2D (v0.3.4 target).

Design goals:
- Single-spectrum first (1D arrays), mapping support later (v0.3.5).
- Modality-aware presets can be built on top without forcing PL to over-process.
- Keep operations explicit, composable, and reproducible (serialisable step specs).

This module does NOT change existing behaviour until PLfit/RamanFit start using it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter

# Local imports (package-safe)
try:
    from .baselineAPI import BaselineAPI
    from .dataImporter import DataImporter
except Exception:  # pragma: no cover (fallback for running as a script)
    from baselineAPI import BaselineAPI
    from dataImporter import DataImporter


ArrayLike1D = Union[np.ndarray, Sequence[float]]
Range2 = Optional[Tuple[float, float]]
BaselineSpec = Union[str, Dict[str, Any]]

@dataclass(frozen=True)
class SpectralDataset:
    """
    Canonical in-memory container for preprocessing.

    Parameters
    ----------
    x
        Spectral axis (1D). Raman: wavenumber (cm^-1). PL: energy (eV) or wavelength (nm).
    y
        Intensity array. For v0.3.4 single-spectrum pipeline this is 1D.
    modality
        "Raman" or "PL" (free string for now; later we can make Literal).
    axis_kind
        "raman_shift_cm-1" | "energy_eV" | "wavelength_nm" (free string for now).
    meta
        Metadata/provenance dict (e.g. filename, acquisition settings, flags).
    """
    x: np.ndarray
    y: np.ndarray
    modality: str
    axis_kind: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **kwargs) -> "SpectralDataset":
        d = {
            "x": self.x,
            "y": self.y,
            "modality": self.modality,
            "axis_kind": self.axis_kind,
            "meta": self.meta,
        }
        d.update(kwargs)
        # Copy meta defensively if being replaced/edited
        if "meta" in kwargs:
            d["meta"] = dict(kwargs["meta"])
        return SpectralDataset(**d)


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

    Supports:
    - baseline_spec as a string method name: "poly", "asls", "arpls", "airpls", ...
    - baseline_spec as a dict, e.g. {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}
      In this case, dict values override defaults.
    """
    baseline_spec: BaselineSpec = "poly"
    poly_degree: int = 3
    gaussian_sigma: int = 50
    clip_nonnegative: bool = True
    name: str = "baseline"

    def apply(self, ds: SpectralDataset) -> SpectralDataset:
        x = np.asarray(ds.x, dtype=float).ravel()
        y = np.asarray(ds.y, dtype=float).ravel()

        # Preserve dict specs (do NOT str(...) them)
        spec = self.baseline_spec

        # Allow dict to override defaults cleanly
        if isinstance(spec, dict):
            spec = dict(spec)  # defensive copy
            # If user provided explicit parameters, prefer them over defaults
            poly_degree = int(spec.get("poly_degree", self.poly_degree))
            gaussian_sigma = int(spec.get("gaussian_sigma", self.gaussian_sigma))
        else:
            poly_degree = int(self.poly_degree)
            gaussian_sigma = int(self.gaussian_sigma)

        method, bkwargs = BaselineAPI.parse_spec(
            spec,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma,
        )

        result = BaselineAPI.subtract(
            x=x,
            y=y,
            method=method,
            clip_nonnegative=bool(self.clip_nonnegative),
            **bkwargs,
        )

        meta = dict(ds.meta)
        meta["baseline"] = {
            "spec": spec,  # may be str or dict
            "resolved_method": method,
            "poly_degree": poly_degree,
            "gaussian_sigma": gaussian_sigma,
            "clip_nonnegative": bool(self.clip_nonnegative),
            "kwargs": dict(bkwargs),
        }
        meta["_baseline_last"] = np.asarray(result.baseline, dtype=float).ravel()

        return ds.copy_with(y=np.asarray(result.y_corrected, dtype=float).ravel(), meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "baseline_spec": self.baseline_spec,
            "poly_degree": int(self.poly_degree),
            "gaussian_sigma": int(self.gaussian_sigma),
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