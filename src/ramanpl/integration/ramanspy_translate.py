from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from ..dataImporter import DataImporter
    from ..schema import baseline_spec_to_runtime, normalise_baseline_spec
    from .ramanspy_adapter import (
        require_ramanspy,
        to_ramanspy_spectrum,
        from_ramanspy_spectrum,
        to_ramanspy_image,
        from_ramanspy_image,
    )
except Exception:  # pragma: no cover
    from ramanpl.dataImporter import DataImporter
    from ramanpl.schema import baseline_spec_to_runtime, normalise_baseline_spec
    from ramanpl.integration.ramanspy_adapter import (
        require_ramanspy,
        to_ramanspy_spectrum,
        from_ramanspy_spectrum,
        to_ramanspy_image,
        from_ramanspy_image,
    )


_SUPPORTED_SIMPLE_STEP_TYPES = {"CropByRange", "SmoothSavGol"}
_SUPPORTED_BASELINE_METHODS = {"poly", "asls", "airpls", "arpls"}


def _normalise_baseline_step_spec(step) -> Dict[str, Any]:
    return normalise_baseline_spec(
        getattr(step, "baseline_spec", {"method": "poly"}),
        poly_degree=getattr(step, "poly_degree", 3),
        gaussian_sigma=getattr(step, "gaussian_sigma", 50),
    )


def _mask_to_regions(x: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert a boolean mask on the spectral axis into contiguous RamanSPy
    region tuples for polynomial baseline fitting.
    """
    x = np.asarray(x, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    if x.shape != mask.shape:
        raise ValueError(f"x/mask shape mismatch: {x.shape} vs {mask.shape}")

    if mask.size == 0 or not np.any(mask):
        return []

    idx = np.flatnonzero(mask)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    regions: List[Tuple[float, float]] = []
    for g in groups:
        regions.append((float(x[g[0]]), float(x[g[-1]])))
    return regions


def _resolve_savgol_params(n_points: int, window_length: int, polyorder: int) -> Tuple[int, int]:
    """
    Mirror RamanPL native SmoothSavGol parameter handling so that RamanSPy
    execution stays behaviourally aligned with the native backend.
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


def _crop_mask_from_current_axis(x_current: np.ndarray, x_range) -> np.ndarray:
    mask = DataImporter.mask_by_xrange(x_current, x_range)
    if mask is None or np.sum(mask) < 3:
        raise ValueError(f"CropByRange produced too few points. data_range={x_range}")
    return np.asarray(mask, dtype=bool)


def _baseline_translation_support(step) -> Tuple[bool, str | None]:
    spec = _normalise_baseline_step_spec(step)
    method, kwargs = baseline_spec_to_runtime(spec)
    method = str(method).strip().lower()

    if method == "gaussian":
        return False, (
            "BaselineSubtract(method='gaussian') remains native-only. "
            "RamanSPy exposes Gaussian denoising, not the same baseline estimator."
        )

    if method not in _SUPPORTED_BASELINE_METHODS:
        return False, f"BaselineSubtract(method='{method}') is not yet translated."

    allowed_keys_by_method = {
        "poly": {"poly_degree", "poly_order", "mask"},
        "asls": {"lam", "p", "diff_order", "niter", "max_iter", "tol", "mask", "weights"},
        "airpls": {"lam", "diff_order", "niter", "max_iter", "tol", "mask", "weights"},
        "arpls": {"lam", "diff_order", "niter", "max_iter", "tol", "mask", "weights"},
    }

    unsupported = sorted(set(kwargs.keys()) - allowed_keys_by_method[method])
    if unsupported:
        return False, (
            f"BaselineSubtract(method='{method}') contains unsupported kwargs for "
            f"RamanSPy translation: {unsupported}"
        )

    return True, None


def pipeline_supports_ramanspy_translation(pipeline) -> Tuple[bool, List[str]]:
    """
    Return whether the given RamanPL Pipeline is translatable to RamanSPy
    in the current implementation.

    Supported in this build:
    - CropByRange
    - SmoothSavGol
    - BaselineSubtract with methods: poly, asls, airpls, arpls
    """
    if pipeline is None:
        return True, []

    unsupported: List[str] = []
    for step in getattr(pipeline, "steps", []):
        step_name = type(step).__name__

        if step_name in _SUPPORTED_SIMPLE_STEP_TYPES:
            continue

        if step_name == "BaselineSubtract":
            ok, reason = _baseline_translation_support(step)
            if not ok:
                unsupported.append(reason or "BaselineSubtract")
            continue

        unsupported.append(step_name)

    return len(unsupported) == 0, unsupported


def _build_ramanspy_baseline_step(step, x_current: np.ndarray):
    rp = require_ramanspy()

    spec = _normalise_baseline_step_spec(step)
    method, kwargs = baseline_spec_to_runtime(spec)
    method = str(method).strip().lower()

    clip_nonnegative = bool(getattr(step, "clip_nonnegative", True))
    kwargs = dict(kwargs)

    mask = kwargs.pop("mask", None)
    weights = kwargs.pop("weights", None)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.shape != x_current.shape:
            raise ValueError(
                f"Baseline mask shape {mask.shape} does not match spectral axis shape {x_current.shape}."
            )

    if method == "poly":
        poly_degree = int(
            kwargs.pop(
                "poly_degree",
                kwargs.pop("poly_order", spec.get("poly_order", 3))
            )
        )
        regions = _mask_to_regions(x_current, mask) if mask is not None else None
        rp_step = rp.preprocessing.baseline.Poly(
            poly_order=poly_degree,
            regions=regions,
        )
        runtime_kwargs_meta = {
            "poly_degree": poly_degree,
            "poly_order": poly_degree,
            "mask": mask,
        }

    elif method == "asls":
        rp_step = rp.preprocessing.baseline.ASLS(
            lam=float(kwargs.pop("lam", 1e6)),
            p=float(kwargs.pop("p", 0.01)),
            diff_order=int(kwargs.pop("diff_order", 2)),
            max_iter=int(kwargs.pop("niter", kwargs.pop("max_iter", 20))),
            tol=float(kwargs.pop("tol", 1e-3)),
            weights=weights if weights is not None else (mask.astype(float) if mask is not None else None),
        )
        runtime_kwargs_meta = {
            "lam": getattr(rp_step, "kwargs", {}).get("lam", None),
            "p": getattr(rp_step, "kwargs", {}).get("p", None),
            "diff_order": getattr(rp_step, "kwargs", {}).get("diff_order", None),
            "max_iter": getattr(rp_step, "kwargs", {}).get("max_iter", None),
            "tol": getattr(rp_step, "kwargs", {}).get("tol", None),
            "mask": mask,
        }

    elif method == "airpls":
        rp_step = rp.preprocessing.baseline.AIRPLS(
            lam=float(kwargs.pop("lam", 1e6)),
            diff_order=int(kwargs.pop("diff_order", 2)),
            max_iter=int(kwargs.pop("niter", kwargs.pop("max_iter", 50))),
            tol=float(kwargs.pop("tol", 1e-3)),
            weights=weights if weights is not None else (mask.astype(float) if mask is not None else None),
        )
        runtime_kwargs_meta = {
            "lam": getattr(rp_step, "kwargs", {}).get("lam", None),
            "diff_order": getattr(rp_step, "kwargs", {}).get("diff_order", None),
            "max_iter": getattr(rp_step, "kwargs", {}).get("max_iter", None),
            "tol": getattr(rp_step, "kwargs", {}).get("tol", None),
            "mask": mask,
        }

    elif method == "arpls":
        rp_step = rp.preprocessing.baseline.ARPLS(
            lam=float(kwargs.pop("lam", 1e6)),
            diff_order=int(kwargs.pop("diff_order", 2)),
            max_iter=int(kwargs.pop("niter", kwargs.pop("max_iter", 50))),
            tol=float(kwargs.pop("tol", 1e-6)),
            weights=weights if weights is not None else (mask.astype(float) if mask is not None else None),
        )
        runtime_kwargs_meta = {
            "lam": getattr(rp_step, "kwargs", {}).get("lam", None),
            "diff_order": getattr(rp_step, "kwargs", {}).get("diff_order", None),
            "max_iter": getattr(rp_step, "kwargs", {}).get("max_iter", None),
            "tol": getattr(rp_step, "kwargs", {}).get("tol", None),
            "mask": mask,
        }

    else:
        raise NotImplementedError(
            f"RamanSPy translation not implemented for baseline method '{method}'."
        )

    return rp_step, method, spec, clip_nonnegative, runtime_kwargs_meta


def apply_pipeline_ramanspy_single(ds, pipeline):
    """
    Apply a RamanPL Pipeline to a single Raman spectrum via RamanSPy.

    Returns
    -------
    SpectralDataset-compatible payload:
        x_out, y_out, meta_out
    """
    rp = require_ramanspy()

    x_original = np.asarray(ds.x, dtype=float).ravel()
    y_original = np.asarray(ds.y, dtype=float).ravel()

    meta = dict(ds.meta)
    x_current = x_original.copy()
    y_current = y_original.copy()

    # Track crop mask against the ORIGINAL x-grid
    orig_idx = np.arange(x_original.size, dtype=int)
    crop_history = []

    rp_obj = to_ramanspy_spectrum(
        x=x_current,
        y=y_current,
        modality=ds.modality,
        axis_kind=ds.axis_kind,
    )

    for step in pipeline.steps:
        step_name = type(step).__name__

        if step_name == "CropByRange":
            data_range = getattr(step, "data_range", None)

            if data_range is None:
                continue

            if bool(meta.get("x_trimmed_on_load", False)):
                crop_history.append({
                    "data_range": data_range,
                    "applied": False,
                    "reason": "x_trimmed_on_load",
                })
                continue

            local_mask = _crop_mask_from_current_axis(x_current, data_range)
            rp_step = rp.preprocessing.misc.Cropper(region=data_range)
            rp_obj = rp_step.apply(rp_obj)

            x_current, y_current = from_ramanspy_spectrum(rp_obj)
            orig_idx = orig_idx[local_mask]

            crop_history.append({
                "data_range": data_range,
                "applied": True,
            })

        elif step_name == "SmoothSavGol":
            w, p = _resolve_savgol_params(
                n_points=y_current.size,
                window_length=getattr(step, "window_length"),
                polyorder=getattr(step, "polyorder"),
            )
            rp_step = rp.preprocessing.denoise.SavGol(
                window_length=w,
                polyorder=p,
            )
            rp_obj = rp_step.apply(rp_obj)

            x_current, y_current = from_ramanspy_spectrum(rp_obj)
            meta["smoothing"] = {
                "method": "savgol",
                "window_length": w,
                "polyorder": p,
            }
            meta["_smoothed_last"] = np.asarray(y_current, dtype=float).ravel()

        elif step_name == "BaselineSubtract":
            y_before = y_current.copy()

            rp_step, method, spec, clip_nonnegative, runtime_kwargs_meta = _build_ramanspy_baseline_step(
                step, x_current
            )
            rp_obj = rp_step.apply(rp_obj)
            x_current, y_after = from_ramanspy_spectrum(rp_obj)

            baseline = y_before - y_after
            if clip_nonnegative:
                y_after = np.clip(y_after, a_min=0.0, a_max=None)

            y_current = np.asarray(y_after, dtype=float).ravel()
            meta["baseline"] = {
                "spec": spec,
                "resolved_method": method,
                "clip_nonnegative": clip_nonnegative,
                "kwargs": runtime_kwargs_meta,
            }
            meta["_baseline_last"] = np.asarray(baseline, dtype=float).ravel()

            rp_obj = to_ramanspy_spectrum(
                x=x_current,
                y=y_current,
                modality=ds.modality,
                axis_kind=ds.axis_kind,
            )

        else:  # pragma: no cover
            raise NotImplementedError(
                f"RamanSPy translation not implemented for step '{step_name}'."
            )

    if crop_history:
        crop_mask = np.zeros(x_original.size, dtype=bool)
        crop_mask[orig_idx] = True
        meta["crop"] = crop_history[0] if len(crop_history) == 1 else crop_history
        meta["crop_mask"] = crop_mask

    return x_current, y_current, meta


def apply_pipeline_ramanspy_mapping(
    *,
    x: np.ndarray,
    cube_yxn: np.ndarray,
    pipeline,
    modality: str,
    axis_kind: str,
    meta: Dict[str, Any] | None = None,
):
    """
    Apply a RamanPL Pipeline to a Raman mapping cube via RamanSPy.

    Parameters
    ----------
    x : ndarray, shape [N]
    cube_yxn : ndarray, shape [Y, X, N]

    Returns
    -------
    x_out : ndarray
    cube_out : ndarray, shape [Y, X, N]
    meta_out : dict
    """
    rp = require_ramanspy()

    x_original = np.asarray(x, dtype=float).ravel()
    cube_original = np.asarray(cube_yxn, dtype=float)

    if cube_original.ndim != 3:
        raise ValueError("cube_yxn must be 3D with shape [Y, X, N].")
    if cube_original.shape[2] != x_original.size:
        raise ValueError(
            f"cube_yxn.shape[2] ({cube_original.shape[2]}) must match len(x) ({x_original.size})."
        )

    meta_out: Dict[str, Any] = {} if meta is None else dict(meta)

    x_current = x_original.copy()
    cube_current = cube_original.copy()

    orig_idx = np.arange(x_original.size, dtype=int)
    crop_history = []

    rp_obj = to_ramanspy_image(
        x=x_current,
        cube_yxn=cube_current,
        modality=modality,
        axis_kind=axis_kind,
    )

    for step in pipeline.steps:
        step_name = type(step).__name__

        if step_name == "CropByRange":
            data_range = getattr(step, "data_range", None)

            if data_range is None:
                continue

            if bool(meta_out.get("x_trimmed_on_load", False)):
                crop_history.append({
                    "data_range": data_range,
                    "applied": False,
                    "reason": "x_trimmed_on_load",
                })
                continue

            local_mask = _crop_mask_from_current_axis(x_current, data_range)
            rp_step = rp.preprocessing.misc.Cropper(region=data_range)
            rp_obj = rp_step.apply(rp_obj)

            x_current, cube_current = from_ramanspy_image(rp_obj)
            orig_idx = orig_idx[local_mask]

            crop_history.append({
                "data_range": data_range,
                "applied": True,
            })

        elif step_name == "SmoothSavGol":
            w, p = _resolve_savgol_params(
                n_points=x_current.size,
                window_length=getattr(step, "window_length"),
                polyorder=getattr(step, "polyorder"),
            )
            rp_step = rp.preprocessing.denoise.SavGol(
                window_length=w,
                polyorder=p,
            )
            rp_obj = rp_step.apply(rp_obj)

            x_current, cube_current = from_ramanspy_image(rp_obj)
            meta_out["smoothing"] = {
                "method": "savgol",
                "window_length": w,
                "polyorder": p,
            }
            meta_out["_smoothed_last"] = np.asarray(cube_current[0, 0, :], dtype=float).ravel()

        elif step_name == "BaselineSubtract":
            cube_before = cube_current.copy()

            rp_step, method, spec, clip_nonnegative, runtime_kwargs_meta = _build_ramanspy_baseline_step(
                step, x_current
            )
            rp_obj = rp_step.apply(rp_obj)
            x_current, cube_after = from_ramanspy_image(rp_obj)

            baseline = cube_before - cube_after
            if clip_nonnegative:
                cube_after = np.clip(cube_after, a_min=0.0, a_max=None)

            cube_current = np.asarray(cube_after, dtype=float)
            meta_out["baseline"] = {
                "spec": spec,
                "resolved_method": method,
                "clip_nonnegative": clip_nonnegative,
                "kwargs": runtime_kwargs_meta,
            }
            meta_out["_baseline_last"] = np.asarray(baseline[0, 0, :], dtype=float).ravel()

            rp_obj = to_ramanspy_image(
                x=x_current,
                cube_yxn=cube_current,
                modality=modality,
                axis_kind=axis_kind,
            )

        else:  # pragma: no cover
            raise NotImplementedError(
                f"RamanSPy translation not implemented for step '{step_name}'."
            )

    if crop_history:
        crop_mask = np.zeros(x_original.size, dtype=bool)
        crop_mask[orig_idx] = True
        meta_out["crop"] = crop_history[0] if len(crop_history) == 1 else crop_history
        meta_out["crop_mask"] = crop_mask

    meta_out["pipeline_steps_applied"] = [
        s.to_dict() if hasattr(s, "to_dict") else {"name": type(s).__name__}
        for s in pipeline.steps
    ]

    return x_current, cube_current, meta_out