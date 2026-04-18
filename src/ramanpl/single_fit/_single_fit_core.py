from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from ..schema import normalise_baseline_spec


def make_compat_data_importer(axis: str, default_readlines=(300, 780)):
    """
    Compatibility shim so old calls such as:
        RamanFit.DataImporter.data_import(...)
        PLfit.DataImporter.data_import(...)
    still work while delegating to the shared importer.
    """
    class _CompatDataImporter:
        @staticmethod
        def data_import(filename, readlines=None, x_range=None):
            from ramanpl.dataImporter import DataImporter as _Shared

            if x_range is None and readlines is None and default_readlines is not None:
                readlines = default_readlines

            return _Shared.data_import(
                filename=filename,
                readlines=readlines,
                x_range=x_range,
                axis=axis,
            )

    _CompatDataImporter.__name__ = "DataImporter"
    _CompatDataImporter.__qualname__ = "DataImporter"
    return _CompatDataImporter


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(r * r)))


def params_at_bounds(
    params: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> np.ndarray:
    params = np.asarray(params, dtype=float).ravel()
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()

    on_lb = np.isclose(params, lb, rtol=rtol, atol=atol)
    on_ub = np.isclose(params, ub, rtol=rtol, atol=atol)
    return on_lb | on_ub


def generate_p0_trials(
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    base_p0: np.ndarray,
    n_starts: int,
    strategy: str,
    random_state=None,
) -> list[np.ndarray]:
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    base_p0 = np.asarray(base_p0, dtype=float).ravel()

    if n_starts is None:
        n_starts = 1
    n_starts = max(int(n_starts), 1)

    strategy = (strategy or "midpoint").lower()
    if strategy not in {"midpoint", "random", "jitter"}:
        raise ValueError("p0_strategy must be one of: 'midpoint', 'random', 'jitter'.")

    trials = [base_p0.copy()]
    if n_starts == 1:
        return trials

    rng = np.random.default_rng(random_state)
    m = n_starts - 1

    if strategy == "random":
        for _ in range(m):
            trials.append(rng.uniform(lb, ub))
        return trials

    # midpoint: jitter around the bound midpoint (lb+ub)/2
    # jitter:   jitter around the user-supplied base_p0
    anchor = 0.5 * (lb + ub) if strategy == "midpoint" else base_p0

    scale = 0.10 * (ub - lb)
    scale = np.where(scale > 0, scale, 1.0)

    for _ in range(m):
        p = anchor + rng.normal(loc=0.0, scale=scale)
        trials.append(np.clip(p, lb, ub))

    return trials


def run_multistart_curve_fit(
    *,
    model,
    x: np.ndarray,
    y: np.ndarray,
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
    base_p0: Sequence[float],
    n_starts: int = 1,
    p0_strategy: str = "midpoint",
    random_state=None,
    maxfev: int = 6400,
    jac=None,
    diagnose_bounds: bool = True,
    bounds_tol: float = 1e-6,
    fail_label: str = "single-spectrum fitter",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Shared multi-start bounded curve_fit runner for single-spectrum fitters.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    lb = np.asarray(lower_bound, dtype=float).ravel()
    ub = np.asarray(upper_bound, dtype=float).ravel()
    base_p0 = np.asarray(base_p0, dtype=float).ravel()

    p0_trials = generate_p0_trials(
        lb,
        ub,
        base_p0=base_p0,
        n_starts=n_starts,
        strategy=p0_strategy,
        random_state=random_state,
    )

    best_params = None
    best_cov = None
    best_rmse = np.inf
    best_p0 = None
    n_fail = 0
    last_exception = None

    for p0 in p0_trials:
        try:
            params, params_cov = optimize.curve_fit(
                model,
                x,
                y,
                p0=p0,
                bounds=(lb, ub),
                maxfev=int(maxfev),
                jac=jac,
            )
        except Exception as e:
            n_fail += 1
            last_exception = e
            continue

        y_hat = model(x, *params)
        score = rmse(y, y_hat)

        if score < best_rmse:
            best_rmse = score
            best_params = params
            best_cov = params_cov
            best_p0 = p0

    if best_params is None:
        if last_exception is not None:
            raise RuntimeError(
                f"{fail_label} failed for all starts (n_starts={int(n_starts)}). "
                f"Last exception: {type(last_exception).__name__}: {last_exception}"
            ) from last_exception
        raise RuntimeError(
            f"{fail_label} failed for all starts (n_starts={int(n_starts)}). "
            "Check bounds/preprocessing, or reduce model complexity."
        )

    diagnostics: Dict[str, Any] = {
        "rmse": float(best_rmse),
        "n_starts": int(n_starts),
        "n_fail": int(n_fail),
        "p0_strategy": str(p0_strategy),
        "best_p0": np.asarray(best_p0, dtype=float),
    }

    if diagnose_bounds:
        at_bounds = params_at_bounds(
            np.asarray(best_params, dtype=float),
            lb,
            ub,
            rtol=float(bounds_tol),
        )
        diagnostics["n_params_at_bounds"] = int(np.count_nonzero(at_bounds))
        diagnostics["params_at_bounds_mask"] = at_bounds

    return best_params, best_cov, diagnostics

def build_single_fit_export_metadata(
    obj,
    *,
    include_legacy: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Shared export metadata builder for RamanFit / PLfit.
    """
    from ..exporter import serialise_backend_provenance

    meta = {
        "schema_version": "0.3.8",
        "spectrum_type": getattr(obj, "spectrum_type", None),
        "x_quantity": getattr(obj, "x_quantity", None),
        "x_unit": getattr(obj, "x_unit", None),
        "normalize": getattr(obj, "normalize", None),
        "intensity_scale": getattr(obj, "peak_intensity", None),
        "peak_labels": getattr(obj, "peak_labels", None),
        "custom_peaks": bool(getattr(obj, "custom_peaks", None) is not None),
        "remove_peaks": getattr(obj, "remove_peaks_list", None),
        "peak_profile": getattr(obj, "peak_profile", None),
        "params_per_peak": getattr(obj, "params_per_peak", None),
        "baseline_spec": getattr(obj, "baseline_method", None),
    }

    pp = getattr(obj, "preprocessing_recipe", None)
    if pp is None:
        raw_pp = getattr(obj, "preprocessing", None)
        if raw_pp is not None and hasattr(raw_pp, "to_dict") and callable(getattr(raw_pp, "to_dict")):
            try:
                pp = raw_pp.to_dict()
            except Exception:
                pp = str(raw_pp)
        else:
            pp = raw_pp

    if pp is not None:
        meta["preprocessing_recipe"] = pp

    # Backend provenance: use canonical serialiser when _backend_outcome is available,
    # otherwise fall back to direct attribute reading for older fitter objects.
    backend_outcome = getattr(obj, "_backend_outcome", None) or getattr(obj, "preprocessing_backend_info", None)
    if backend_outcome:
        meta.update(serialise_backend_provenance(backend_outcome))
    else:
        requested = getattr(obj, "preprocessing_backend", None)
        resolved = getattr(obj, "preprocessing_backend_resolved", None)
        if requested is not None:
            meta["preprocessing_backend_requested"] = requested
        if resolved is not None:
            meta["preprocessing_backend_resolved"] = resolved

    if extra_meta:
        meta.update(extra_meta)

    if include_legacy:
        meta.update({
            "background_remove": getattr(obj, "background_remove", None),
            "baseline_method": getattr(obj, "baseline_method", None),
            "gaussian_sigma": getattr(obj, "gaussian_sigma", None),
            "smoothing": getattr(obj, "smoothing", None),
            "smooth_window": getattr(obj, "smooth_window", None),
            "smooth_order": getattr(obj, "smooth_order", None),
            "intensity_scale(peak_intensity)": getattr(obj, "peak_intensity", None),
        })

    return {k: v for k, v in meta.items() if v is not None}


def export_p0_dict(obj) -> dict:
    if not hasattr(obj, "params_fit") or obj.params_fit is None:
        raise ValueError("No fitted parameters found. Run fit_spectrum() first.")

    return {
        "p0": np.asarray(obj.params_fit, dtype=float).copy(),
        "peak_order": list(getattr(obj, "peak_labels", [])),
    }