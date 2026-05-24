"""
Row-band parallelism helpers for RamanMapping and PLMapping (v0.6.4).

All band-worker functions are module-level so they can be pickled by the
joblib loky backend.  No function in this module holds a reference to
any mapping object.
"""
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Row-band utilities
# ---------------------------------------------------------------------------

def _split_rows(Y, n_jobs):
    """Return list of (j_start, j_end) row-band slices for n_jobs workers."""
    actual = min(n_jobs, Y)
    base, rem = divmod(Y, actual)
    bands = []
    j = 0
    for k in range(actual):
        end = j + base + (1 if k < rem else 0)
        bands.append((j, end))
        j = end
    return bands


def _validate_parallel_kwargs(n_jobs, warm_start, row_reset, Y):
    """Validate n_jobs and parallel-safety rules. Returns effective n_jobs."""
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, got {n_jobs!r}")
    if n_jobs > 1 and warm_start and not row_reset:
        raise ValueError(
            "n_jobs > 1 with warm_start=True requires row_reset=True. "
            "Either set row_reset=True or use n_jobs=1."
        )
    if n_jobs > Y:
        warnings.warn(
            f"n_jobs={n_jobs} exceeds the number of mapping rows Y={Y}; "
            f"clamping to {Y}.",
            UserWarning,
            stacklevel=3,
        )
        return Y
    return n_jobs


def _merge_band_outputs(mapping_obj, fitted_params, bands, band_results):
    """Write per-band results back into the parent mapping_obj arrays."""
    for (j_start, j_end), result in zip(bands, band_results):
        fitted_params[j_start:j_end] = result["fitted_params"]
        mapping_obj.residual_map[j_start:j_end] = result["residual_map"]
        mapping_obj.norm_scale_map[j_start:j_end] = result["norm_scale_map"]
        mapping_obj.peak_positions[j_start:j_end] = result["peak_positions"]
        mapping_obj.peak_intensities[j_start:j_end] = result["peak_intensities"]
        if mapping_obj.fit_diagnostics_map is not None and result.get("diagnostics") is not None:
            mapping_obj.fit_diagnostics_map[j_start:j_end] = result["diagnostics"]
        for key in ("Peaks_distance", "ratio_A1g_E2g", "ratio_E2g_A1g"):
            if key in result and hasattr(mapping_obj, key):
                getattr(mapping_obj, key)[j_start:j_end] = result[key]


# ---------------------------------------------------------------------------
# Raman band worker (no reference to self)
# ---------------------------------------------------------------------------

def _raman_fit_band(j_start, j_end, band_cube, xdata, cfg):
    """
    Module-level worker for parallel Raman row-band fitting.
    All inputs are arrays or scalars — no mapping object reference.
    Returns a dict of band-local output arrays indexed by local row (0-based).
    """
    import numpy as np
    from ramanpl.mapping._fit_utils import _run_mapping_curve_fit_trials, _params_at_bounds
    from ramanpl.peak_models import sum_peaks, single_peak

    X = cfg["X"]
    n_params = cfg["n_params"]
    n_peaks = cfg["n_peaks"]
    idx_a1g = cfg["idx_a1g"]
    idx_e2g = cfg["idx_e2g"]
    p0_base = cfg["p0_base"]
    p0_start = cfg["p0_start"]
    peak_profile = cfg["peak_profile"]
    stride = cfg["stride"]
    normalize = cfg["normalize"]
    warm_start = cfg["warm_start"]
    row_reset = cfg["row_reset"]
    warm_start_rmse_gate = cfg["warm_start_rmse_gate"]
    reset_on_fail = cfg["reset_on_fail"]
    fit_normalize = cfg["fit_normalize"]
    maxfev = cfg["maxfev"]
    lower_bound = cfg["lower_bound"]
    upper_bound = cfg["upper_bound"]
    adaptive_multistart = cfg["adaptive_multistart"]
    fast_n_starts = cfg["fast_n_starts"]
    fast_p0_strategy = cfg["fast_p0_strategy"]
    fast_random_state = cfg["fast_random_state"]
    fallback_n_starts = cfg["fallback_n_starts"]
    fallback_p0_strategy = cfg["fallback_p0_strategy"]
    fallback_random_state = cfg["fallback_random_state"]
    retry_on_fail = cfg["retry_on_fail"]
    retry_on_high_rmse = cfg["retry_on_high_rmse"]
    retry_on_bound_hit = cfg["retry_on_bound_hit"]
    retry_rmse_gate = cfg["retry_rmse_gate"]
    width_penalty = cfg["width_penalty"]
    prefer_nonbound = cfg["prefer_nonbound"]
    score_tie_tol = cfg["score_tie_tol"]
    diagnostics_mode = cfg["diagnostics_mode"]

    if peak_profile == "lorentzian":
        def model_fn(x, *params):
            return sum_peaks(np.asarray(x), params, profile="lorentzian", stride=3)
    else:
        def model_fn(x, *params):
            return sum_peaks(np.asarray(x), params, profile="pvoigt", stride=4)

    n_rows = j_end - j_start
    fitted_params_band = np.full((n_rows, X, n_params), np.nan)
    residual_map_band = np.full((n_rows, X), np.nan)
    norm_scale_map_band = np.full((n_rows, X), np.nan)
    peak_positions_band = np.full((n_rows, X, n_peaks), np.nan)
    peak_intensities_band = np.full((n_rows, X, n_peaks), np.nan)
    Peaks_distance_band = np.full((n_rows, X), np.nan)
    ratio_A1g_E2g_band = np.full((n_rows, X), np.nan)
    ratio_E2g_A1g_band = np.full((n_rows, X), np.nan)
    diagnostics_band = np.empty((n_rows, X), dtype=object)
    diagnostics_band[:, :] = None

    def _store_diag(jj, i, payload):
        if diagnostics_mode == "none":
            return
        if diagnostics_mode == "full":
            diagnostics_band[jj, i] = payload
            return
        light = {"ok": payload.get("ok", None)}
        for key in ("reason", "rmse", "n_starts", "n_fail", "p0_strategy",
                    "adaptive_retry_used", "n_params_at_lower_bounds",
                    "n_params_at_upper_bounds"):
            if key in payload:
                light[key] = payload[key]
        diagnostics_band[jj, i] = light

    def _params_plausible(params, lb, ub, n_peaks, stride, profile, tol=1e-10):
        params = np.asarray(params, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        for k in range(n_peaks):
            base = stride * k
            c = params[base + 0]
            w = params[base + 1]
            a = params[base + 2]
            if (not np.isfinite(w)) or w <= 1e-8:
                return False
            if abs(c - lb[base + 0]) < tol or abs(c - ub[base + 0]) < tol:
                return False
            if abs(w - lb[base + 1]) < tol or abs(w - ub[base + 1]) < tol:
                return False
            if abs(a - lb[base + 2]) < tol or abs(a - ub[base + 2]) < tol:
                return False
            if profile == "pvoigt":
                eta = params[base + 3]
                if (not np.isfinite(eta)) or eta < -1e-6 or eta > 1 + 1e-6:
                    return False
                if abs(eta - lb[base + 3]) < tol or abs(eta - ub[base + 3]) < tol:
                    return False
        return True

    p0_current = p0_start.copy()
    for jj in range(n_rows):
        if warm_start and row_reset:
            p0_current = p0_base.copy()

        for i in range(X):
            y = np.asarray(band_cube[jj, i, :], dtype=float)
            x = xdata

            scale_val = np.nanmax(y)
            if (not np.isfinite(scale_val)) or scale_val <= 0:
                norm_scale_map_band[jj, i] = np.nan
                residual_map_band[jj, i] = np.nan
                _store_diag(jj, i, {"ok": False, "reason": "no_positive_signal"})
                if reset_on_fail:
                    p0_current = p0_base.copy()
                continue

            scale = float(scale_val)
            norm_scale_map_band[jj, i] = scale
            spec_fit = y / scale if fit_normalize else y

            retry_used = False
            retry_result = None

            if adaptive_multistart:
                quick_result = _run_mapping_curve_fit_trials(
                    model_fn=model_fn,
                    x=x,
                    y=spec_fit,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    p0_current=p0_current,
                    maxfev=maxfev,
                    n_starts=fast_n_starts,
                    p0_strategy=fast_p0_strategy,
                    random_state=fast_random_state,
                    width_penalty=width_penalty,
                    prefer_nonbound=prefer_nonbound,
                    score_tie_tol=score_tie_tol,
                    peak_profile=peak_profile,
                    stride=stride,
                )

                fit_result = quick_result
                need_retry = False
                if not quick_result["ok"]:
                    need_retry = retry_on_fail
                else:
                    if retry_on_high_rmse and quick_result["best_rmse"] > retry_rmse_gate:
                        need_retry = True
                    if retry_on_bound_hit and quick_result["best_hits"] > 0:
                        need_retry = True

                if need_retry and fallback_n_starts > 1:
                    retry_used = True
                    retry_result = _run_mapping_curve_fit_trials(
                        model_fn=model_fn,
                        x=x,
                        y=spec_fit,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        p0_current=p0_current,
                        maxfev=maxfev,
                        n_starts=fallback_n_starts,
                        p0_strategy=fallback_p0_strategy,
                        random_state=fallback_random_state,
                        width_penalty=width_penalty,
                        prefer_nonbound=prefer_nonbound,
                        score_tie_tol=score_tie_tol,
                        peak_profile=peak_profile,
                        stride=stride,
                    )

                    if retry_result["ok"]:
                        if not fit_result["ok"]:
                            fit_result = retry_result
                        else:
                            better = retry_result["best_score"] < fit_result["best_score"]
                            near_tie = abs(retry_result["best_score"] - fit_result["best_score"]) <= score_tie_tol
                            if better or (prefer_nonbound and near_tie
                                          and retry_result["best_hits"] < fit_result["best_hits"]):
                                fit_result = retry_result
                    elif not fit_result["ok"]:
                        fit_result = retry_result

                n_fail_total = quick_result["n_fail"] + (
                    retry_result["n_fail"] if retry_result is not None else 0
                )
                n_starts_total = fast_n_starts + (fallback_n_starts if retry_used else 0)

            else:
                fit_result = _run_mapping_curve_fit_trials(
                    model_fn=model_fn,
                    x=x,
                    y=spec_fit,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    p0_current=p0_current,
                    maxfev=maxfev,
                    n_starts=fallback_n_starts,
                    p0_strategy=fallback_p0_strategy,
                    random_state=fallback_random_state,
                    width_penalty=width_penalty,
                    prefer_nonbound=prefer_nonbound,
                    score_tie_tol=score_tie_tol,
                    peak_profile=peak_profile,
                    stride=stride,
                )
                n_fail_total = fit_result["n_fail"]
                n_starts_total = fit_result["n_starts"]

            if not fit_result["ok"]:
                residual_map_band[jj, i] = np.nan
                fitted_params_band[jj, i, :] = np.nan
                _store_diag(jj, i, {
                    "ok": False,
                    "n_starts": n_starts_total,
                    "n_fail": n_fail_total,
                    "p0_strategy": ("adaptive" if adaptive_multistart else fallback_p0_strategy),
                    "adaptive_retry_used": bool(retry_used),
                })
                if reset_on_fail:
                    p0_current = p0_base.copy()
                continue

            best_params = fit_result["best_params"]
            best_rmse = fit_result["best_rmse"]
            best_p0 = fit_result["best_p0"]

            residual_map_band[jj, i] = best_rmse
            fitted_params_band[jj, i, :] = best_params
            at_lo = _params_at_bounds(best_params, lower_bound, upper_bound, which="lower", rtol=1e-6)
            at_hi = _params_at_bounds(best_params, lower_bound, upper_bound, which="upper", rtol=1e-6)

            payload = {
                "ok": True,
                "rmse": best_rmse,
                "n_starts": n_starts_total,
                "n_fail": n_fail_total,
                "p0_strategy": ("adaptive" if adaptive_multistart else fallback_p0_strategy),
                "adaptive_retry_used": bool(retry_used),
                "n_params_at_lower_bounds": int(np.count_nonzero(at_lo)),
                "n_params_at_upper_bounds": int(np.count_nonzero(at_hi)),
            }

            if diagnostics_mode == "full":
                payload["best_p0"] = np.asarray(best_p0, dtype=float)
                payload["params_at_lower_bounds_mask"] = at_lo
                payload["params_at_upper_bounds_mask"] = at_hi

            _store_diag(jj, i, payload)

            for k in range(n_peaks):
                block = np.asarray(best_params[stride*k:stride*(k+1)], dtype=float)
                center = float(block[0])
                peak_positions_band[jj, i, k] = center

                if peak_profile == "lorentzian":
                    width = float(block[1])
                    amp = float(block[2])
                    if (not np.isfinite(width)) or width <= 0:
                        height_fitspace = np.nan
                    else:
                        height_fitspace = amp / (np.pi * width)
                else:
                    y_one = single_peak(xdata, block, profile="pvoigt")
                    height_fitspace = float(np.nanmax(y_one))

                if normalize:
                    peak_intensities_band[jj, i, k] = height_fitspace
                else:
                    if fit_normalize:
                        peak_intensities_band[jj, i, k] = height_fitspace * scale
                    else:
                        peak_intensities_band[jj, i, k] = height_fitspace

            if (idx_a1g is not None) and (idx_e2g is not None):
                a1g_pos = peak_positions_band[jj, i, idx_a1g]
                e2g_pos = peak_positions_band[jj, i, idx_e2g]
                a1g_I = peak_intensities_band[jj, i, idx_a1g]
                e2g_I = peak_intensities_band[jj, i, idx_e2g]
                Peaks_distance_band[jj, i] = (
                    (a1g_pos - e2g_pos)
                    if (np.isfinite(a1g_pos) and np.isfinite(e2g_pos))
                    else np.nan
                )
                ratio_A1g_E2g_band[jj, i] = (
                    (a1g_I / e2g_I)
                    if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and e2g_I > 0)
                    else np.nan
                )
                ratio_E2g_A1g_band[jj, i] = (
                    (e2g_I / a1g_I)
                    if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and a1g_I > 0)
                    else np.nan
                )
            else:
                Peaks_distance_band[jj, i] = np.nan
                ratio_A1g_E2g_band[jj, i] = np.nan
                ratio_E2g_A1g_band[jj, i] = np.nan

            if warm_start:
                ok_rmse = (best_rmse <= warm_start_rmse_gate)
                ok_params = _params_plausible(
                    best_params, lower_bound, upper_bound, n_peaks, stride, peak_profile
                )
                if ok_rmse and ok_params:
                    p0_current = np.asarray(best_params, dtype=float)
                else:
                    if reset_on_fail:
                        p0_current = p0_base.copy()

    return {
        "fitted_params": fitted_params_band,
        "residual_map": residual_map_band,
        "norm_scale_map": norm_scale_map_band,
        "peak_positions": peak_positions_band,
        "peak_intensities": peak_intensities_band,
        "Peaks_distance": Peaks_distance_band,
        "ratio_A1g_E2g": ratio_A1g_E2g_band,
        "ratio_E2g_A1g": ratio_E2g_A1g_band,
        "diagnostics": diagnostics_band if diagnostics_mode != "none" else None,
    }


# ---------------------------------------------------------------------------
# PL band worker (no reference to self)
# ---------------------------------------------------------------------------

def _pl_fit_band(j_start, j_end, band_cube, xdata, cfg):
    """
    Module-level worker for parallel PL row-band fitting.
    All inputs are arrays or scalars — no mapping object reference.
    Returns a dict of band-local output arrays indexed by local row (0-based).
    """
    import numpy as np
    from ramanpl.mapping._fit_utils import _run_mapping_curve_fit_trials, _params_at_bounds
    from ramanpl.peak_models import sum_peaks, single_peak

    X = cfg["X"]
    n_params = cfg["n_params"]
    n_peaks = cfg["n_peaks"]
    p0_base = cfg["p0_base"]
    p0_start = cfg["p0_start"]
    peak_profile = cfg["peak_profile"]
    stride = cfg["stride"]
    normalize = cfg["normalize"]
    warm_start = cfg["warm_start"]
    row_reset = cfg["row_reset"]
    warm_start_rmse_gate = cfg["warm_start_rmse_gate"]
    reset_on_fail = cfg["reset_on_fail"]
    fit_normalize = cfg["fit_normalize"]
    maxfev = cfg["maxfev"]
    lower_bound = cfg["lower_bound"]
    upper_bound = cfg["upper_bound"]
    adaptive_multistart = cfg["adaptive_multistart"]
    fast_n_starts = cfg["fast_n_starts"]
    fast_p0_strategy = cfg["fast_p0_strategy"]
    fast_random_state = cfg["fast_random_state"]
    fallback_n_starts = cfg["fallback_n_starts"]
    fallback_p0_strategy = cfg["fallback_p0_strategy"]
    fallback_random_state = cfg["fallback_random_state"]
    retry_on_fail = cfg["retry_on_fail"]
    retry_on_high_rmse = cfg["retry_on_high_rmse"]
    retry_on_bound_hit = cfg["retry_on_bound_hit"]
    retry_rmse_gate = cfg["retry_rmse_gate"]
    width_penalty = cfg["width_penalty"]
    prefer_nonbound = cfg["prefer_nonbound"]
    score_tie_tol = cfg["score_tie_tol"]
    diagnostics_mode = cfg["diagnostics_mode"]

    if peak_profile == "lorentzian":
        def model_fn(x, *params):
            return sum_peaks(np.asarray(x), params, profile="lorentzian", stride=3)
    else:
        def model_fn(x, *params):
            return sum_peaks(np.asarray(x), params, profile="pvoigt", stride=4)

    n_rows = j_end - j_start
    fitted_params_band = np.full((n_rows, X, n_params), np.nan)
    residual_map_band = np.full((n_rows, X), np.nan)
    norm_scale_map_band = np.full((n_rows, X), np.nan)
    peak_positions_band = np.full((n_rows, X, n_peaks), np.nan)
    peak_intensities_band = np.full((n_rows, X, n_peaks), np.nan)
    diagnostics_band = np.empty((n_rows, X), dtype=object)
    diagnostics_band[:, :] = None

    def _store_diag(jj, i, payload):
        if diagnostics_mode == "none":
            return
        if diagnostics_mode == "full":
            diagnostics_band[jj, i] = payload
            return
        light = {"ok": payload.get("ok", None)}
        for key in ("reason", "rmse", "n_starts", "n_fail", "p0_strategy",
                    "adaptive_retry_used", "n_params_at_lower_bounds",
                    "n_params_at_upper_bounds"):
            if key in payload:
                light[key] = payload[key]
        diagnostics_band[jj, i] = light

    p0_current = p0_start.copy()
    for jj in range(n_rows):
        for i in range(X):
            y = np.asarray(band_cube[jj, i, :], dtype=float)
            x = xdata

            scale_val = np.nanmax(y)
            if (not np.isfinite(scale_val)) or scale_val <= 0:
                norm_scale_map_band[jj, i] = np.nan
                residual_map_band[jj, i] = np.nan
                _store_diag(jj, i, {"ok": False, "reason": "no_positive_signal"})
                if reset_on_fail:
                    p0_current = p0_base.copy()
                continue

            s = float(scale_val)
            norm_scale_map_band[jj, i] = s
            y_fitspace = y / s if fit_normalize else y

            retry_used = False
            retry_result = None

            if adaptive_multistart:
                quick_result = _run_mapping_curve_fit_trials(
                    model_fn=model_fn,
                    x=x,
                    y=y_fitspace,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    p0_current=p0_current,
                    maxfev=maxfev,
                    n_starts=fast_n_starts,
                    p0_strategy=fast_p0_strategy,
                    random_state=fast_random_state,
                    width_penalty=width_penalty,
                    prefer_nonbound=prefer_nonbound,
                    score_tie_tol=score_tie_tol,
                    peak_profile=peak_profile,
                    stride=stride,
                )

                fit_result = quick_result
                need_retry = False
                if not quick_result["ok"]:
                    need_retry = retry_on_fail
                else:
                    if retry_on_high_rmse and quick_result["best_rmse"] > retry_rmse_gate:
                        need_retry = True
                    if retry_on_bound_hit and quick_result["best_hits"] > 0:
                        need_retry = True

                if need_retry and fallback_n_starts > 1:
                    retry_used = True
                    retry_result = _run_mapping_curve_fit_trials(
                        model_fn=model_fn,
                        x=x,
                        y=y_fitspace,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        p0_current=p0_current,
                        maxfev=maxfev,
                        n_starts=fallback_n_starts,
                        p0_strategy=fallback_p0_strategy,
                        random_state=fallback_random_state,
                        width_penalty=width_penalty,
                        prefer_nonbound=prefer_nonbound,
                        score_tie_tol=score_tie_tol,
                        peak_profile=peak_profile,
                        stride=stride,
                    )

                    if retry_result["ok"]:
                        if not fit_result["ok"]:
                            fit_result = retry_result
                        else:
                            better = retry_result["best_score"] < fit_result["best_score"]
                            near_tie = abs(retry_result["best_score"] - fit_result["best_score"]) <= score_tie_tol
                            if better or (prefer_nonbound and near_tie
                                          and retry_result["best_hits"] < fit_result["best_hits"]):
                                fit_result = retry_result
                    elif not fit_result["ok"]:
                        fit_result = retry_result

                n_fail_total = quick_result["n_fail"] + (
                    retry_result["n_fail"] if retry_result is not None else 0
                )
                n_starts_total = fast_n_starts + (fallback_n_starts if retry_used else 0)

            else:
                fit_result = _run_mapping_curve_fit_trials(
                    model_fn=model_fn,
                    x=x,
                    y=y_fitspace,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    p0_current=p0_current,
                    maxfev=maxfev,
                    n_starts=fallback_n_starts,
                    p0_strategy=fallback_p0_strategy,
                    random_state=fallback_random_state,
                    width_penalty=width_penalty,
                    prefer_nonbound=prefer_nonbound,
                    score_tie_tol=score_tie_tol,
                    peak_profile=peak_profile,
                    stride=stride,
                )
                n_fail_total = fit_result["n_fail"]
                n_starts_total = fit_result["n_starts"]

            if not fit_result["ok"]:
                fitted_params_band[jj, i, :] = np.nan
                residual_map_band[jj, i] = np.nan
                _store_diag(jj, i, {
                    "ok": False,
                    "n_starts": n_starts_total,
                    "n_fail": n_fail_total,
                    "p0_strategy": ("adaptive" if adaptive_multistart else fallback_p0_strategy),
                    "adaptive_retry_used": bool(retry_used),
                })
                if reset_on_fail:
                    p0_current = p0_base.copy()
                continue

            best_params = fit_result["best_params"]
            best_rmse = fit_result["best_rmse"]
            best_p0 = fit_result["best_p0"]

            if best_params is None:
                fitted_params_band[jj, i, :] = np.nan
                residual_map_band[jj, i] = np.nan
                _store_diag(jj, i, {
                    "ok": False,
                    "n_starts": n_starts_total,
                    "n_fail": n_fail_total,
                    "p0_strategy": ("adaptive" if adaptive_multistart else fallback_p0_strategy),
                    "adaptive_retry_used": bool(retry_used),
                })
                if reset_on_fail:
                    p0_current = p0_base.copy()
                continue

            fitted_params_band[jj, i, :] = best_params
            residual_map_band[jj, i] = float(best_rmse)
            at_lo = _params_at_bounds(best_params, lower_bound, upper_bound, which="lower", rtol=1e-6)
            at_hi = _params_at_bounds(best_params, lower_bound, upper_bound, which="upper", rtol=1e-6)

            payload = {
                "ok": True,
                "rmse": float(best_rmse),
                "n_starts": n_starts_total,
                "n_fail": n_fail_total,
                "p0_strategy": ("adaptive" if adaptive_multistart else fallback_p0_strategy),
                "adaptive_retry_used": bool(retry_used),
                "n_params_at_lower_bounds": int(np.count_nonzero(at_lo)),
                "n_params_at_upper_bounds": int(np.count_nonzero(at_hi)),
            }

            if diagnostics_mode == "full":
                payload["best_p0"] = np.asarray(best_p0, dtype=float)
                payload["params_at_lower_bounds_mask"] = at_lo
                payload["params_at_upper_bounds_mask"] = at_hi

            _store_diag(jj, i, payload)

            for k in range(n_peaks):
                block = np.asarray(best_params[stride*k:stride*(k+1)], dtype=float)
                centre = float(block[0])
                peak_positions_band[jj, i, k] = centre

                if peak_profile == "lorentzian":
                    hwhm = float(block[1])
                    amp_area = float(block[2])
                    if (not np.isfinite(hwhm)) or hwhm <= 0:
                        height_fitspace = np.nan
                    else:
                        height_fitspace = amp_area / (np.pi * hwhm)
                else:
                    y_one = single_peak(x, block, profile="pvoigt")
                    height_fitspace = float(np.nanmax(y_one))

                if normalize:
                    peak_intensities_band[jj, i, k] = height_fitspace
                else:
                    if fit_normalize:
                        peak_intensities_band[jj, i, k] = height_fitspace * s
                    else:
                        peak_intensities_band[jj, i, k] = height_fitspace

            if warm_start and best_rmse <= warm_start_rmse_gate:
                p0_current = np.asarray(best_params, dtype=float)
            else:
                if reset_on_fail:
                    p0_current = p0_base.copy()

        if row_reset:
            p0_current = p0_base.copy()

    return {
        "fitted_params": fitted_params_band,
        "residual_map": residual_map_band,
        "norm_scale_map": norm_scale_map_band,
        "peak_positions": peak_positions_band,
        "peak_intensities": peak_intensities_band,
        "diagnostics": diagnostics_band if diagnostics_mode != "none" else None,
    }
