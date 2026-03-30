import numpy as np
from scipy import optimize


def _mapping_rng(random_state=None):
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)


def _mapping_generate_p0_trials(lb, ub, base_p0, n_starts, strategy="midpoint", random_state=None):
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    base_p0 = np.asarray(base_p0, dtype=float).ravel()

    if n_starts is None:
        n_starts = 1
    n_starts = int(n_starts)
    if n_starts < 1:
        n_starts = 1

    strategy = (strategy or "midpoint").lower()
    if strategy not in {"midpoint", "random", "jitter"}:
        raise ValueError("p0_strategy must be one of: 'midpoint', 'random', 'jitter'.")

    trials = [base_p0.copy()]
    if n_starts == 1:
        return trials

    rng = _mapping_rng(random_state)
    m = n_starts - 1

    if strategy == "random":
        for _ in range(m):
            trials.append(rng.uniform(lb, ub))
        return trials

    scale = 0.10 * (ub - lb)
    scale = np.where(scale > 0, scale, 1.0)
    for _ in range(m):
        p = base_p0 + rng.normal(loc=0.0, scale=scale)
        p = np.clip(p, lb, ub)
        trials.append(p)

    return trials


def _params_at_bounds(params, lb, ub, *, which="both", rtol=1e-6, atol=1e-12):
    """
    Return a boolean mask for parameters that are (numerically) at bounds.
    """
    p = np.asarray(params, dtype=float).ravel()
    lo = np.asarray(lb, dtype=float).ravel()
    hi = np.asarray(ub, dtype=float).ravel()

    if p.size != lo.size or p.size != hi.size:
        raise ValueError("params/lb/ub length mismatch.")

    which = (which or "both").lower().strip()
    if which not in {"lower", "upper", "both"}:
        raise ValueError("which must be one of: 'lower', 'upper', 'both'.")

    at_lo = np.isclose(p, lo, rtol=rtol, atol=atol)
    at_hi = np.isclose(p, hi, rtol=rtol, atol=atol)

    if which == "lower":
        return at_lo
    if which == "upper":
        return at_hi
    return at_lo | at_hi


def seed_p0_from_coord(mapping_obj, coord, seed_roi=None, *, maxfev=6400):
    """
    Fit ONE spectrum from an already-loaded mapping object and return:
        {"p0": <vector>, "peak_order": <list>}
    """
    if coord is None:
        raise ValueError("coord must be provided as (x, y).")

    x, y = int(coord[0]), int(coord[1])
    if not (0 <= x < mapping_obj.X and 0 <= y < mapping_obj.Y):
        raise ValueError(
            f"coord {(x, y)} out of bounds for map size (X={mapping_obj.X}, Y={mapping_obj.Y})."
        )

    roi = None
    if seed_roi is not None:
        if isinstance(seed_roi, (int, np.integer)):
            r = int(seed_roi)
            x0 = max(0, x - r)
            x1 = min(mapping_obj.X - 1, x + r)
            y0 = max(0, y - r)
            y1 = min(mapping_obj.Y - 1, y + r)
            roi = (x0, x1, y0, y1)
        else:
            if (not isinstance(seed_roi, (tuple, list))) or len(seed_roi) != 4:
                raise ValueError("seed_roi must be an int radius or a 4-tuple (x0, x1, y0, y1).")
            roi = tuple(int(v) for v in seed_roi)

    y_ref, xaxis = mapping_obj.get_reference_spectrum(x=x, y=y, roi=roi)

    y_ref = np.asarray(y_ref, dtype=float).ravel()
    xaxis = np.asarray(xaxis, dtype=float).ravel()
    if y_ref.size != xaxis.size:
        raise ValueError("Seed spectrum length mismatch with axis length.")
    if not np.all(np.isfinite(xaxis)):
        raise ValueError("Axis contains NaN/Inf; cannot seed.")
    if not np.any(np.isfinite(y_ref)):
        raise ValueError("Seed spectrum contains only NaN/Inf; cannot seed.")

    spec_norm, scale = mapping_obj._preprocess_single_spectrum(xaxis, y_ref)
    if spec_norm is None or scale is None:
        raise ValueError(f"Seed spectrum at {(x, y)} has no positive signal after preprocessing; cannot seed.")

    spec_norm = np.asarray(spec_norm, dtype=float).ravel()
    if not np.all(np.isfinite(spec_norm)):
        raise ValueError("Preprocessed seed spectrum contains NaN/Inf; cannot seed.")

    lower_bound, upper_bound = [], []
    for params_range in mapping_obj.custom_peaks.values():
        lower_bound.extend(params_range[0])
        upper_bound.extend(params_range[1])

    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)

    if lower_bound.size != upper_bound.size:
        raise ValueError("custom_peaks bounds size mismatch.")
    if lower_bound.size == 0:
        raise ValueError("custom_peaks is empty; cannot seed.")
    if np.any(lower_bound >= upper_bound):
        raise ValueError("Invalid bounds in custom_peaks: found lower_bound >= upper_bound.")

    p0_base = (lower_bound + upper_bound) / 2.0

    if hasattr(mapping_obj, "_model_dispatch"):
        model = mapping_obj._model_dispatch()
    elif hasattr(mapping_obj, "lorentzian"):
        model = mapping_obj.lorentzian
    elif hasattr(mapping_obj, "lorentzian_raman"):
        model = mapping_obj.lorentzian_raman
    else:
        raise RuntimeError("mapping_obj must implement _model_dispatch, lorentzian (PL) or lorentzian_raman (Raman).")

    try:
        params, _ = optimize.curve_fit(
            model,
            xaxis,
            spec_norm,
            p0=p0_base,
            bounds=(lower_bound, upper_bound),
            maxfev=maxfev,
        )
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(f"Seed fit failed at coord={(x, y)} (roi={roi}): {e}") from e

    return {"p0": np.asarray(params, dtype=float), "peak_order": list(mapping_obj.peak_params)}


def _width_param_to_fwhm(width_param: np.ndarray, profile: str) -> np.ndarray:
    """Convert model width parameter to FWHM in x-units."""
    profile = str(profile).lower().strip()
    w = np.asarray(width_param, dtype=float)
    if profile == "lorentzian":
        return 2.0 * w
    return w