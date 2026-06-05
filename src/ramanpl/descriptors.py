"""
Peak-pair-agnostic descriptor builder for ramanpl feature tables.

Public API
----------
build_feature_row   -- construct a flat feature dict for one pixel/spectrum
validate_peak_pairs -- validate peak-pair references against available labels
"""

import numpy as np


def validate_peak_pairs(pairs, peak_labels):
    """
    Raise ValueError if any label in *pairs* is not in *peak_labels*.

    Parameters
    ----------
    pairs : iterable of (str, str)
        Pairs to validate (ratios or separations list).
    peak_labels : sequence of str
        Known peak labels.
    """
    label_set = set(peak_labels)
    for p1, p2 in (pairs or []):
        for label in (p1, p2):
            if label not in label_set:
                raise ValueError(
                    f"Unknown peak label '{label}'. "
                    f"Available labels: {list(peak_labels)}"
                )


def build_feature_row(
    per_peak_dict, qa_dict, peak_labels,
    *, ratios=None, separations=None, area_ratios=None,
):
    """
    Build a flat feature dict for one pixel/spectrum.

    Parameters
    ----------
    per_peak_dict : dict
        Keyed by peak label; each value is a dict from ``_params_to_export_dict()``
        with keys: ``centre``, ``fwhm``, ``peak_height``, ``peak_height_norm``,
        ``amp``, ``amp_scaled`` (plus ``scale``, optionally ``eta`` — silently ignored).
    qa_dict : dict
        QA fields: ``rmse``, ``ok``, ``n_starts``, ``n_params_at_bounds``.
    peak_labels : sequence of str
        Ordered peak labels; controls column order.
    ratios : list of (str, str) or None
        Each ``(P1, P2)`` adds ``{P1}_{P2}_ratio = peak_height[P1] / peak_height[P2]``.
        Zero denominator → NaN.
    separations : list of (str, str) or None
        Each ``(P1, P2)`` adds ``{P1}_{P2}_separation = position[P1] - position[P2]``.
    area_ratios : list of (str, str) or None
        Each ``(P1, P2)`` adds ``{P1}_{P2}_area_ratio = component_area_norm[P1] /
        component_area_norm[P2]``. Zero denominator → NaN.

    Returns
    -------
    dict
        Flat row dict: per-peak columns, then derived columns, then QA columns.
        NaN inputs propagate to all derived columns via float arithmetic.

    Notes
    -----
    ``component_area_norm`` equals ``amp`` (area-normalised fit parameter); fraction and
    ratio are computed from the normalised form because ``intensity_scale`` is constant
    across peaks within a row, making the scale-invariant result identical to using the
    scaled form.  The analytic area assumes integration over (−∞, ∞); window truncation
    is negligible unless a peak is very broad relative to the spectral window.
    """
    # Pre-compute area totals for fraction (needed to emit fractions inline with each peak)
    _area_norms = {
        name: float(per_peak_dict.get(name, {}).get("amp", float("nan")))
        for name in peak_labels
    }
    _finite_norms = [a for a in _area_norms.values() if np.isfinite(a)]
    _area_total = float(np.sum(_finite_norms)) if _finite_norms else float("nan")

    row = {}

    for name in peak_labels:
        d = per_peak_dict.get(name, {})
        row[f"{name}_position"] = float(d.get("centre", float("nan")))
        row[f"{name}_fwhm"] = float(d.get("fwhm", float("nan")))
        row[f"{name}_peak_height"] = float(d.get("peak_height", float("nan")))
        row[f"{name}_peak_height_norm"] = float(d.get("peak_height_norm", float("nan")))
        row[f"{name}_component_area"] = float(d.get("amp_scaled", float("nan")))
        row[f"{name}_component_area_norm"] = float(d.get("amp", float("nan")))
        a = _area_norms[name]
        if not np.isfinite(a) or _area_total == 0.0:
            row[f"{name}_component_area_fraction"] = float("nan")
        else:
            row[f"{name}_component_area_fraction"] = float(np.float64(a) / np.float64(_area_total))

    for p1, p2 in (separations or []):
        pos1 = row[f"{p1}_position"]
        pos2 = row[f"{p2}_position"]
        row[f"{p1}_{p2}_separation"] = float(np.float64(pos1) - np.float64(pos2))

    for p1, p2 in (ratios or []):
        h1 = row[f"{p1}_peak_height"]
        h2 = row[f"{p2}_peak_height"]
        h2_f = float(np.float64(h2))
        if h2_f == 0.0:
            row[f"{p1}_{p2}_ratio"] = float("nan")
        else:
            row[f"{p1}_{p2}_ratio"] = float(np.float64(h1) / np.float64(h2))

    for p1, p2 in (area_ratios or []):
        a1 = row[f"{p1}_component_area_norm"]
        a2 = row[f"{p2}_component_area_norm"]
        a2_f = float(np.float64(a2))
        if a2_f == 0.0:
            row[f"{p1}_{p2}_area_ratio"] = float("nan")
        else:
            row[f"{p1}_{p2}_area_ratio"] = float(np.float64(a1) / np.float64(a2_f))

    row.update(qa_dict)
    return row
