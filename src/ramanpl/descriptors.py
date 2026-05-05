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


def build_feature_row(per_peak_dict, qa_dict, peak_labels, *, ratios=None, separations=None):
    """
    Build a flat feature dict for one pixel/spectrum.

    Parameters
    ----------
    per_peak_dict : dict
        Keyed by peak label; each value is a dict from ``_params_to_export_dict()``
        with keys: ``centre``, ``fwhm``, ``peak_height``, ``peak_height_norm``
        (plus ``amp``, ``scale``, optionally ``eta`` — silently ignored here).
    qa_dict : dict
        QA fields: ``rmse``, ``ok``, ``n_starts``, ``n_params_at_bounds``.
    peak_labels : sequence of str
        Ordered peak labels; controls column order.
    ratios : list of (str, str) or None
        Each ``(P1, P2)`` adds ``{P1}_{P2}_ratio = peak_height[P1] / peak_height[P2]``.
        Zero denominator → NaN.
    separations : list of (str, str) or None
        Each ``(P1, P2)`` adds ``{P1}_{P2}_separation = position[P1] - position[P2]``.

    Returns
    -------
    dict
        Flat row dict: per-peak columns, then derived columns, then QA columns.
        NaN inputs propagate to all derived columns via float arithmetic.
    """
    row = {}

    for name in peak_labels:
        d = per_peak_dict.get(name, {})
        row[f"{name}_position"] = float(d.get("centre", float("nan")))
        row[f"{name}_fwhm"] = float(d.get("fwhm", float("nan")))
        row[f"{name}_peak_height"] = float(d.get("peak_height", float("nan")))
        row[f"{name}_peak_height_norm"] = float(d.get("peak_height_norm", float("nan")))

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

    row.update(qa_dict)
    return row
