"""
Classical peak-proposal helpers for failed-fit recovery.

These functions are pure signal-processing utilities: they take a 1-D
spectrum and return candidate peak parameters. They do not call any
optimiser and have no imports from ramanpl internals.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences


def propose_peaks(spectrum, wavenumber, n_peaks, prominence_rel=0.05, width_min_pts=3):
    """
    Propose up to n_peaks candidate peaks from a baseline-subtracted 1-D spectrum.

    Parameters
    ----------
    spectrum : 1-D array
        Baseline-subtracted intensity values.
    wavenumber : 1-D array
        Spectral axis (cm⁻¹ or eV), same length as spectrum.
    n_peaks : int
        Maximum number of proposals to return (matches the fitter's expected
        peak count).
    prominence_rel : float
        Minimum peak prominence as a fraction of the spectrum's value range.
        Default 0.05 (5 %).
    width_min_pts : int
        Minimum peak width in array index units. Default 3.

    Returns
    -------
    list of dict
        Each dict has keys ``"centre"`` (float, wavenumber), ``"width"``
        (float, estimated FWHM in wavenumber units), ``"height"`` (float).
        Sorted by height descending, truncated to n_peaks.
        Returns an empty list if no peaks meet the criteria — caller must
        handle this.
    """
    spec = np.asarray(spectrum, dtype=float).ravel()
    wn = np.asarray(wavenumber, dtype=float).ravel()

    if spec.size < 3 or not np.any(np.isfinite(spec)):
        return []

    spec_range = float(np.nanmax(spec) - np.nanmin(spec))
    if spec_range <= 0:
        return []

    min_prominence = prominence_rel * spec_range

    peaks_idx, _ = find_peaks(spec, prominence=min_prominence, width=width_min_pts)

    if peaks_idx.size == 0:
        return []

    widths_pts, _, _, _ = peak_widths(spec, peaks_idx, rel_height=0.5)

    # Convert width from array-index units to wavenumber units
    dw = float(np.abs(np.mean(np.diff(wn)))) if wn.size > 1 else 1.0
    widths_wn = widths_pts * dw

    proposals = []
    for k, idx in enumerate(peaks_idx):
        proposals.append({
            "centre": float(wn[idx]),
            "width": float(widths_wn[k]),
            "height": float(spec[idx]),
        })

    proposals.sort(key=lambda p: p["height"], reverse=True)
    return proposals[:int(n_peaks)]


def p0_from_proposals(proposals, peak_profile, current_p0, bounds):
    """
    Build a revised p0 by substituting proposal centres and widths where they
    improve on current_p0 and lie within bounds.

    Only the centre (parameter index 0 per peak) and width (index 1) are
    replaced. Amplitude and eta are left unchanged.

    Parameters
    ----------
    proposals : list of dict
        Output of propose_peaks.
    peak_profile : str
        ``"lorentzian"`` or ``"pseudovoigt"`` / ``"pvoigt"``.
    current_p0 : 1-D array
        The current p0 that failed. Used as fallback for any parameter not
        improved by a proposal.
    bounds : tuple (lb, ub)
        Parameter bounds from the fitter.

    Returns
    -------
    1-D array
        Revised p0, same length as current_p0.
    """
    p0 = np.asarray(current_p0, dtype=float).copy()
    lb = np.asarray(bounds[0], dtype=float).ravel()
    ub = np.asarray(bounds[1], dtype=float).ravel()

    if not proposals:
        return p0

    profile = str(peak_profile).lower().strip()
    stride = 3 if profile == "lorentzian" else 4
    n_peaks = len(p0) // stride

    # Greedy assignment: each proposal is used at most once, matched to the
    # peak whose current centre it is closest to.
    remaining = list(proposals)
    for k in range(n_peaks):
        if not remaining:
            break
        base = stride * k
        current_centre = float(p0[base])

        best_prop = min(remaining, key=lambda p: abs(p["centre"] - current_centre))
        remaining.remove(best_prop)

        # Replace centre if within bounds
        c_lo, c_hi = float(lb[base]), float(ub[base])
        if c_lo <= best_prop["centre"] <= c_hi:
            p0[base] = best_prop["centre"]

        # Convert FWHM proposal to model width parameter and replace if in bounds
        fwhm = best_prop["width"]
        model_width = fwhm / 2.0 if profile == "lorentzian" else fwhm
        w_lo, w_hi = float(lb[base + 1]), float(ub[base + 1])
        if w_lo <= model_width <= w_hi:
            p0[base + 1] = model_width

    return p0
