"""Peak line-shape models for RamanPL_2D.

Centralises analytical peak functions used across:
- PLfit / RamanFit
- PLMapping / RamanMapping

v0.3.x goal:
- keep existing Lorentzian behaviour (3 params/peak)
- add extension point for v0.3.3 pseudo-Voigt (4 params/peak)
"""

from __future__ import annotations
from typing import Literal
import numpy as np

PeakProfile = Literal["lorentzian", "gaussian", "pvoigt"]


def lorentzian_area(x: np.ndarray, centre: float, scale: float, amp: float) -> np.ndarray:
    """Historical Lorentzian (area-like amp), scale = HWHM (FWHM = 2*scale)."""
    x = np.asarray(x, dtype=float)
    centre = np.asarray(centre, dtype=float)
    scale = np.asarray(scale, dtype=float)
    amp = np.asarray(amp, dtype=float)
    return (scale / ((x - centre) ** 2 + scale**2)) * (amp / np.pi)

def gaussian_area(x: np.ndarray, centre: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian with an area-like amplitude (amp = area). width parameter is sigma."""
    x = np.asarray(x, dtype=float)
    centre = np.asarray(centre, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    amp = np.asarray(amp, dtype=float)

    safe_sigma = np.where(sigma > 0, sigma, np.nan)
    y = (amp / (safe_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - centre) / safe_sigma) ** 2)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def pseudo_voigt_area(x: np.ndarray, centre: float, fwhm: float, amp: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt = eta*Lorentzian + (1-eta)*Gaussian, width = FWHM for both components."""
    x = np.asarray(x, dtype=float)
    centre = np.asarray(centre, dtype=float)
    fwhm = np.asarray(fwhm, dtype=float)
    amp = np.asarray(amp, dtype=float)
    eta = np.asarray(eta, dtype=float)

    eta = np.clip(eta, 0.0, 1.0)

    return (
        eta * lorentzian_area_fwhm(x, centre, fwhm, amp)
        + (1.0 - eta) * gaussian_area_fwhm(x, centre, fwhm, amp)
    )

def lorentzian_area_fwhm(x: np.ndarray, centre: float, fwhm: float, amp: float) -> np.ndarray:
    # Lorentzian: HWHM = fwhm/2
    return lorentzian_area(x, centre, 0.5 * fwhm, amp)

def gaussian_area_fwhm(x: np.ndarray, centre: float, fwhm: float, amp: float) -> np.ndarray:
    # Gaussian: sigma = fwhm / (2*sqrt(2*ln2))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return gaussian_area(x, centre, sigma, amp)

def sum_peaks(
    x: np.ndarray,
    params,
    *,
    profile: PeakProfile = "lorentzian",
    stride: int,
) -> np.ndarray:
    """
    Sum peaks defined by a flattened parameter vector.

    Implementation note
    -------------------
    This version vectorises the per-peak evaluation using NumPy broadcasting,
    which reduces Python-loop overhead during repeated optimiser calls.
    """
    x = np.asarray(x, dtype=float).ravel()
    p = np.asarray(params, dtype=float).ravel()

    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    if p.size % stride != 0:
        raise ValueError(
            f"Parameter vector length ({p.size}) is not divisible by stride ({stride})."
        )

    if p.size == 0:
        return np.zeros_like(x, dtype=float)

    blocks = p.reshape(-1, stride)   # shape [n_peaks, stride]
    x2 = x[:, None]                  # shape [n_points, 1]

    if profile == "lorentzian":
        if stride != 3:
            raise ValueError("lorentzian requires stride=3 (centre, width, amp)")
        centre = blocks[:, 0]
        width = blocks[:, 1]
        amp = blocks[:, 2]

        y_all = lorentzian_area(x2, centre, width, amp)   # shape [n_points, n_peaks]
        return np.sum(y_all, axis=1)

    if profile == "gaussian":
        if stride != 3:
            raise ValueError("gaussian requires stride=3 (centre, width, amp)")
        centre = blocks[:, 0]
        width = blocks[:, 1]
        amp = blocks[:, 2]

        y_all = gaussian_area(x2, centre, width, amp)     # shape [n_points, n_peaks]
        return np.sum(y_all, axis=1)

    if profile == "pvoigt":
        if stride != 4:
            raise ValueError("pseudo-Voigt requires stride=4 (centre, width, amp, eta)")
        centre = blocks[:, 0]
        width = blocks[:, 1]
        amp = blocks[:, 2]
        eta = blocks[:, 3]

        y_all = pseudo_voigt_area(x2, centre, width, amp, eta)   # shape [n_points, n_peaks]
        return np.sum(y_all, axis=1)

    raise ValueError(f"Unknown profile: {profile}")

def single_peak(x, params, *, profile: PeakProfile) -> np.ndarray:
    p = list(params)
    if profile == "lorentzian":
        centre, width, amp = p
        return lorentzian_area(x, centre, width, amp)
    if profile == "gaussian":
        centre, width, amp = p
        return gaussian_area(x, centre, width, amp)
    if profile == "pvoigt":
        centre, width, amp, eta = p
        return pseudo_voigt_area(x, centre, width, amp, eta)
    raise ValueError(f"Unknown profile: {profile}")
