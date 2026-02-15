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
    x = np.asarray(x)
    return (scale / ((x - centre) ** 2 + scale**2)) * (amp / np.pi)


def gaussian_area(x: np.ndarray, centre: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian with an area-like amplitude (amp = area). width parameter is sigma."""
    x = np.asarray(x)
    sigma = float(sigma)
    if sigma <= 0:
        return np.zeros_like(x, dtype=float)
    return (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - centre) / sigma) ** 2)

def pseudo_voigt_area(x: np.ndarray, centre: float, fwhm: float, amp: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt = eta*Lorentzian + (1-eta)*Gaussian, width = FWHM for both components."""
    eta = float(eta)
    eta = 0.0 if eta < 0.0 else 1.0 if eta > 1.0 else eta
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
    """Sum peaks defined by a flattened parameter vector."""
    x = np.asarray(x)
    p = np.asarray(params, dtype=float)
    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    if p.size % stride != 0:
        raise ValueError(f"Parameter vector length ({p.size}) is not divisible by stride ({stride}).")

    y = np.zeros_like(x, dtype=float)

    if profile == "lorentzian":
        for i in range(0, p.size, stride):
            centre, width, amp = p[i : i + 3]
            y += lorentzian_area(x, centre, width, amp)
        return y

    if profile == "gaussian":
        for i in range(0, p.size, stride):
            centre, width, amp = p[i : i + 3]
            y += gaussian_area(x, centre, width, amp)
        return y

    if profile == "pvoigt":
        if stride != 4:
            raise ValueError("pseudo-Voigt requires stride=4 (centre, width, amp, eta)")
        for i in range(0, p.size, stride):
            centre, width, amp, eta = p[i : i + 4]
            y += pseudo_voigt_area(x, centre, width, amp, eta)
        return y

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
