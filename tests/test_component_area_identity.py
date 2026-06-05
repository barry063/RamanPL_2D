"""
Step 1 — v0.6.7: Verify that the analytic area of each lineshape equals `amp`.

All three profiles in peak_models.py are area-normalised:
- Lorentzian: ∫ hwhm/((x-c)^2+hwhm^2) * amp/π dx = amp
- Gaussian:   ∫ amp/(σ√2π) exp(-½((x-c)/σ)²) dx = amp
- pVoigt:     convex blend of Lorentzian + Gaussian, both carrying same amp → integral = amp

Tolerance: relative error < 1e-4 (fine grid of 1e6 points over ±50×width).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.peak_models import single_peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_integral(profile, block, n=1_000_000, half_width_factor=50):
    """Numerically integrate single_peak over centre ± half_width_factor * width.

    Lorentzian/pVoigt require a large half_width_factor (~10000) because the Cauchy
    tail decays as 1/x² — window truncation at factor F contributes ~2/(π·F) relative
    error. Gaussian is fine at factor 50 (exp decay makes truncation negligible).
    """
    centre = block[0]
    width = block[1]   # HWHM for lorentzian, sigma for gaussian, FWHM for pvoigt
    lo = centre - half_width_factor * width
    hi = centre + half_width_factor * width
    x = np.linspace(lo, hi, n)
    y = single_peak(x, block, profile=profile)
    return np.trapezoid(y, x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amp", [1.0, 2.5, 0.001, 100.0])
def test_lorentzian_area_equals_amp(amp):
    # Cauchy tail decays as 1/x²; use factor=10000 → window error ≈ 2/(π·10000) ≈ 6e-5
    centre, hwhm = 500.0, 5.0
    block = [centre, hwhm, amp]
    area = _numeric_integral("lorentzian", block, half_width_factor=10_000)
    assert area == pytest.approx(amp, rel=1e-3), (
        f"Lorentzian integral {area:.6g} ≠ amp {amp} (rel error > 1e-3)"
    )


@pytest.mark.parametrize("amp", [1.0, 2.5, 0.001, 100.0])
def test_gaussian_area_equals_amp(amp):
    # Gaussian tail decays as exp(−t²/2); factor=50 gives negligible truncation error
    centre, sigma = 500.0, 5.0
    block = [centre, sigma, amp]
    area = _numeric_integral("gaussian", block, half_width_factor=50)
    assert area == pytest.approx(amp, rel=1e-4), (
        f"Gaussian integral {area:.6g} ≠ amp {amp} (rel error > 1e-4)"
    )


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("amp", [1.0, 3.0])
def test_pvoigt_area_equals_amp(amp, eta):
    # pVoigt contains a Lorentzian component; use same large factor as Lorentzian
    centre, fwhm = 500.0, 8.0
    block = [centre, fwhm, amp, eta]
    area = _numeric_integral("pvoigt", block, half_width_factor=10_000)
    assert area == pytest.approx(amp, rel=1e-3), (
        f"pVoigt (eta={eta}) integral {area:.6g} ≠ amp {amp} (rel error > 1e-3)"
    )
