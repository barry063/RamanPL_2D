"""
test_peak_proposal_integration.py
-----------------------------------
Integration tests for the proposal fallback wired into
_run_mapping_curve_fit_trials in mapping/_fit_utils.py.

All spectra are synthetic and generated inline.
No file I/O. Tests call _run_mapping_curve_fit_trials directly.

Hard-pixel recipe
-----------------
Two Lorentzians at 380 and 420 cm⁻¹ (40 cm⁻¹ apart).
p0_current has both peaks at 390 — equidistant between the two true peaks.
maxfev=1 forces the normal trial to fail immediately (scipy raises
RuntimeError: maxfev exceeded before even one full iteration completes).
The proposal fallback uses max(maxfev, 6400) evals, giving it a fair chance
from the proposal-corrected starting point.

Easy-pixel recipe
-----------------
Single Lorentzian at 400 cm⁻¹, p0_current near the true parameters,
maxfev=6400.  The normal trial always succeeds; the proposal block is
unreachable.
"""
import numpy as np
import pytest

from ramanpl.mapping._fit_utils import _run_mapping_curve_fit_trials
from ramanpl.peak_models import sum_peaks


# ---------------------------------------------------------------------------
# Shared model and spectrum builders
# ---------------------------------------------------------------------------

def _model_lorentzian(x, *params):
    return sum_peaks(np.asarray(x, dtype=float), params, profile="lorentzian", stride=3)


def _model_pvoigt(x, *params):
    return sum_peaks(np.asarray(x, dtype=float), params, profile="pvoigt", stride=4)


def _lorentzian_peak(x, centre, hwhm, amp):
    return (amp / (np.pi * hwhm)) * hwhm ** 2 / ((x - centre) ** 2 + hwhm ** 2)


# ---------------------------------------------------------------------------
# Easy pixel (single Lorentzian, good starting p0)
# ---------------------------------------------------------------------------

_x_easy = np.linspace(360.0, 440.0, 300)
_spec_easy = _lorentzian_peak(_x_easy, 400.0, 8.0, 300.0)
_lb_easy = np.array([365.0, 1.0, 10.0])
_ub_easy = np.array([435.0, 25.0, 1000.0])
_p0_easy = np.array([400.5, 8.2, 305.0])   # near true [400, 8, 300]

# Easy pvoigt pixel
_x_pv = np.linspace(360.0, 440.0, 300)
_spec_pv = _lorentzian_peak(_x_pv, 400.0, 8.0, 300.0)   # same shape, different model
_lb_pv = np.array([365.0, 1.0, 10.0, 0.0])
_ub_pv = np.array([435.0, 35.0, 1000.0, 1.0])
_p0_pv = np.array([400.2, 16.5, 290.0, 0.5])   # near true

# ---------------------------------------------------------------------------
# Hard pixel (two Lorentzians, bad starting p0, maxfev=1)
# ---------------------------------------------------------------------------

_x_hard = np.linspace(355.0, 445.0, 400)
_spec_hard = (
    _lorentzian_peak(_x_hard, 380.0, 5.0, 300.0)
    + _lorentzian_peak(_x_hard, 420.0, 5.0, 300.0)
)
_lb_hard = np.array([360.0, 0.5, 10.0, 360.0, 0.5, 10.0])
_ub_hard = np.array([440.0, 20.0, 1000.0, 440.0, 20.0, 1000.0])
# Both peaks at 390 — equidistant between 380 and 420
_p0_hard = np.array([390.0, 10.0, 500.0, 390.0, 10.0, 500.0])

# ---------------------------------------------------------------------------
# Weak pixel (single small peak, bad starting p0, maxfev=1)
# ---------------------------------------------------------------------------

_x_weak = np.linspace(360.0, 440.0, 200)
_spec_weak = _lorentzian_peak(_x_weak, 400.0, 5.0, 50.0)   # small amplitude
_lb_weak = np.array([365.0, 0.5, 5.0])
_ub_weak = np.array([435.0, 20.0, 200.0])
_p0_weak = np.array([430.0, 15.0, 150.0])   # far from true peak


def _run_hard(use_peak_proposals):
    return _run_mapping_curve_fit_trials(
        model_fn=_model_lorentzian,
        x=_x_hard,
        y=_spec_hard,
        lower_bound=_lb_hard,
        upper_bound=_ub_hard,
        p0_current=_p0_hard,
        maxfev=1,   # guaranteed normal failure
        n_starts=1,
        p0_strategy="midpoint",
        peak_profile="lorentzian",
        stride=3,
        use_peak_proposals=use_peak_proposals,
    )


def _run_weak(use_peak_proposals):
    return _run_mapping_curve_fit_trials(
        model_fn=_model_lorentzian,
        x=_x_weak,
        y=_spec_weak,
        lower_bound=_lb_weak,
        upper_bound=_ub_weak,
        p0_current=_p0_weak,
        maxfev=1,
        n_starts=1,
        p0_strategy="midpoint",
        peak_profile="lorentzian",
        stride=3,
        use_peak_proposals=use_peak_proposals,
    )


def _run_easy(use_peak_proposals):
    return _run_mapping_curve_fit_trials(
        model_fn=_model_lorentzian,
        x=_x_easy,
        y=_spec_easy,
        lower_bound=_lb_easy,
        upper_bound=_ub_easy,
        p0_current=_p0_easy,
        maxfev=6400,
        n_starts=1,
        p0_strategy="midpoint",
        peak_profile="lorentzian",
        stride=3,
        use_peak_proposals=use_peak_proposals,
    )


def _run_easy_pv(use_peak_proposals):
    return _run_mapping_curve_fit_trials(
        model_fn=_model_pvoigt,
        x=_x_pv,
        y=_spec_pv,
        lower_bound=_lb_pv,
        upper_bound=_ub_pv,
        p0_current=_p0_pv,
        maxfev=6400,
        n_starts=1,
        p0_strategy="midpoint",
        peak_profile="pvoigt",
        stride=4,
        use_peak_proposals=use_peak_proposals,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_proposal_fallback_saves_overlapping_pixel():
    """Hard two-peak pixel fails normally (maxfev=1), succeeds with proposals."""
    result_off = _run_hard(use_peak_proposals=False)
    result_on = _run_hard(use_peak_proposals=True)

    assert not result_off["ok"], "Normal trial should fail with maxfev=1"
    assert result_on["ok"], "Proposal fallback should rescue the pixel"


def test_proposal_fallback_saves_weak_pixel():
    """Weak single-peak pixel fails normally (maxfev=1), succeeds with proposals."""
    result_off = _run_weak(use_peak_proposals=False)
    result_on = _run_weak(use_peak_proposals=True)

    assert not result_off["ok"], "Normal trial should fail with maxfev=1"
    assert result_on["ok"], "Proposal fallback should rescue the pixel"


def test_no_change_on_easy_pixel():
    """
    Easy pixel: proposal block is unreachable.
    Both runs execute identical code → bit-identical best_params.
    This is the scientific parity guard.
    """
    result_off = _run_easy(use_peak_proposals=False)
    result_on = _run_easy(use_peak_proposals=True)

    assert result_off["ok"] and result_on["ok"]
    np.testing.assert_array_equal(
        result_off["best_params"],
        result_on["best_params"],
        err_msg="Proposal flag must not affect easy-pixel result (proposals unreachable)",
    )


def test_no_change_on_easy_pixel_pvoigt():
    """Same parity check for pseudo-Voigt profile."""
    result_off = _run_easy_pv(use_peak_proposals=False)
    result_on = _run_easy_pv(use_peak_proposals=True)

    assert result_off["ok"] and result_on["ok"]
    np.testing.assert_array_equal(
        result_off["best_params"],
        result_on["best_params"],
        err_msg="Proposal flag must not affect easy pvoigt pixel",
    )


def test_proposal_fallback_disabled_flag():
    """use_peak_proposals=False must prevent the fallback from being entered."""
    result = _run_hard(use_peak_proposals=False)
    assert not result["ok"], "With proposals disabled, hard pixel must still fail"


def test_position_drift_within_tolerance():
    """
    On easy pixels, peak positions are identical between proposals-on and
    proposals-off (tolerance 0.1 cm⁻¹).  Verifies the fallback path does
    not interfere with the normal fitting path.
    """
    off = _run_easy(use_peak_proposals=False)
    on = _run_easy(use_peak_proposals=True)
    assert off["ok"] and on["ok"]
    centre_off = float(off["best_params"][0])
    centre_on = float(on["best_params"][0])
    assert abs(centre_on - centre_off) < 0.1, (
        f"Centre drift {abs(centre_on - centre_off):.4f} cm⁻¹ exceeds 0.1 cm⁻¹"
    )


def test_fwhm_drift_within_tolerance():
    """
    On easy pixels, FWHM (2×HWHM for Lorentzian) differs by < 1% between
    proposals-on and proposals-off.
    """
    off = _run_easy(use_peak_proposals=False)
    on = _run_easy(use_peak_proposals=True)
    assert off["ok"] and on["ok"]
    hwhm_off = float(off["best_params"][1])
    hwhm_on = float(on["best_params"][1])
    rel_diff = abs(hwhm_on - hwhm_off) / max(abs(hwhm_off), 1e-12)
    assert rel_diff < 0.01, f"FWHM drift {rel_diff*100:.3f}% exceeds 1%"


def test_n_failed_pixels_decreases():
    """
    On 8 hard two-peak pixels, the number of failures decreases when
    proposals are enabled.  At least one pixel must be rescued.
    """
    n_failed_off = sum(
        not _run_hard(use_peak_proposals=False)["ok"]
        for _ in range(8)
    )
    n_failed_on = sum(
        not _run_hard(use_peak_proposals=True)["ok"]
        for _ in range(8)
    )
    assert n_failed_on <= n_failed_off, (
        f"Proposals on: {n_failed_on} failed; off: {n_failed_off} failed — "
        "proposals should not increase failures"
    )
    assert n_failed_on < n_failed_off, (
        "Proposals must rescue at least one hard pixel"
    )
