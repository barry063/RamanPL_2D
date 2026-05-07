"""
test_peak_proposal.py
----------------------
Unit tests for ramanpl.single_fit.initialisation.

All test spectra are synthetic and generated inline.
No file I/O. No mapping class imports.
"""
import numpy as np
import pytest

from ramanpl.single_fit.initialisation import propose_peaks, p0_from_proposals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lorentzian(x, centre, hwhm, amp):
    """Lorentzian peak: amp / (pi * hwhm) * hwhm^2 / ((x-centre)^2 + hwhm^2)."""
    return (amp / (np.pi * hwhm)) * (hwhm ** 2) / ((x - centre) ** 2 + hwhm ** 2)


def _make_axis(n=200, lo=350.0, hi=450.0):
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# propose_peaks tests
# ---------------------------------------------------------------------------

def test_propose_peaks_detects_single_peak():
    x = _make_axis()
    spec = _lorentzian(x, centre=400.0, hwhm=5.0, amp=1000.0)
    props = propose_peaks(spec, x, n_peaks=1)
    assert len(props) == 1
    # centre should be within 2 array steps of true peak
    step = float(x[1] - x[0])
    assert abs(props[0]["centre"] - 400.0) < 2 * step


def test_propose_peaks_detects_two_peaks():
    x = _make_axis(n=300)
    spec = (
        _lorentzian(x, centre=380.0, hwhm=4.0, amp=800.0)
        + _lorentzian(x, centre=420.0, hwhm=4.0, amp=1200.0)
    )
    props = propose_peaks(spec, x, n_peaks=2)
    assert len(props) == 2
    centres = sorted(p["centre"] for p in props)
    assert abs(centres[0] - 380.0) < 3.0
    assert abs(centres[1] - 420.0) < 3.0


def test_propose_peaks_respects_n_peaks_limit():
    x = np.linspace(300.0, 700.0, 400)
    # Five well-separated peaks
    centres = [320.0, 380.0, 440.0, 520.0, 620.0]
    amps = [500.0, 300.0, 700.0, 400.0, 600.0]
    spec = sum(_lorentzian(x, c, 3.0, a) for c, a in zip(centres, amps))
    props = propose_peaks(spec, x, n_peaks=2)
    assert len(props) == 2
    # Should be the two tallest
    returned_heights = [p["height"] for p in props]
    assert returned_heights[0] >= returned_heights[1]


def test_propose_peaks_empty_on_flat_spectrum():
    x = _make_axis()
    spec = np.ones_like(x)
    assert propose_peaks(spec, x, n_peaks=1) == []


def test_propose_peaks_width_estimate_reasonable():
    x = _make_axis(n=500, lo=350.0, hi=450.0)
    true_hwhm = 8.0
    true_fwhm = 2.0 * true_hwhm
    spec = _lorentzian(x, centre=400.0, hwhm=true_hwhm, amp=1000.0)
    props = propose_peaks(spec, x, n_peaks=1)
    assert len(props) == 1
    estimated_fwhm = props[0]["width"]
    # Tolerate 50% error — find_peaks uses half-prominence which differs from Lorentzian FWHM
    assert 0.5 * true_fwhm <= estimated_fwhm <= 1.5 * true_fwhm + 1.0


# ---------------------------------------------------------------------------
# p0_from_proposals tests
# ---------------------------------------------------------------------------

def test_p0_from_proposals_replaces_centre():
    # 1 Lorentzian peak: params = [centre, hwhm, amp]
    current_p0 = np.array([384.0, 10.0, 500.0])
    lb = np.array([350.0, 1.0, 10.0])
    ub = np.array([450.0, 50.0, 2000.0])
    proposals = [{"centre": 400.0, "width": 16.0, "height": 300.0}]
    result = p0_from_proposals(proposals, "lorentzian", current_p0, (lb, ub))
    assert result[0] == pytest.approx(400.0)


def test_p0_from_proposals_falls_back_when_oob():
    current_p0 = np.array([400.0, 10.0, 500.0])
    lb = np.array([350.0, 1.0, 10.0])
    ub = np.array([450.0, 50.0, 2000.0])
    # Proposal centre outside bounds
    proposals = [{"centre": 500.0, "width": 16.0, "height": 300.0}]
    result = p0_from_proposals(proposals, "lorentzian", current_p0, (lb, ub))
    assert result[0] == pytest.approx(400.0)  # unchanged


def test_p0_from_proposals_falls_back_on_empty():
    current_p0 = np.array([400.0, 10.0, 500.0])
    lb = np.array([350.0, 1.0, 10.0])
    ub = np.array([450.0, 50.0, 2000.0])
    result = p0_from_proposals([], "lorentzian", current_p0, (lb, ub))
    np.testing.assert_array_equal(result, current_p0)


def test_p0_from_proposals_preserves_other_params():
    # Lorentzian: centre, hwhm, amp — only centre and hwhm should change
    current_p0 = np.array([384.0, 10.0, 777.0])
    lb = np.array([350.0, 1.0, 10.0])
    ub = np.array([450.0, 50.0, 2000.0])
    proposals = [{"centre": 400.0, "width": 12.0, "height": 300.0}]
    result = p0_from_proposals(proposals, "lorentzian", current_p0, (lb, ub))
    # amp unchanged
    assert result[2] == pytest.approx(777.0)
    # centre and hwhm replaced
    assert result[0] == pytest.approx(400.0)
    assert result[1] == pytest.approx(6.0)  # FWHM=12 → HWHM=6

    # pvoigt: centre, fwhm, amp, eta — only centre and fwhm should change
    current_p0_pv = np.array([384.0, 10.0, 777.0, 0.5])
    lb_pv = np.array([350.0, 1.0, 10.0, 0.0])
    ub_pv = np.array([450.0, 50.0, 2000.0, 1.0])
    result_pv = p0_from_proposals(proposals, "pvoigt", current_p0_pv, (lb_pv, ub_pv))
    assert result_pv[2] == pytest.approx(777.0)  # amp unchanged
    assert result_pv[3] == pytest.approx(0.5)    # eta unchanged
    assert result_pv[0] == pytest.approx(400.0)  # centre replaced
    assert result_pv[1] == pytest.approx(12.0)   # fwhm replaced (pvoigt uses fwhm directly)
