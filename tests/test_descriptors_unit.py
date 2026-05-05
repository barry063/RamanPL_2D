"""
Unit tests for ramanpl.descriptors (Step 1 — v0.5.2).

All tests are pure Python/NumPy; no mapping or fitting infrastructure required.
"""

import math
import pytest
import numpy as np

from ramanpl.descriptors import build_feature_row, validate_peak_pairs


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _lorentzian_peak_dict(centre, hwhm, amp_area, intensity_scale=1.0):
    """Build a per_peak_dict entry in _params_to_export_dict() format (lorentzian)."""
    peak_height_norm = (amp_area / (math.pi * hwhm)) if hwhm != 0 else float("nan")
    return {
        "centre": centre,
        "fwhm": 2.0 * hwhm,
        "peak_height": peak_height_norm * intensity_scale,
        "peak_height_norm": peak_height_norm,
        "amp": amp_area,
        "scale": hwhm,
    }


def _qa(ok=True, rmse=0.05, n_starts=1, n_params_at_bounds=0):
    return {"ok": ok, "rmse": rmse, "n_starts": n_starts, "n_params_at_bounds": n_params_at_bounds}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_feature_row_lorentzian_two_peaks_one_ratio_one_separation():
    """Values match hand calculation for a two-peak Lorentzian case."""
    # Peak A: centre=520, hwhm=5, amp_area=10 → height_norm=10/(π*5)=0.6366..., height=6.366 (scale=10)
    # Peak B: centre=500, hwhm=3, amp_area=6  → height_norm=6/(π*3)=0.6366..., height=6.366 (scale=10)
    scale = 10.0
    per_peak = {
        "A": _lorentzian_peak_dict(520.0, 5.0, 10.0, intensity_scale=scale),
        "B": _lorentzian_peak_dict(500.0, 3.0, 6.0, intensity_scale=scale),
    }
    qa = _qa()

    row = build_feature_row(
        per_peak, qa, ["A", "B"],
        ratios=[("A", "B")],
        separations=[("A", "B")],
    )

    # --- per-peak columns ---
    assert row["A_position"] == pytest.approx(520.0)
    assert row["A_fwhm"] == pytest.approx(10.0)
    assert row["B_position"] == pytest.approx(500.0)
    assert row["B_fwhm"] == pytest.approx(6.0)

    # --- separation ---
    expected_sep = 520.0 - 500.0
    assert row["A_B_separation"] == pytest.approx(expected_sep)

    # --- ratio ---
    h_A = per_peak["A"]["peak_height"]
    h_B = per_peak["B"]["peak_height"]
    expected_ratio = h_A / h_B
    assert row["A_B_ratio"] == pytest.approx(expected_ratio)

    # --- QA passthrough ---
    assert row["ok"] is True
    assert row["rmse"] == pytest.approx(0.05)
    assert row["n_starts"] == 1
    assert row["n_params_at_bounds"] == 0


def test_build_feature_row_pvoigt_eta_silently_ignored():
    """pVoigt input dict carries an 'eta' field; eta must NOT appear in output."""
    per_peak = {
        "P": {
            "centre": 520.0,
            "fwhm": 8.0,
            "peak_height": 3.0,
            "peak_height_norm": 0.75,
            "amp": 4.0,
            "scale": 8.0,
            "eta": 0.5,   # pVoigt-specific field — silently ignored
        }
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["P"])

    assert "P_position" in row
    assert "P_fwhm" in row
    assert "P_peak_height" in row
    assert "P_peak_height_norm" in row
    # eta must NOT appear as a column
    assert not any("eta" in k for k in row)


def test_build_feature_row_failed_pixel_propagates_nan():
    """NaN inputs propagate to per-peak and derived columns; QA ok=False passes through."""
    per_peak = {
        "A": {"centre": float("nan"), "fwhm": float("nan"),
              "peak_height": float("nan"), "peak_height_norm": float("nan")},
        "B": {"centre": float("nan"), "fwhm": float("nan"),
              "peak_height": float("nan"), "peak_height_norm": float("nan")},
    }
    qa = _qa(ok=False, rmse=float("nan"))
    row = build_feature_row(
        per_peak, qa, ["A", "B"],
        ratios=[("A", "B")],
        separations=[("A", "B")],
    )

    assert math.isnan(row["A_position"])
    assert math.isnan(row["A_fwhm"])
    assert math.isnan(row["A_peak_height"])
    assert math.isnan(row["B_peak_height_norm"])
    assert math.isnan(row["A_B_separation"])
    assert math.isnan(row["A_B_ratio"])
    assert row["ok"] is False
    assert math.isnan(row["rmse"])


def test_build_feature_row_zero_denominator_ratio_returns_nan():
    """Ratio with zero denominator returns NaN (not inf)."""
    per_peak = {
        "A": _lorentzian_peak_dict(520.0, 5.0, 10.0),
        "B": {"centre": 500.0, "fwhm": 6.0, "peak_height": 0.0, "peak_height_norm": 0.0},
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["A", "B"], ratios=[("A", "B")])

    assert math.isnan(row["A_B_ratio"]), "Expected NaN for zero denominator, got inf or other"
    assert not math.isinf(row.get("A_B_ratio", float("nan")))


def test_validate_peak_pairs_raises_on_unknown_label():
    """ValueError is raised and the offending label is named in the message."""
    peak_labels = ["A1g", "E2g"]
    with pytest.raises(ValueError, match="Xbad"):
        validate_peak_pairs([("A1g", "Xbad")], peak_labels)
