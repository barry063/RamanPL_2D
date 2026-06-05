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


# ---------------------------------------------------------------------------
# Order-convention tests (v0.5.5 freeze)
# ---------------------------------------------------------------------------

def test_ratio_order_swap_yields_reciprocal_when_both_finite_nonzero():
    """Swapping (A,B) → (B,A) yields the reciprocal ratio for finite nonzero heights."""
    per_peak = {
        "A": _lorentzian_peak_dict(520.0, 5.0, 10.0),
        "B": _lorentzian_peak_dict(500.0, 3.0, 6.0),
    }
    qa = _qa()
    row_ab = build_feature_row(per_peak, qa, ["A", "B"], ratios=[("A", "B")])
    row_ba = build_feature_row(per_peak, qa, ["A", "B"], ratios=[("B", "A")])
    assert row_ab["A_B_ratio"] * row_ba["B_A_ratio"] == pytest.approx(1.0, rel=1e-12)


def test_separation_order_swap_yields_negation():
    """Swapping (A,B) → (B,A) yields the negated separation."""
    per_peak = {
        "A": _lorentzian_peak_dict(520.0, 5.0, 10.0),
        "B": _lorentzian_peak_dict(500.0, 3.0, 6.0),
    }
    qa = _qa()
    row_ab = build_feature_row(per_peak, qa, ["A", "B"], separations=[("A", "B")])
    row_ba = build_feature_row(per_peak, qa, ["A", "B"], separations=[("B", "A")])
    assert row_ab["A_B_separation"] == pytest.approx(-row_ba["B_A_separation"])


def test_ratio_zero_denominator_asymmetry():
    """(A,B) with h_B=0 → NaN; (B,A) with h_A>0 → 0.0 exactly."""
    per_peak = {
        "A": _lorentzian_peak_dict(520.0, 5.0, 10.0),
        "B": {"centre": 500.0, "fwhm": 6.0, "peak_height": 0.0, "peak_height_norm": 0.0},
    }
    qa = _qa()
    row_ab = build_feature_row(per_peak, qa, ["A", "B"], ratios=[("A", "B")])
    row_ba = build_feature_row(per_peak, qa, ["A", "B"], ratios=[("B", "A")])
    assert math.isnan(row_ab["A_B_ratio"])
    assert row_ba["B_A_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# v0.6.7: Component-area columns (Step 4)
# ---------------------------------------------------------------------------

def _lorentzian_peak_dict_with_area(centre, hwhm, amp_area, intensity_scale=1.0):
    """Like _lorentzian_peak_dict but also supplies amp_scaled."""
    peak_height_norm = (amp_area / (math.pi * hwhm)) if hwhm != 0 else float("nan")
    return {
        "centre": centre,
        "fwhm": 2.0 * hwhm,
        "peak_height": peak_height_norm * intensity_scale,
        "peak_height_norm": peak_height_norm,
        "amp": amp_area,
        "amp_scaled": amp_area * intensity_scale,
        "scale": hwhm,
    }


def test_component_area_columns_values():
    """component_area_norm == amp; component_area == amp * scale; fraction sums to 1."""
    scale = 5.0
    per_peak = {
        "A": _lorentzian_peak_dict_with_area(520.0, 5.0, 10.0, intensity_scale=scale),
        "B": _lorentzian_peak_dict_with_area(500.0, 3.0, 6.0, intensity_scale=scale),
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["A", "B"])

    assert row["A_component_area_norm"] == pytest.approx(10.0)
    assert row["B_component_area_norm"] == pytest.approx(6.0)
    assert row["A_component_area"] == pytest.approx(10.0 * scale)
    assert row["B_component_area"] == pytest.approx(6.0 * scale)

    assert row["A_component_area_fraction"] == pytest.approx(10.0 / 16.0)
    assert row["B_component_area_fraction"] == pytest.approx(6.0 / 16.0)
    assert row["A_component_area_fraction"] + row["B_component_area_fraction"] == pytest.approx(1.0)


def test_component_area_fraction_sum_to_one_three_peaks():
    """Fraction sum = 1.0 for any number of finite peaks."""
    scale = 2.0
    per_peak = {
        "A": _lorentzian_peak_dict_with_area(520.0, 5.0, 10.0, intensity_scale=scale),
        "B": _lorentzian_peak_dict_with_area(500.0, 3.0, 6.0, intensity_scale=scale),
        "C": _lorentzian_peak_dict_with_area(480.0, 4.0, 4.0, intensity_scale=scale),
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["A", "B", "C"])
    total = row["A_component_area_fraction"] + row["B_component_area_fraction"] + row["C_component_area_fraction"]
    assert total == pytest.approx(1.0)


def test_area_ratio_values_and_zero_denominator():
    """area_ratio = amp[P1] / amp[P2]; zero denominator → NaN."""
    per_peak = {
        "A": _lorentzian_peak_dict_with_area(520.0, 5.0, 10.0),
        "B": _lorentzian_peak_dict_with_area(500.0, 3.0, 4.0),
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["A", "B"], area_ratios=[("A", "B"), ("B", "A")])

    assert row["A_B_area_ratio"] == pytest.approx(10.0 / 4.0)
    assert row["B_A_area_ratio"] == pytest.approx(4.0 / 10.0)
    assert row["A_B_area_ratio"] * row["B_A_area_ratio"] == pytest.approx(1.0, rel=1e-12)

    # zero denominator
    per_peak_zero = {
        "A": _lorentzian_peak_dict_with_area(520.0, 5.0, 10.0),
        "B": {**_lorentzian_peak_dict_with_area(500.0, 3.0, 0.0), "amp": 0.0, "amp_scaled": 0.0},
    }
    row_zero = build_feature_row(per_peak_zero, qa, ["A", "B"], area_ratios=[("A", "B")])
    assert math.isnan(row_zero["A_B_area_ratio"])


def test_component_area_nan_when_amp_missing():
    """If amp/amp_scaled absent from per_peak dict, new columns are NaN (NaN-safe .get)."""
    per_peak = {
        "A": {"centre": 520.0, "fwhm": 10.0, "peak_height": 2.0, "peak_height_norm": 0.4},
    }
    qa = _qa()
    row = build_feature_row(per_peak, qa, ["A"])

    assert math.isnan(row["A_component_area"])
    assert math.isnan(row["A_component_area_norm"])
    assert math.isnan(row["A_component_area_fraction"])


def test_component_area_failed_pixel_all_nan():
    """Full NaN per_peak dict → all area columns NaN; fraction NaN (not 0/0 error)."""
    per_peak = {
        "A": {"centre": float("nan"), "fwhm": float("nan"),
              "peak_height": float("nan"), "peak_height_norm": float("nan"),
              "amp": float("nan"), "amp_scaled": float("nan")},
        "B": {"centre": float("nan"), "fwhm": float("nan"),
              "peak_height": float("nan"), "peak_height_norm": float("nan"),
              "amp": float("nan"), "amp_scaled": float("nan")},
    }
    qa = _qa(ok=False, rmse=float("nan"))
    row = build_feature_row(per_peak, qa, ["A", "B"], area_ratios=[("A", "B")])

    assert math.isnan(row["A_component_area"])
    assert math.isnan(row["A_component_area_norm"])
    assert math.isnan(row["A_component_area_fraction"])
    assert math.isnan(row["B_component_area_fraction"])
    assert math.isnan(row["A_B_area_ratio"])


def test_qa_block_remains_last():
    """QA columns are always the last four columns in the row."""
    per_peak = {
        "A": _lorentzian_peak_dict_with_area(520.0, 5.0, 10.0),
        "B": _lorentzian_peak_dict_with_area(500.0, 3.0, 6.0),
    }
    qa = _qa()
    row = build_feature_row(
        per_peak, qa, ["A", "B"],
        ratios=[("A", "B")], separations=[("A", "B")], area_ratios=[("A", "B")],
    )
    qa_keys = ["ok", "rmse", "n_starts", "n_params_at_bounds"]
    keys = list(row.keys())
    assert keys[-4:] == qa_keys, f"QA not last: tail={keys[-4:]}"
