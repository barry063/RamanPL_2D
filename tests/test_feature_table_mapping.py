"""
test_feature_table_mapping.py
------------------------------
Tests for _MappingPreprocessMixin.feature_table() (Step 2 — v0.5.2).

Uses synthetic mapping fixtures without real fitting (same approach as
test_export_fit_map_qa_columns.py).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.mapping import RamanMapping
from ramanpl.mapping._pl_mapping import PLMapping


# ---------------------------------------------------------------------------
# Synthetic cube helpers (shared by Raman and PL fixtures)
# ---------------------------------------------------------------------------

_N_PTS = 80
_X_RAMAN = np.linspace(300.0, 700.0, _N_PTS)
_X_PL = np.linspace(1.85, 2.10, _N_PTS)
_Y_MAP, _X_MAP = 3, 4   # map size: Y=3 rows, X=4 columns → 12 pixels

_CUSTOM_PEAKS_RAMAN = {
    "E2g": ([380.0, 1.0, 0.001], [392.0, 20.0, 5.0]),
    "A1g": ([402.0, 1.0, 0.001], [412.0, 20.0, 5.0]),
}
_CUSTOM_PEAKS_PL = {
    "Trion":   ([1.88, 0.005, 0.001], [1.96, 0.10, 5.0]),
    "Exciton": ([1.96, 0.005, 0.001], [2.05, 0.10, 5.0]),
}

# Lorentzian params: [centre, hwhm, amp_area]
_GOOD_PARAMS_RAMAN = np.array([386.0, 3.0, 2.0, 407.0, 4.0, 3.0], dtype=float)
_GOOD_PARAMS_PL    = np.array([1.91, 0.02, 1.5, 2.00, 0.03, 2.5], dtype=float)
_FAIL_PARAMS       = None   # sentinel — will be set to NaN array of right size


def _make_raman_mapping():
    rng = np.random.default_rng(10)
    peak1 = np.exp(-0.5 * ((_X_RAMAN - 386.0) / 3.0) ** 2)
    peak2 = np.exp(-0.5 * ((_X_RAMAN - 407.0) / 4.0) ** 2)
    signal = peak1 + peak2
    cube = (signal[None, None, :] + rng.uniform(0.0, 0.02, (_Y_MAP, _X_MAP, _N_PTS))).astype(float)
    m = RamanMapping.from_arrays(
        cube, _X_RAMAN, _X_MAP, _Y_MAP,
        custom_peaks=_CUSTOM_PEAKS_RAMAN,
        data_range=(300.0, 700.0),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    return m


def _make_pl_mapping():
    rng = np.random.default_rng(20)
    peak1 = np.exp(-0.5 * ((_X_PL - 1.91) / 0.02) ** 2)
    peak2 = np.exp(-0.5 * ((_X_PL - 2.00) / 0.03) ** 2)
    signal = peak1 + peak2
    cube = (signal[None, None, :] + rng.uniform(0.0, 0.02, (_Y_MAP, _X_MAP, _N_PTS))).astype(float)
    m = PLMapping.from_arrays(
        cube, _X_PL, _X_MAP, _Y_MAP,
        custom_peaks=_CUSTOM_PEAKS_PL,
        data_range=(1.85, 2.10),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    return m


def _inject_fit(m, good_params, fail_pixels=None):
    """Inject synthetic fit results without running fit_spectra."""
    n_params = len(good_params)
    fail_params = np.full(n_params, float("nan"), dtype=float)

    m.fitted_params = np.full((m.Y, m.X, n_params), float("nan"), dtype=float)
    m.residual_map = np.full((m.Y, m.X), float("nan"), dtype=float)
    m.norm_scale_map = np.full((m.Y, m.X), 1.0, dtype=float)
    m.fit_diagnostics_map = np.empty((m.Y, m.X), dtype=object)
    m.fit_diagnostics_map[:, :] = None

    fail_set = set(fail_pixels or [])

    for j in range(m.Y):
        for i in range(m.X):
            if (j, i) in fail_set:
                m.fitted_params[j, i, :] = fail_params
                m.residual_map[j, i] = float("nan")
                m.fit_diagnostics_map[j, i] = {
                    "ok": False, "n_starts": 2,
                    "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
                }
            else:
                m.fitted_params[j, i, :] = good_params
                m.residual_map[j, i] = 0.04
                m.norm_scale_map[j, i] = 5.0
                m.fit_diagnostics_map[j, i] = {
                    "ok": True, "n_starts": 1,
                    "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
                }

    # allocate peak_intensities (used by test_7)
    n_peaks = len(list(m.peak_params))
    m.peak_intensities = np.full((m.Y, m.X, n_peaks), float("nan"), dtype=float)
    for j in range(m.Y):
        for i in range(m.X):
            if (j, i) not in fail_set:
                for k in range(n_peaks):
                    # peak height = peak_height_norm * norm_scale_map
                    hwhm = float(good_params[m.params_per_peak * k + 1])
                    amp  = float(good_params[m.params_per_peak * k + 2])
                    h_norm = amp / (np.pi * hwhm) if hwhm != 0 else float("nan")
                    m.peak_intensities[j, i, k] = h_norm * m.norm_scale_map[j, i]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_table_returns_dataframe_with_expected_shape():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    assert len(df) == _Y_MAP * _X_MAP


def test_feature_table_column_schema():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()

    assert "x" in df.columns
    assert "y" in df.columns

    for peak in ("E2g", "A1g"):
        for suffix in ("_position", "_fwhm", "_peak_height", "_peak_height_norm"):
            assert f"{peak}{suffix}" in df.columns, f"Missing column {peak}{suffix}"

    for qa_col in ("rmse", "ok", "n_starts", "n_params_at_bounds"):
        assert qa_col in df.columns, f"Missing QA column {qa_col}"


def test_feature_table_with_ratios_and_separations():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)

    df = m.feature_table(
        ratios=[("A1g", "E2g")],
        separations=[("A1g", "E2g")],
    )

    assert "A1g_E2g_separation" in df.columns
    assert "A1g_E2g_ratio" in df.columns

    # Verify values in a good pixel (0, 0)
    row = df[(df["x"] == 0) & (df["y"] == 0)].iloc[0]
    expected_sep = row["A1g_position"] - row["E2g_position"]
    assert row["A1g_E2g_separation"] == pytest.approx(expected_sep, rel=1e-9)

    expected_ratio = row["A1g_peak_height"] / row["E2g_peak_height"]
    assert row["A1g_E2g_ratio"] == pytest.approx(expected_ratio, rel=1e-9)


def test_feature_table_failed_pixel_emits_nan_row():
    m = _make_raman_mapping()
    fail_px = (1, 2)
    _inject_fit(m, _GOOD_PARAMS_RAMAN, fail_pixels=[fail_px])

    df = m.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])

    assert len(df) == _Y_MAP * _X_MAP

    # failed pixel: x=2, y=1 (i=2, j=1 → x=i, y=j in pixel mode)
    fail_row = df[(df["x"] == fail_px[1]) & (df["y"] == fail_px[0])].iloc[0]
    assert fail_row["ok"] is False or fail_row["ok"] == False
    assert math.isnan(fail_row["E2g_position"])
    assert math.isnan(fail_row["A1g_fwhm"])
    assert math.isnan(fail_row["A1g_E2g_separation"])
    assert math.isnan(fail_row["A1g_E2g_ratio"])


def test_feature_table_invalid_pair_raises():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)

    with pytest.raises(ValueError, match="Xbad"):
        m.feature_table(ratios=[("A1g", "Xbad")])

    with pytest.raises(ValueError, match="Ybad"):
        m.feature_table(separations=[("Ybad", "E2g")])


def test_feature_table_works_for_pl_mapping():
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)

    df = m.feature_table(separations=[("Exciton", "Trion")])

    assert len(df) == _Y_MAP * _X_MAP
    assert "Exciton_position" in df.columns
    assert "Trion_peak_height" in df.columns
    assert "Exciton_Trion_separation" in df.columns

    row0 = df.iloc[0]
    expected_sep = row0["Exciton_position"] - row0["Trion_position"]
    assert row0["Exciton_Trion_separation"] == pytest.approx(expected_sep, rel=1e-9)


def test_feature_table_peak_height_matches_peak_intensities_array():
    """feature_table() peak-height column equals peak_intensities[:,:,k] for same peak."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)

    df = m.feature_table()

    for k, peak in enumerate(m.peak_params):
        for j in range(m.Y):
            for i in range(m.X):
                expected = float(m.peak_intensities[j, i, k])
                row = df[(df["x"] == i) & (df["y"] == j)].iloc[0]
                actual = float(row[f"{peak}_peak_height"])
                if math.isnan(expected):
                    assert math.isnan(actual)
                else:
                    assert actual == pytest.approx(expected, rel=1e-9), (
                        f"peak_height mismatch for {peak} at ({j},{i}): "
                        f"feature_table={actual}, peak_intensities={expected}"
                    )


def test_feature_table_no_pairs_omits_derived_columns():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()  # no ratios, no separations

    derived_cols = [c for c in df.columns if "_separation" in c or "_ratio" in c]
    assert len(derived_cols) == 0, f"Unexpected derived columns: {derived_cols}"


def test_feature_table_raises_when_no_fit_run():
    m = _make_raman_mapping()
    del m.fitted_params   # simulate missing fit state (triggers the guard)
    with pytest.raises(ValueError, match="fitted_params"):
        m.feature_table()


def test_feature_table_pl_mapping_ratios_and_separations():
    """PLMapping feature_table with both ratios and separations returns correct derived columns."""
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)

    df = m.feature_table(
        ratios=[("Exciton", "Trion")],
        separations=[("Exciton", "Trion")],
    )

    assert "Exciton_Trion_separation" in df.columns
    assert "Exciton_Trion_ratio" in df.columns
    assert len(df) == _Y_MAP * _X_MAP

    row = df[(df["x"] == 0) & (df["y"] == 0)].iloc[0]
    expected_sep = row["Exciton_position"] - row["Trion_position"]
    assert row["Exciton_Trion_separation"] == pytest.approx(expected_sep, rel=1e-9)

    expected_ratio = row["Exciton_peak_height"] / row["Trion_peak_height"]
    assert row["Exciton_Trion_ratio"] == pytest.approx(expected_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# v0.6.7: Component-area columns (Step 4)
# ---------------------------------------------------------------------------

def test_component_area_columns_present_raman():
    """component_area, _norm, and _fraction columns appear for every peak (RamanMapping)."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    for peak in ("E2g", "A1g"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, f"Missing {peak}{suffix}"


def test_component_area_columns_present_pl():
    """component_area, _norm, and _fraction columns appear for every peak (PLMapping)."""
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)
    df = m.feature_table()
    for peak in ("Trion", "Exciton"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, f"Missing {peak}{suffix}"


def test_component_area_norm_equals_amp_raman():
    """component_area_norm == amp for each peak (Lorentzian, Raman, intensity_scale=5.0)."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    # _GOOD_PARAMS_RAMAN = [386.0, 3.0, 2.0, 407.0, 4.0, 3.0] → amp_E2g=2.0, amp_A1g=3.0
    for row in df.itertuples():
        assert row.E2g_component_area_norm == pytest.approx(2.0, rel=1e-9)
        assert row.A1g_component_area_norm == pytest.approx(3.0, rel=1e-9)


def test_component_area_equals_amp_times_scale_raman():
    """component_area == amp * intensity_scale for each peak (intensity_scale=5.0)."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)   # norm_scale_map set to 5.0
    df = m.feature_table()
    scale = 5.0
    for row in df.itertuples():
        assert row.E2g_component_area == pytest.approx(2.0 * scale, rel=1e-9)
        assert row.A1g_component_area == pytest.approx(3.0 * scale, rel=1e-9)


def test_component_area_fraction_sums_to_one():
    """Fraction columns sum to 1.0 for every finite (good) pixel."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    for _, row in df.iterrows():
        total = row["E2g_component_area_fraction"] + row["A1g_component_area_fraction"]
        assert total == pytest.approx(1.0, rel=1e-9), f"Fraction sum={total} at row {_}"


def test_area_ratio_column_and_value():
    """area_ratios= adds {P1}_{P2}_area_ratio = amp[P1] / amp[P2]."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table(area_ratios=[("A1g", "E2g")])
    assert "A1g_E2g_area_ratio" in df.columns
    expected = 3.0 / 2.0   # amp_A1g / amp_E2g
    for row in df.itertuples():
        assert row.A1g_E2g_area_ratio == pytest.approx(expected, rel=1e-9)


def test_failed_pixel_component_area_all_nan():
    """Failed pixel row → NaN for all new component-area columns."""
    m = _make_raman_mapping()
    fail_px = (0, 0)
    _inject_fit(m, _GOOD_PARAMS_RAMAN, fail_pixels=[fail_px])
    df = m.feature_table(area_ratios=[("A1g", "E2g")])

    fail_row = df[(df["x"] == 0) & (df["y"] == 0)].iloc[0]
    for col in ("E2g_component_area", "E2g_component_area_norm", "E2g_component_area_fraction",
                "A1g_component_area", "A1g_component_area_norm", "A1g_component_area_fraction",
                "A1g_E2g_area_ratio"):
        assert math.isnan(fail_row[col]), f"Expected NaN for {col} on failed pixel"


def test_byte_parity_existing_columns_raman():
    """Existing feature-table columns are identical to the v0.6.6 golden CSV."""
    import pandas as pd
    golden = pd.read_csv(
        Path(__file__).parent.parent / "snapshot" / "featuretable_raman_map_v0.6.6.csv"
    )
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])

    shared_cols = [c for c in golden.columns if c in df.columns]
    pd.testing.assert_frame_equal(
        df[shared_cols].reset_index(drop=True),
        golden[shared_cols].reset_index(drop=True),
        rtol=1e-9, atol=0,
        check_like=False,
    )


def test_byte_parity_existing_columns_pl():
    """Existing feature-table columns are identical to the v0.6.6 golden CSV (PL mapping)."""
    import pandas as pd
    golden = pd.read_csv(
        Path(__file__).parent.parent / "snapshot" / "featuretable_pl_map_v0.6.6.csv"
    )
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)
    df = m.feature_table(ratios=[("Exciton", "Trion")], separations=[("Exciton", "Trion")])

    shared_cols = [c for c in golden.columns if c in df.columns]
    pd.testing.assert_frame_equal(
        df[shared_cols].reset_index(drop=True),
        golden[shared_cols].reset_index(drop=True),
        rtol=1e-9, atol=0,
        check_like=False,
    )
