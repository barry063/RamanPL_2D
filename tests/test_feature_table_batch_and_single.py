"""
test_feature_table_batch_and_single.py
---------------------------------------
Tests for feature_table() on _BaseBatch (RamanBatch / PLBatch) and
RamanFit / PLfit (Step 3 — v0.5.2).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.batch import RamanBatch, PLBatch
from ramanpl.single_fit.RamanFit import RamanFit
from ramanpl.single_fit.PLfit import PLfit


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_X_RAMAN = np.linspace(350.0, 470.0, 121)
_X_PL = np.linspace(1.85, 2.10, 101)

_RAMAN_CUSTOM_PEAKS = {
    "A1g": ([390.0, 1.0, 0.001], [415.0, 15.0, 50.0]),
    "E12g": ([370.0, 1.0, 0.001], [395.0, 15.0, 50.0]),
}

_PL_CUSTOM_PEAKS = {
    "Trion":   ([1.88, 0.005, 0.001], [1.96, 0.10, 5.0]),
    "Exciton": ([1.96, 0.005, 0.001], [2.05, 0.10, 5.0]),
}


def _raman_spectrum(rng=None):
    """Synthetic two-peak Raman spectrum with small noise."""
    if rng is None:
        rng = np.random.default_rng(0)
    p1 = 5.0 * np.exp(-0.5 * ((_X_RAMAN - 405.0) / 4.0) ** 2)
    p2 = 3.0 * np.exp(-0.5 * ((_X_RAMAN - 382.0) / 3.0) ** 2)
    return p1 + p2 + rng.uniform(0.0, 0.1, _X_RAMAN.size)


def _pl_spectrum(rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    p1 = 2.0 * np.exp(-0.5 * ((_X_PL - 1.91) / 0.02) ** 2)
    p2 = 3.0 * np.exp(-0.5 * ((_X_PL - 2.00) / 0.03) ** 2)
    return p1 + p2 + rng.uniform(0.0, 0.05, _X_PL.size)


def _write_raman_txt(path, y):
    header = "wavenumber\tintensity"
    data = np.column_stack([_X_RAMAN, y])
    np.savetxt(str(path), data, delimiter="\t", header=header, comments="")


def _write_pl_txt(path, y):
    header = "energy\tintensity"
    data = np.column_stack([_X_PL, y])
    np.savetxt(str(path), data, delimiter="\t", header=header, comments="")


# ---------------------------------------------------------------------------
# Tests: RamanBatch
# ---------------------------------------------------------------------------

def test_raman_batch_feature_table_one_row_per_source(tmp_path):
    rng = np.random.default_rng(42)
    n = 4
    files = []
    for i in range(n):
        p = tmp_path / f"raman_{i}.txt"
        _write_raman_txt(p, _raman_spectrum(rng))
        files.append(str(p))

    b = RamanBatch(
        files,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    b.fit()
    df = b.feature_table()

    assert len(df) == n
    assert "source" in df.columns
    assert "A1g_position" in df.columns
    assert "E12g_fwhm" in df.columns
    assert "rmse" in df.columns
    assert "ok" in df.columns


def test_pl_batch_feature_table_one_row_per_source(tmp_path):
    rng = np.random.default_rng(43)
    n = 3
    files = []
    for i in range(n):
        p = tmp_path / f"pl_{i}.txt"
        _write_pl_txt(p, _pl_spectrum(rng))
        files.append(str(p))

    b = PLBatch(
        files,
        custom_peaks=_PL_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    b.fit()
    df = b.feature_table()

    assert len(df) == n
    assert "source" in df.columns
    assert "Exciton_position" in df.columns
    assert "Trion_peak_height" in df.columns


# ---------------------------------------------------------------------------
# Tests: RamanFit / PLfit single spectrum
# ---------------------------------------------------------------------------

def test_raman_fit_feature_table_single_row_dataframe():
    rng = np.random.default_rng(10)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    df = fitter.feature_table()

    assert len(df) == 1
    assert "A1g_position" in df.columns
    assert "E12g_fwhm" in df.columns
    assert "rmse" in df.columns
    assert "ok" in df.columns
    assert "n_starts" in df.columns
    assert "n_params_at_bounds" in df.columns


def test_pl_fit_feature_table_single_row_dataframe():
    rng = np.random.default_rng(11)
    y = _pl_spectrum(rng)
    fitter = PLfit(
        y, _X_PL,
        custom_peaks=_PL_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    df = fitter.feature_table()

    assert len(df) == 1
    assert "Exciton_position" in df.columns
    assert "Trion_peak_height" in df.columns
    assert "rmse" in df.columns


def test_raman_fit_feature_table_separation_matches_summary():
    """Separation column agrees with direct get_fitted_parameters() subtraction."""
    rng = np.random.default_rng(20)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()

    fitted = fitter.get_fitted_parameters()
    expected_sep = fitted["A1g"]["position"] - fitted["E12g"]["position"]

    df = fitter.feature_table(separations=[("A1g", "E12g")])
    actual_sep = float(df.iloc[0]["A1g_E12g_separation"])

    assert actual_sep == pytest.approx(expected_sep, rel=1e-9)


def test_feature_table_qa_columns_consistent_across_classes(tmp_path):
    """Same four QA column names on mapping, batch, and single-fit outputs."""
    from ramanpl.mapping import RamanMapping

    _QA_COLS = {"rmse", "ok", "n_starts", "n_params_at_bounds"}

    # --- mapping ---
    rng = np.random.default_rng(30)
    x_raman = np.linspace(300.0, 700.0, 80)
    peak = np.exp(-0.5 * ((x_raman - 386.0) / 3.0) ** 2)
    cube = (peak[None, None, :] + rng.uniform(0.0, 0.02, (2, 2, 80))).astype(float)
    m = RamanMapping.from_arrays(
        cube, x_raman, 2, 2,
        custom_peaks={"E2g": ([380.0, 1.0, 0.001], [392.0, 20.0, 5.0])},
        data_range=(300.0, 700.0),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    m.fit_spectra(fit_spectrum_kwargs={"n_starts": 1, "random_state": 0})
    mapping_cols = set(m.feature_table().columns)
    assert _QA_COLS.issubset(mapping_cols), f"Mapping missing QA cols: {_QA_COLS - mapping_cols}"

    # --- single-fit ---
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    sf_cols = set(fitter.feature_table().columns)
    assert _QA_COLS.issubset(sf_cols), f"Single-fit missing QA cols: {_QA_COLS - sf_cols}"

    # --- batch ---
    files = []
    for i in range(2):
        p = tmp_path / f"qa_spec_{i}.txt"
        _write_raman_txt(p, _raman_spectrum(rng))
        files.append(str(p))
    b = RamanBatch(files, custom_peaks=_RAMAN_CUSTOM_PEAKS, smoothing=False, background_remove=False)
    b.fit()
    batch_cols = set(b.feature_table().columns)
    assert _QA_COLS.issubset(batch_cols), f"Batch missing QA cols: {_QA_COLS - batch_cols}"


# ---------------------------------------------------------------------------
# v0.6.7: Component-area columns — single-fit and batch (Step 4)
# ---------------------------------------------------------------------------

def test_raman_fit_component_area_columns_present():
    """RamanFit.feature_table() includes all three new component-area columns."""
    rng = np.random.default_rng(50)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    df = fitter.feature_table()
    for peak in ("A1g", "E12g"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, f"Missing {peak}{suffix}"


def test_pl_fit_component_area_columns_present():
    """PLfit.feature_table() includes all three new component-area columns."""
    rng = np.random.default_rng(51)
    y = _pl_spectrum(rng)
    fitter = PLfit(
        y, _X_PL,
        custom_peaks=_PL_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    df = fitter.feature_table()
    for peak in ("Trion", "Exciton"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, f"Missing {peak}{suffix}"


def test_raman_fit_component_area_norm_equals_amp():
    """component_area_norm == amp from get_fitted_parameters()."""
    rng = np.random.default_rng(52)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    fitted = fitter.get_fitted_parameters()
    df = fitter.feature_table()
    row = df.iloc[0]
    for peak in ("A1g", "E12g"):
        assert row[f"{peak}_component_area_norm"] == pytest.approx(fitted[peak]["amp"], rel=1e-9)


def test_raman_fit_component_area_equals_amp_times_scale():
    """component_area == amp * peak_intensity (= amp_scaled)."""
    rng = np.random.default_rng(53)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    fitted = fitter.get_fitted_parameters()
    df = fitter.feature_table()
    row = df.iloc[0]
    for peak in ("A1g", "E12g"):
        assert row[f"{peak}_component_area"] == pytest.approx(fitted[peak]["amp_scaled"], rel=1e-9)


def test_raman_fit_fraction_sums_to_one():
    """component_area_fraction sums to 1.0 for a single-fit row."""
    rng = np.random.default_rng(54)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    df = fitter.feature_table()
    row = df.iloc[0]
    total = row["A1g_component_area_fraction"] + row["E12g_component_area_fraction"]
    assert total == pytest.approx(1.0, rel=1e-9)


def test_raman_fit_area_ratio_column():
    """area_ratios= keyword adds {P1}_{P2}_area_ratio column."""
    rng = np.random.default_rng(55)
    y = _raman_spectrum(rng)
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    fitted = fitter.get_fitted_parameters()
    df = fitter.feature_table(area_ratios=[("A1g", "E12g")])
    expected = fitted["A1g"]["amp"] / fitted["E12g"]["amp"]
    assert df.iloc[0]["A1g_E12g_area_ratio"] == pytest.approx(expected, rel=1e-9)


def test_batch_component_area_columns_present(tmp_path):
    """RamanBatch.feature_table() includes component-area columns."""
    rng = np.random.default_rng(56)
    files = []
    for i in range(2):
        p = tmp_path / f"comp_{i}.txt"
        _write_raman_txt(p, _raman_spectrum(rng))
        files.append(str(p))
    b = RamanBatch(files, custom_peaks=_RAMAN_CUSTOM_PEAKS, smoothing=False, background_remove=False)
    b.fit()
    df = b.feature_table()
    for peak in ("A1g", "E12g"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, f"Missing {peak}{suffix}"


def test_byte_parity_existing_columns_single(tmp_path):
    """Existing feature-table columns for single-fit match v0.6.6 golden CSV."""
    import pandas as pd
    from ramanpl.mapping import RamanMapping

    golden = pd.read_csv(
        Path(__file__).parent.parent / "snapshot" / "featuretable_single_v0.6.6.csv"
    )

    # Reproduce same fixture as gen_goldens.py
    n_pts = 80
    x_raman = np.linspace(300.0, 700.0, n_pts)
    custom_peaks_raman = {
        "E2g": ([380.0, 1.0, 0.001], [392.0, 20.0, 5.0]),
        "A1g": ([402.0, 1.0, 0.001], [412.0, 20.0, 5.0]),
    }
    good_params = np.array([386.0, 3.0, 2.0, 407.0, 4.0, 3.0], dtype=float)
    peak1 = np.exp(-0.5 * ((x_raman - 386.0) / 3.0) ** 2)
    peak2 = np.exp(-0.5 * ((x_raman - 407.0) / 4.0) ** 2)

    rf = RamanFit(
        x_raman, peak1 + peak2,
        custom_peaks=custom_peaks_raman,
        normalize=False,
        background_remove=False,
        smoothing=False,
    )
    rf.params_fit = good_params.copy()
    rf.peak_intensity = 1.0
    rf.fit_diagnostics = {"rmse": 0.04, "n_starts": 1, "n_params_at_bounds": 0}
    df = rf.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])

    shared_cols = [c for c in golden.columns if c in df.columns]
    pd.testing.assert_frame_equal(
        df[shared_cols].reset_index(drop=True),
        golden[shared_cols].reset_index(drop=True),
        rtol=1e-9, atol=0,
        check_like=False,
    )


def test_cross_path_component_area_single_vs_batch(tmp_path):
    """Single-fit and batch produce identical component-area columns for the same spectrum."""
    rng = np.random.default_rng(60)
    y = _raman_spectrum(rng)
    spec_file = tmp_path / "cross_path.txt"
    _write_raman_txt(spec_file, y)

    # Single-fit
    fitter = RamanFit(
        y, _X_RAMAN,
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    fitter.fit_spectrum()
    sf_df = fitter.feature_table()

    # Batch (same spectrum)
    b = RamanBatch(
        [str(spec_file)],
        custom_peaks=_RAMAN_CUSTOM_PEAKS,
        smoothing=False,
        background_remove=False,
    )
    b.fit()
    batch_df = b.feature_table()

    area_cols = [c for c in sf_df.columns if "component_area" in c]
    for col in area_cols:
        assert col in batch_df.columns, f"Batch missing column {col}"
        sf_val = float(sf_df.iloc[0][col])
        batch_val = float(batch_df.iloc[0][col])
        if math.isnan(sf_val):
            assert math.isnan(batch_val)
        else:
            assert sf_val == pytest.approx(batch_val, rel=1e-6), (
                f"Cross-path mismatch for {col}: single={sf_val}, batch={batch_val}"
            )
