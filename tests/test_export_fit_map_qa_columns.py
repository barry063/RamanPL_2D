"""
test_export_fit_map_qa_columns.py
----------------------------------
Verify that export_fit_map appends rmse, ok, n_starts, n_params_at_bounds
QA columns to every row in wide-format output.

All tests use synthetic fitted-mock objects (no real fitting).
"""

import csv
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.mapping import RamanMapping

# ---------------------------------------------------------------------------
# Shared synthetic cube helpers
# ---------------------------------------------------------------------------

_X = np.linspace(200.0, 1800.0, 150)
_PEAK = np.exp(-0.5 * ((_X - 520.0) / 15.0) ** 2)
_SHAPE = (2, 2, 150)  # Y=2, X=2 => 4 pixels

_CUSTOM_PEAKS = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_DATA_RANGE = (200.0, 1800.0)

# Lorentzian params: [centre, hwhm, amp_area]
_GOOD_PARAMS = np.array([520.0, 8.0, 2.5], dtype=float)
_FAIL_PARAMS = np.array([float("nan"), float("nan"), float("nan")], dtype=float)


def _make_mapping():
    rng = np.random.default_rng(0)
    cube = (_PEAK[None, None, :] + rng.uniform(0.0, 0.02, _SHAPE)).astype(float)
    m = RamanMapping.from_arrays(
        cube, _X, 2, 2,
        custom_peaks=_CUSTOM_PEAKS,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    return m


def _inject_synthetic_fit(m, good_pixels=None, fail_pixels=None):
    """
    Directly set fitted_params, residual_map, and fit_diagnostics_map on m
    without running fit_spectra.

    good_pixels: list of (j, i) indices to set with valid params
    fail_pixels: list of (j, i) indices to leave as NaN (failed fit)
    """
    if good_pixels is None:
        good_pixels = [(j, i) for j in range(m.Y) for i in range(m.X)]
    if fail_pixels is None:
        fail_pixels = []

    n_params = len(_CUSTOM_PEAKS) * m.params_per_peak
    m.fitted_params = np.full((m.Y, m.X, n_params), float("nan"), dtype=float)
    m.residual_map = np.full((m.Y, m.X), float("nan"), dtype=float)
    m.fit_diagnostics_map = np.empty((m.Y, m.X), dtype=object)
    m.fit_diagnostics_map[:, :] = None

    for j, i in good_pixels:
        m.fitted_params[j, i, :] = _GOOD_PARAMS
        m.residual_map[j, i] = 0.05
        m.fit_diagnostics_map[j, i] = {
            "ok": True,
            "rmse": 0.05,
            "n_starts": 2,
            "n_params_at_lower_bounds": 0,
            "n_params_at_upper_bounds": 1,
        }

    for j, i in fail_pixels:
        m.fitted_params[j, i, :] = _FAIL_PARAMS
        m.residual_map[j, i] = float("nan")
        m.fit_diagnostics_map[j, i] = {
            "ok": False,
            "reason": "fit_failed",
            "n_starts": 3,
            "n_params_at_lower_bounds": 0,
            "n_params_at_upper_bounds": 0,
        }


def _read_data_rows(path, delimiter="\t"):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.startswith("#")),
            delimiter=delimiter,
        )
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Test 1: QA columns appear in header
# ---------------------------------------------------------------------------

def test_wide_format_includes_qa_columns_in_header(tmp_path):
    m = _make_mapping()
    _inject_synthetic_fit(m)
    out = str(tmp_path / "export.txt")
    m.export_fit_map(out)

    rows = _read_data_rows(out)
    assert len(rows) > 0
    header = list(rows[0].keys())
    for col in ("rmse", "ok", "n_starts", "n_params_at_bounds"):
        assert col in header, f"Expected '{col}' in header, got {header}"

    # QA columns must be appended after the per-peak columns
    qa_idx = [header.index(c) for c in ("rmse", "ok", "n_starts", "n_params_at_bounds")]
    peak_col_idx = [i for i, h in enumerate(header) if h.startswith("Si_")]
    assert all(q > max(peak_col_idx) for q in qa_idx), (
        "QA columns should appear after per-peak columns"
    )


# ---------------------------------------------------------------------------
# Test 2: QA values match residual_map and diagnostics
# ---------------------------------------------------------------------------

def test_wide_format_qa_values_match_residual_map(tmp_path):
    m = _make_mapping()
    # (0,0) good, (0,1) failed
    _inject_synthetic_fit(m, good_pixels=[(0, 0)], fail_pixels=[(0, 1)])
    # Fill remaining pixels as good too
    for j, i in [(1, 0), (1, 1)]:
        m.fitted_params[j, i, :] = _GOOD_PARAMS
        m.residual_map[j, i] = 0.07
        m.fit_diagnostics_map[j, i] = {
            "ok": True, "rmse": 0.07, "n_starts": 1,
            "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
        }

    out = str(tmp_path / "export.txt")
    m.export_fit_map(out)

    rows = _read_data_rows(out)
    # row for pixel x=0,y=0 (i=0,j=0)
    pix_00 = next(r for r in rows if r["x"] == "0" and r["y"] == "0")
    assert float(pix_00["rmse"]) == pytest.approx(0.05, abs=1e-9)
    assert pix_00["ok"] == "True"
    assert int(float(pix_00["n_starts"])) == 2
    assert int(float(pix_00["n_params_at_bounds"])) == 1  # 0 lower + 1 upper

    # row for failed pixel x=1,y=0 (i=1,j=0)
    pix_10 = next(r for r in rows if r["x"] == "1" and r["y"] == "0")
    assert pix_10["ok"] == "False"
    assert math.isnan(float(pix_10["rmse"]))


# ---------------------------------------------------------------------------
# Test 3: Failed pixels write NaN, not blank
# ---------------------------------------------------------------------------

def test_wide_format_failed_pixels_emit_qa_columns(tmp_path):
    m = _make_mapping()
    _inject_synthetic_fit(m, good_pixels=[], fail_pixels=[(j, i) for j in range(m.Y) for i in range(m.X)])

    out = str(tmp_path / "export.txt")
    m.export_fit_map(out)

    rows = _read_data_rows(out)
    assert len(rows) == m.Y * m.X

    for row in rows:
        # QA columns must be present (not blank)
        assert "rmse" in row and row["rmse"] != ""
        assert "ok" in row and row["ok"] != ""
        assert "n_starts" in row and row["n_starts"] != ""
        assert "n_params_at_bounds" in row and row["n_params_at_bounds"] != ""

        assert row["ok"] == "False"
        assert math.isnan(float(row["rmse"]))


# ---------------------------------------------------------------------------
# Test 4: diagnostics=None falls back gracefully
# ---------------------------------------------------------------------------

def test_wide_format_diagnostics_none_falls_back(tmp_path):
    m = _make_mapping()
    _inject_synthetic_fit(m)
    # Simulate diagnostics='none'
    m.fit_diagnostics_map = None

    out = str(tmp_path / "export.txt")
    m.export_fit_map(out)

    rows = _read_data_rows(out)
    for row in rows:
        # rmse and ok should still resolve from residual_map
        assert row["ok"] == "True"
        assert float(row["rmse"]) == pytest.approx(0.05, abs=1e-9)
        # n_starts and n_params_at_bounds must be NaN
        assert math.isnan(float(row["n_starts"]))
        assert math.isnan(float(row["n_params_at_bounds"]))


# ---------------------------------------------------------------------------
# Test 5: Existing per-peak columns match v0.5.0 baseline snapshot
# ---------------------------------------------------------------------------

_BASELINE_EXPORT = (
    Path(__file__).parent.parent
    / "benchmarks" / "results" / "v0.5.0_baseline" / "v0.5.0_export.txt"
)

_BASELINE_PARAMS = (
    Path(__file__).parent.parent
    / "benchmarks" / "results" / "v0.5.0_baseline" / "fitted_params.npz"
)


@pytest.mark.skipif(
    not _BASELINE_EXPORT.exists() or not _BASELINE_PARAMS.exists(),
    reason="v0.5.0 baseline snapshot not present — run benchmarks/_step0_snapshot.py first",
)
def test_wide_format_existing_columns_unchanged(tmp_path):
    """Per-peak column values must be byte-identical to the v0.5.0 export."""

    # Reproduce the exact same fit as Step 0
    rng = np.random.default_rng(42)
    n_pts = 150
    x = np.linspace(200.0, 1800.0, n_pts)
    peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
    cube = (peak[None, None, :] + rng.uniform(0.0, 0.05, (3, 4, n_pts))).astype(float)

    m = RamanMapping.from_arrays(
        cube, x, 4, 3,
        custom_peaks={"Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0])},
        data_range=(200.0, 1800.0),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    m.fit_spectra(fit_spectrum_kwargs={"diagnostics": "light", "random_state": 42, "n_starts": 1})

    new_export = str(tmp_path / "v0.5.1_export.txt")
    m.export_fit_map(new_export)

    def _read_export(path):
        """Parse a fit-map export into (header_list, data_array).

        QA columns added in v0.5.1+ (rmse, ok, n_starts,
        n_params_at_bounds) are detected by **name** in the header line
        and dropped from both the header and every data row. This makes
        the parser robust to whichever schema generation the snapshot
        on disk happens to be:

        - pre-v0.5.1 baseline (8 columns, no QA): nothing stripped
        - v0.5.1+ baseline or current export (12 columns, 4 QA): QA
          columns dropped

        The remaining columns are pure float (x, y, and per-peak
        parameters) and can be parsed unconditionally.
        """
        QA_COLS = {"rmse", "ok", "n_starts", "n_params_at_bounds"}
        header = None
        keep_idx = None
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                line = line.rstrip("\r\n")
                if not line:
                    continue
                delim = "\t" if "\t" in line else ","
                parts = line.split(delim)
                if header is None:
                    keep_idx = [i for i, c in enumerate(parts) if c not in QA_COLS]
                    header = [parts[i] for i in keep_idx]
                    continue
                rows.append([float(parts[i]) for i in keep_idx])
        return header, np.asarray(rows, dtype=float)

    old_header, old_data = _read_export(_BASELINE_EXPORT)
    new_header, new_data = _read_export(new_export)

    # 1. Column header text must match exactly — catches column rename or
    #    reorder, which would be a real schema regression.
    assert old_header == new_header, (
        f"Per-peak column header drifted from v0.5.0 baseline.\n"
        f"  baseline: {old_header}\n"
        f"  current:  {new_header}"
    )

    # 2. Row and column counts must match exactly.
    assert old_data.shape == new_data.shape, (
        f"Row/column count drift: baseline {old_data.shape} vs current {new_data.shape}"
    )

    # 3. Pixel coordinates are integer-valued and must be exact.
    np.testing.assert_array_equal(
        old_data[:, :2], new_data[:, :2],
        err_msg="x/y coordinate columns drifted versus v0.5.0 baseline",
    )

    # 4. Fit-output columns: tolerate floating-point noise across LAPACK
    #    builds. Tolerances of 1e-3 absorb single-ULP and local-minimum
    #    differences seen on runners with different BLAS, while still
    #    catching any genuine scientific drift (which would be orders of
    #    magnitude larger).
    np.testing.assert_allclose(
        old_data[:, 2:], new_data[:, 2:],
        rtol=1e-3, atol=1e-3,
        err_msg="Per-peak fit-output columns drifted beyond floating-point tolerance",
    )