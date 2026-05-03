"""
test_export_fit_map_long_format.py
------------------------------------
Verify the long=True option of export_fit_map (one row per pixel×peak).

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
# Shared synthetic cube helpers  (mirrors test_export_fit_map_qa_columns.py)
# ---------------------------------------------------------------------------

_X = np.linspace(200.0, 1800.0, 150)
_PEAK = np.exp(-0.5 * ((_X - 520.0) / 15.0) ** 2)

_CUSTOM_PEAKS_1 = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_CUSTOM_PEAKS_2 = {
    "A": ([490.0, 1.0, 0.001], [540.0, 20.0, 3.0]),
    "B": ([560.0, 1.0, 0.001], [620.0, 20.0, 3.0]),
}
_DATA_RANGE = (200.0, 1800.0)

_GOOD_PARAMS_1 = np.array([520.0, 8.0, 2.5], dtype=float)
_GOOD_PARAMS_2 = np.array([515.0, 7.0, 2.0, 590.0, 9.0, 2.2], dtype=float)


def _make_mapping(Y, X, custom_peaks):
    rng = np.random.default_rng(1)
    cube = (_PEAK[None, None, :] + rng.uniform(0.0, 0.02, (Y, X, 150))).astype(float)
    m = RamanMapping.from_arrays(
        cube, _X, X, Y,
        custom_peaks=custom_peaks,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    return m


def _inject_all_good(m, good_params):
    n_params = len(good_params)
    m.fitted_params = np.tile(
        good_params, (m.Y, m.X, 1)
    ).reshape(m.Y, m.X, n_params)
    m.residual_map = np.full((m.Y, m.X), 0.04, dtype=float)
    m.fit_diagnostics_map = np.empty((m.Y, m.X), dtype=object)
    for j in range(m.Y):
        for i in range(m.X):
            m.fit_diagnostics_map[j, i] = {
                "ok": True, "rmse": 0.04, "n_starts": 1,
                "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
            }


def _inject_one_failed(m, good_params, fail_j, fail_i):
    _inject_all_good(m, good_params)
    m.fitted_params[fail_j, fail_i, :] = float("nan")
    m.residual_map[fail_j, fail_i] = float("nan")
    m.fit_diagnostics_map[fail_j, fail_i] = {
        "ok": False, "reason": "fit_failed", "n_starts": 2,
        "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
    }


def _read_long_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.startswith("#")),
            delimiter="\t",
        )
        for row in reader:
            rows.append(row)
    return rows


def _read_meta(path):
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line.lstrip("# ").rstrip("\r\n")
            if ": " in line:
                k, _, v = line.partition(": ")
                meta[k.strip()] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Test 1: row count = Y * X * N_peaks
# ---------------------------------------------------------------------------

def test_long_format_row_count(tmp_path):
    Y, X, n_peaks = 3, 4, 2
    m = _make_mapping(Y, X, _CUSTOM_PEAKS_2)
    _inject_all_good(m, _GOOD_PARAMS_2)

    out = str(tmp_path / "export_long.txt")
    m.export_fit_map(out, long=True)

    rows = _read_long_rows(out)
    assert len(rows) == Y * X * n_peaks, (
        f"Expected {Y * X * n_peaks} rows, got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# Test 2: column schema matches documented schema
# ---------------------------------------------------------------------------

def test_long_format_column_schema(tmp_path):
    m = _make_mapping(2, 2, _CUSTOM_PEAKS_1)
    _inject_all_good(m, _GOOD_PARAMS_1)

    out = str(tmp_path / "export_long.txt")
    m.export_fit_map(out, long=True)

    rows = _read_long_rows(out)
    assert len(rows) > 0
    header = list(rows[0].keys())

    expected_prefix = ["x", "y", "peak"]
    expected_peak_fields = ["centre", "fwhm", "peak_height", "peak_height_norm", "amp", "scale"]
    expected_qa = ["rmse", "ok", "n_starts", "n_params_at_bounds"]
    expected = expected_prefix + expected_peak_fields + expected_qa

    assert header == expected, f"Header mismatch:\n  expected={expected}\n  got={header}"


# ---------------------------------------------------------------------------
# Test 3: pivot back to wide reproduces same peak values as wide export
# ---------------------------------------------------------------------------

def test_long_format_pixel_peak_round_trip(tmp_path):
    m = _make_mapping(2, 3, _CUSTOM_PEAKS_1)
    _inject_all_good(m, _GOOD_PARAMS_1)

    wide_out = str(tmp_path / "wide.txt")
    long_out = str(tmp_path / "long.txt")
    m.export_fit_map(wide_out, long=False)
    m.export_fit_map(long_out, long=True)

    # Read wide rows (skip QA columns for comparison)
    def _read_wide_peak_values(path):
        result = {}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(
                (line for line in fh if not line.startswith("#")),
                delimiter="\t",
            )
            for row in reader:
                key = (row["x"], row["y"])
                result[key] = {k: v for k, v in row.items()
                               if k not in ("x", "y", "rmse", "ok", "n_starts", "n_params_at_bounds")}
        return result

    wide_vals = _read_wide_peak_values(wide_out)

    # Read long rows and reconstruct wide-like structure
    long_rows = _read_long_rows(long_out)
    long_vals = {}
    for row in long_rows:
        key = (row["x"], row["y"])
        peak = row["peak"]
        if key not in long_vals:
            long_vals[key] = {}
        for field in ("centre", "fwhm", "peak_height", "peak_height_norm", "amp", "scale"):
            long_vals[key][f"{peak}_{field}"] = row[field]

    assert set(wide_vals.keys()) == set(long_vals.keys())
    for key in wide_vals:
        for col, v in wide_vals[key].items():
            assert long_vals[key].get(col) == v, (
                f"Mismatch at pixel {key}, column {col}: wide={v!r} long={long_vals[key].get(col)!r}"
            )


# ---------------------------------------------------------------------------
# Test 4: failed pixel emits N_peaks rows with NaN parameters
# ---------------------------------------------------------------------------

def test_long_format_failed_pixels_emit_n_peaks_rows(tmp_path):
    Y, X, n_peaks = 2, 2, 2
    m = _make_mapping(Y, X, _CUSTOM_PEAKS_2)
    _inject_one_failed(m, _GOOD_PARAMS_2, fail_j=0, fail_i=1)

    out = str(tmp_path / "export_long.txt")
    m.export_fit_map(out, long=True)

    rows = _read_long_rows(out)

    # Total rows: Y*X*n_peaks (failed pixel still contributes n_peaks rows)
    assert len(rows) == Y * X * n_peaks

    # Failed pixel rows: x=1, y=0 (i=1, j=0)
    fail_rows = [r for r in rows if r["x"] == "1" and r["y"] == "0"]
    assert len(fail_rows) == n_peaks, (
        f"Failed pixel should emit {n_peaks} rows, got {len(fail_rows)}"
    )
    for fr in fail_rows:
        assert math.isnan(float(fr["centre"]))
        assert fr["ok"] == "False"


# ---------------------------------------------------------------------------
# Test 5: metadata header records export_format = long
# ---------------------------------------------------------------------------

def test_long_format_metadata_header_records_export_format(tmp_path):
    m = _make_mapping(2, 2, _CUSTOM_PEAKS_1)
    _inject_all_good(m, _GOOD_PARAMS_1)

    long_out = str(tmp_path / "long.txt")
    wide_out = str(tmp_path / "wide.txt")
    m.export_fit_map(long_out, long=True)
    m.export_fit_map(wide_out, long=False)

    meta_long = _read_meta(long_out)
    meta_wide = _read_meta(wide_out)

    # _meta_value_to_text uses json.dumps, so string values are JSON-quoted
    assert meta_long.get("export_format") in ("long", '"long"'), (
        f"Expected export_format long, got {meta_long.get('export_format')!r}"
    )
    assert meta_wide.get("export_format") in ("wide", '"wide"'), (
        f"Expected export_format wide, got {meta_wide.get('export_format')!r}"
    )
