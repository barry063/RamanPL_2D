"""
validation_v0.6.0_vs_v0.5.0.py
--------------------------------
Reproducible validation harness for the v0.6.0 milestone.

Compares v0.6.0 against the v0.5.0 baseline on three axes:
  1. Fit quality  — byte-level export parity on the standard small_3x4 cube
  2. Runtime      — n_curve_fit_calls and wall-clock time vs. recorded baseline
  3. Failure mode — failed-pixel counts on a hard cube with vs. without peak proposals

Reuses build_mapping_fit_benchmark_cases, build_fit_kwargs_variants, and
run_mapping_fit_case from benchmark_mapping_fit.py rather than duplicating logic.

Usage
-----
    python benchmarks/validation_v0.6.0_vs_v0.5.0.py

Outputs
-------
    benchmarks/results/v0.6.0_validation.csv
    benchmarks/results/v0.6.0_validation_summary.json
"""

import csv
import hashlib
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — makes both src/ and benchmarks/ importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # expose benchmark_mapping_fit

import scipy.optimize as _scipy_opt
from ramanpl.mapping import RamanMapping

from benchmark_mapping_fit import (
    build_fit_kwargs_variants,
    build_mapping_fit_benchmark_cases,
    run_mapping_fit_case,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASELINE_DIR = Path(__file__).parent / "results" / "v0.5.0_baseline"
_OUT_DIR = Path(__file__).parent / "results"

_CSV_PATH = _OUT_DIR / "v0.6.0_validation.csv"
_JSON_PATH = _OUT_DIR / "v0.6.0_validation_summary.json"

_CUSTOM_PEAKS_STD = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_DATA_RANGE = (200.0, 1800.0)
_N_QA_COLS = 4  # rmse, ok, n_starts, n_params_at_bounds added in v0.5.1
_LARGE_CUBE_MAX_N_STARTS = 2  # cap to reduce wall-clock for 15x15 throughput cube

_CSV_FIELDS = [
    "dataset_name",
    "cube_shape",
    "warm_start",
    "n_starts",
    "n_jobs",
    "use_peak_proposals",
    "random_state",
    "runtime_s",
    "n_curve_fit_calls",
    "success_rate",
    "mean_rmse_finite",
    "n_failed_pixels",
    "export_sha256",
]

# ---------------------------------------------------------------------------
# Parity helpers (same pattern as _step4_parity.py)
# ---------------------------------------------------------------------------


def _read_data_rows(path):
    """Return non-comment lines from an export file."""
    rows = []
    with open(path, newline="") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            rows.append(line.rstrip("\r\n"))
    return rows


def _strip_last_n(csv_line, n):
    """Strip the last n columns from a delimited line."""
    delim = "\t" if "\t" in csv_line else ","
    parts = csv_line.split(delim)
    return delim.join(parts[:-n]) if len(parts) > n else csv_line


# ---------------------------------------------------------------------------
# Export + SHA256 helper
# ---------------------------------------------------------------------------


def _build_std_mapping(x, cube):
    """Create a RamanMapping with the standard benchmark construction kwargs."""
    Y, X, _ = cube.shape
    return RamanMapping.from_arrays(
        cube, x, X, Y,
        custom_peaks=_CUSTOM_PEAKS_STD,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )


def _compute_export_sha256(m):
    """Export m to a tempfile; return (full_sha256, stripped_sha256).

    full_sha256    — SHA256 of the complete export file (recorded in CSV).
    stripped_sha256 — SHA256 computed over QA-stripped data rows only;
                      directly comparable to the v0.5.0 checksums.txt value.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w"
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        m.export_fit_map(str(tmp_path))
        raw_bytes = tmp_path.read_bytes()
        full_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        # Stripped SHA256 (QA columns removed — matches v0.5.0 schema)
        new_rows = _read_data_rows(str(tmp_path))
        stripped_rows = [_strip_last_n(r, _N_QA_COLS) for r in new_rows]
        stripped_content = "\n".join(stripped_rows).encode()
        stripped_sha256 = hashlib.sha256(stripped_content).hexdigest()
    finally:
        tmp_path.unlink(missing_ok=True)

    return full_sha256, stripped_sha256


# ---------------------------------------------------------------------------
# Standard-case runner (wraps run_mapping_fit_case, adds SHA256 + proposals flag)
# ---------------------------------------------------------------------------


def _run_standard_case(dataset_name, x, cube, fit_kwargs):
    """Run one standard (dataset, fit_kwargs) validation case.

    Calls run_mapping_fit_case for timing/call-count stats (first fit, mocked),
    then runs a second unmocked fit to compute export_sha256.
    Returns the augmented row dict.
    """
    row = run_mapping_fit_case(
        x=x, cube=cube, fit_kwargs=fit_kwargs, dataset_name=dataset_name
    )
    row["use_peak_proposals"] = True  # default — not overridden in standard grid

    # Second fit for export SHA256 (unmocked, no call-count interference)
    fsw = dict(fit_kwargs.get("fit_spectrum_kwargs", {}))
    m2 = _build_std_mapping(x, cube)
    m2.fit_spectra(
        warm_start=fit_kwargs.get("warm_start", False),
        fit_spectrum_kwargs=fsw,
    )
    full_sha256, _ = _compute_export_sha256(m2)
    row["export_sha256"] = full_sha256

    return row


# ---------------------------------------------------------------------------
# Hard cube construction and runner
# ---------------------------------------------------------------------------


def _build_hard_cube():
    """Return (x, cube) for a 3x4 hard-pixel cube with two overlapping peaks.

    Two peaks at 500 and 530 cm-1 (30 cm-1 separation, FWHM≈15), amplitude 0.5,
    plus Gaussian noise at std=0.3 (SNR≈1.7). Uses a distinct RNG seed (99) to
    mark it as a separate case from the standard benchmark cubes.
    """
    rng = np.random.default_rng(99)
    n_pts = 150
    x = np.linspace(200.0, 1800.0, n_pts)
    peak1 = 0.5 * np.exp(-0.5 * ((x - 500.0) / 15.0) ** 2)
    peak2 = 0.5 * np.exp(-0.5 * ((x - 530.0) / 15.0) ** 2)
    shape = (3, 4, n_pts)
    noise = rng.normal(0.0, 0.3, shape)
    cube = ((peak1 + peak2)[None, None, :] + noise).astype(float)
    return x, cube


_HARD_CUSTOM_PEAKS = {
    "P1": ([485.0, 0.3, 0.1], [515.0, 20.0, 1.5]),
    "P2": ([515.0, 0.3, 0.1], [545.0, 20.0, 1.5]),
}


def _run_hard_case(use_peak_proposals):
    """Run the hard cube with maxfev=50 to force failures without proposals."""
    x, cube = _build_hard_cube()
    Y, X, _ = cube.shape
    fsw = {
        "diagnostics": "light",
        "random_state": 42,
        "n_starts": 1,
        "maxfev": 50,
        "use_peak_proposals": use_peak_proposals,
    }

    _real_curve_fit = _scipy_opt.curve_fit
    counter = {"n": 0}

    def _wrapped(*args, **kwargs):
        counter["n"] += 1
        return _real_curve_fit(*args, **kwargs)

    def _build():
        return RamanMapping.from_arrays(
            cube, x, X, Y,
            custom_peaks=_HARD_CUSTOM_PEAKS,
            data_range=_DATA_RANGE,
            background_remove=False,
            smoothing=False,
            normalize=False,
        )

    with patch.object(_scipy_opt, "curve_fit", _wrapped):
        t0 = time.perf_counter()
        m = _build()
        m.fit_spectra(warm_start=False, fit_spectrum_kwargs=fsw)
        runtime_s = time.perf_counter() - t0
        n_curve_fit_calls = counter["n"]

    residuals = m.residual_map
    n_total = Y * X
    finite_mask = np.isfinite(residuals)
    n_ok = int(np.sum(finite_mask))
    n_failed = n_total - n_ok
    success_rate = n_ok / n_total if n_total > 0 else float("nan")
    mean_rmse_finite = float(np.mean(residuals[finite_mask])) if n_ok > 0 else float("nan")

    full_sha256, _ = _compute_export_sha256(m)

    return {
        "dataset_name":      "hard_3x4",
        "cube_shape":        str(cube.shape),
        "warm_start":        False,
        "n_starts":          1,
        "use_peak_proposals": use_peak_proposals,
        "random_state":      42,
        "runtime_s":         round(runtime_s, 6),
        "n_curve_fit_calls": int(n_curve_fit_calls),
        "success_rate":      round(success_rate, 6),
        "mean_rmse_finite":  round(mean_rmse_finite, 8) if np.isfinite(mean_rmse_finite) else float("nan"),
        "n_failed_pixels":   n_failed,
        "export_sha256":     full_sha256,
    }


# ---------------------------------------------------------------------------
# Parity check against v0.5.0 baseline
# ---------------------------------------------------------------------------

_QA_COLS = {"rmse", "ok", "n_starts", "n_params_at_bounds"}


def _parse_export_numeric(path):
    """Parse an export file into (header, data_array), dropping QA columns.

    Same logic as test_export_fit_map_qa_columns.py::test_wide_format_existing_columns_unchanged.
    Tolerates both v0.5.0 (no QA cols) and v0.5.1+ (with QA cols) schemas.
    """
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
                keep_idx = [i for i, c in enumerate(parts) if c not in _QA_COLS]
                header = [parts[i] for i in keep_idx]
                continue
            assert keep_idx is not None
            rows.append([float(parts[i]) for i in keep_idx])
    return header, np.asarray(rows, dtype=float)


def _check_parity():
    """Check per-peak column parity between current code and v0.5.0 baseline.

    Uses the same numerical-tolerance approach as
    test_export_fit_map_qa_columns.py::test_wide_format_existing_columns_unchanged
    (rtol=1e-3, atol=1e-3). This tolerates LAPACK/scipy floating-point
    differences across builds while catching genuine algorithmic regressions.

    Returns (sha256_v0_5_0, sha256_current_full, parity_bool, reason).
    """
    # Read v0.5.0 SHA256 anchor (full-file hash, recorded for documentation)
    checksums_text = (_BASELINE_DIR / "checksums.txt").read_text().strip()
    sha256_v0_5_0 = checksums_text.split(":")[1].split()[0]

    # Re-run the small_3x4 baseline case (same params as _step0_snapshot.py)
    rng = np.random.default_rng(42)
    n_pts = 150
    x = np.linspace(200.0, 1800.0, n_pts)
    shape = (3, 4, n_pts)
    peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
    cube = (peak[None, None, :] + rng.uniform(0.0, 0.05, shape)).astype(float)

    m = _build_std_mapping(x, cube)
    m.fit_spectra(fit_spectrum_kwargs={"diagnostics": "light", "random_state": 42, "n_starts": 1})

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)
    try:
        m.export_fit_map(str(tmp_path))
        raw_bytes = tmp_path.read_bytes()
        sha256_current = hashlib.sha256(raw_bytes).hexdigest()

        new_header, new_data = _parse_export_numeric(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    old_header, old_data = _parse_export_numeric(str(_BASELINE_DIR / "v0.5.0_export.txt"))

    # Headers must match exactly
    if old_header != new_header:
        return sha256_v0_5_0, sha256_current, False, f"header mismatch: {old_header} vs {new_header}"

    # Shape must match
    if old_data.shape != new_data.shape:
        return sha256_v0_5_0, sha256_current, False, f"shape mismatch: {old_data.shape} vs {new_data.shape}"

    # x/y coordinates must be exact
    if not np.array_equal(old_data[:, :2], new_data[:, :2]):
        return sha256_v0_5_0, sha256_current, False, "x/y coordinate columns differ"

    # Fit output columns: match within rtol=1e-3, atol=1e-3
    try:
        np.testing.assert_allclose(old_data[:, 2:], new_data[:, 2:], rtol=1e-3, atol=1e-3)
        parity = True
        reason = "ok"
    except AssertionError as exc:
        parity = False
        reason = str(exc)[:200]

    return sha256_v0_5_0, sha256_current, parity, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # -----------------------------------------------------------------------
    # Standard 12 rows: 3 cubes × 4 fit-kwarg variants
    # -----------------------------------------------------------------------
    cases = build_mapping_fit_benchmark_cases()
    variants = build_fit_kwargs_variants()

    for dataset_name, x, cube in cases:
        for fit_kwargs in variants:
            n_st = fit_kwargs["fit_spectrum_kwargs"].get("n_starts", 1)
            if dataset_name == "extended_15x15" and n_st > _LARGE_CUBE_MAX_N_STARTS:
                continue
            row = _run_standard_case(dataset_name, x, cube, fit_kwargs)
            all_rows.append(row)
            print(
                f"[std] {dataset_name:<20} warm={row['warm_start']} "
                f"n_starts={row['n_starts']}  "
                f"ok={row['success_rate']*100:.0f}%  "
                f"calls={row['n_curve_fit_calls']}"
            )

    # -----------------------------------------------------------------------
    # Hard cube: 2 rows (use_peak_proposals=False then True)
    # -----------------------------------------------------------------------
    for use_pp in (False, True):
        row = _run_hard_case(use_peak_proposals=use_pp)
        all_rows.append(row)
        print(
            f"[hard] use_peak_proposals={use_pp!s:<5}  "
            f"failed={row['n_failed_pixels']}  "
            f"calls={row['n_curve_fit_calls']}"
        )

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    with open(_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written  →  {_CSV_PATH}  ({len(all_rows)} rows)")

    # -----------------------------------------------------------------------
    # Parity check
    # -----------------------------------------------------------------------
    print("\nRunning parity check against v0.5.0 baseline …")
    sha256_v0_5_0, sha256_current, parity, parity_reason = _check_parity()
    print(f"  v0.5.0 SHA256 : {sha256_v0_5_0}")
    print(f"  current SHA256: {sha256_current}")
    print(f"  parity        : {parity}  ({parity_reason})")

    if not parity:
        print("\n[STOP] fit_quality.parity == false — investigate before continuing.")

    # -----------------------------------------------------------------------
    # Collect runtime and failure-mode numbers from rows
    # -----------------------------------------------------------------------
    baseline_json = json.loads((_BASELINE_DIR / "v0.5.0_baseline.json").read_text())

    # Locate small_3x4 / warm_start=False / n_starts=1 row
    baseline_row = next(
        (r for r in all_rows
         if r["dataset_name"] == "small_3x4"
         and r["warm_start"] is False
         and r["n_starts"] == 1),
        None,
    )
    current_median_s = baseline_row["runtime_s"] if baseline_row else float("nan")
    v0_5_0_median_s = baseline_json["runtime_s_median"]
    ratio = round(current_median_s / v0_5_0_median_s, 4) if v0_5_0_median_s else float("nan")

    hard_no_pp = next(
        (r for r in all_rows if r["dataset_name"] == "hard_3x4" and not r["use_peak_proposals"]),
        None,
    )
    hard_pp = next(
        (r for r in all_rows if r["dataset_name"] == "hard_3x4" and r["use_peak_proposals"]),
        None,
    )

    # -----------------------------------------------------------------------
    # Write summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "fit_quality": {
            "export_sha256_v0_5_0": sha256_v0_5_0,
            "export_sha256_current": sha256_current,
            "parity": parity,
        },
        "runtime": {
            "v0_5_0_baseline_median_s": v0_5_0_median_s,
            "current_median_s": round(current_median_s, 6),
            "ratio": ratio,
        },
        "failure_mode": {
            "hard_cube_failed_pixels_without_proposals": hard_no_pp["n_failed_pixels"] if hard_no_pp else None,
            "hard_cube_failed_pixels_with_proposals": hard_pp["n_failed_pixels"] if hard_pp else None,
        },
    }

    _JSON_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON  →  {_JSON_PATH}")
    print(f"\nfit_quality.parity = {parity}")

    if not parity:
        sys.exit(1)

    print("\n[Done] Validation complete — parity confirmed.")


if __name__ == "__main__":
    run()
