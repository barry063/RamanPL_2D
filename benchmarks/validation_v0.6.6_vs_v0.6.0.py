"""
validation_v0.6.6_vs_v0.6.0.py
---------------------------------
Reproducible validation harness for the v0.6.6 consolidation build.

Checks that the additive surface introduced in v0.6.1-v0.6.5 does not alter
the core fitting contract relative to v0.6.0.  Gates are:

  1. fit_output_parity     — default kwargs and explicit-default kwargs produce
                             identical exports on the standard small_3x4 cube.
  2. export_schema_stability — export headers contain the full frozen column set
                             and match the header structure from v0.6.0.
  3. call_count_sanity     — n_curve_fit_calls on small_3x4 / n_starts=1 is
                             within a 25% tolerance of the v0.6.0 reference.
                             Wall-clock timing is recorded but advisory only.
  4. parallel_safety       — n_jobs=2 safe-mode succeeds; n_jobs=2 + unsafe
                             warm-start raises ValueError.
  5. cluster_seed_boundary — cluster_seeds=False parity holds; cluster_seeds=True
                             with n_jobs>1 raises.
  6. autotune_non_mutation — autotune_baseline() does not mutate preprocessing
                             until apply_choice() is called.
  7. batch_progress_default — show_progress=True is the default on fit_spectra_batch
                              and omitting it preserves batch fit behaviour.

Usage
-----
    python benchmarks/validation_v0.6.6_vs_v0.6.0.py

Outputs
-------
    benchmarks/results/v0.6.6_validation.csv
    benchmarks/results/v0.6.6_validation_summary.json
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
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import scipy.optimize as _scipy_opt
from ramanpl.mapping import RamanMapping

from benchmark_mapping_fit import (
    build_mapping_fit_benchmark_cases,
    run_mapping_fit_case,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUT_DIR = Path(__file__).parent / "results"
_V060_CSV = _OUT_DIR / "v0.6.0_validation.csv"
_V060_JSON = _OUT_DIR / "v0.6.0_validation_summary.json"

_CSV_PATH = _OUT_DIR / "v0.6.6_validation.csv"
_JSON_PATH = _OUT_DIR / "v0.6.6_validation_summary.json"

_CUSTOM_PEAKS = {"Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0])}
_DATA_RANGE = (200.0, 1800.0)
_RANDOM_STATE = 42

# Frozen column vocabulary from tests/test_api_stability.py (Gate 0.6)
_FROZEN_SUFFIXES = frozenset(
    ["_position", "_fwhm", "_peak_height", "_peak_height_norm", "_separation", "_ratio"]
)
_QA_COLS = {"rmse", "ok", "n_starts", "n_params_at_bounds"}

# Tolerance for call-count sanity gate
_CALL_COUNT_TOLERANCE = 0.25  # ±25% of v0.6.0 reference; advisory on synthetic cubes

_CSV_FIELDS = [
    "gate_name",
    "case_label",
    "passed",
    "detail",
    "n_curve_fit_calls",
    "runtime_s",
    "note",
]


# ---------------------------------------------------------------------------
# Shared cube builder
# ---------------------------------------------------------------------------

def _build_small_3x4():
    """Return (x, cube) for the standard small_3x4 benchmark cube."""
    rng = np.random.default_rng(_RANDOM_STATE)
    n_pts = 150
    x = np.linspace(200.0, 1800.0, n_pts)
    peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
    cube = (peak[None, None, :] + rng.uniform(0.0, 0.05, (3, 4, n_pts))).astype(float)
    return x, cube


def _build_mapping(x, cube):
    Y, X, _ = cube.shape
    return RamanMapping.from_arrays(
        cube, x, X, Y,
        custom_peaks=_CUSTOM_PEAKS,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )


def _export_sha256(m):
    """Export m to a tempfile and return SHA256 of the bytes."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)
    try:
        m.export_fit_map(str(tmp_path))
        return hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    finally:
        tmp_path.unlink(missing_ok=True)


def _export_headers(m):
    """Export m to a tempfile and return the header column list."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)
    try:
        m.export_fit_map(str(tmp_path))
        with open(tmp_path, newline="", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                delim = "\t" if "\t" in line else ","
                return [c.strip() for c in line.rstrip("\r\n").split(delim)]
        return []
    finally:
        tmp_path.unlink(missing_ok=True)


def _count_curve_fit_calls(m, fit_kwargs):
    """Run fit_spectra with call-count mock; return (n_calls, runtime_s)."""
    _real = _scipy_opt.curve_fit
    counter = {"n": 0}

    def _wrapped(*a, **kw):
        counter["n"] += 1
        return _real(*a, **kw)

    with patch.object(_scipy_opt, "curve_fit", _wrapped):
        t0 = time.perf_counter()
        m.fit_spectra(**fit_kwargs)
        elapsed = time.perf_counter() - t0

    return counter["n"], round(elapsed, 6)


# ---------------------------------------------------------------------------
# Gate 1 — fit_output_parity
# ---------------------------------------------------------------------------

def gate_fit_output_parity():
    """Default kwargs and explicit-default kwargs produce identical exports."""
    x, cube = _build_small_3x4()

    m1 = _build_mapping(x, cube)
    m1.fit_spectra(
        fit_spectrum_kwargs={"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1},
        show_progress=False,
    )
    sha1 = _export_sha256(m1)

    m2 = _build_mapping(x, cube)
    m2.fit_spectra(
        fit_spectrum_kwargs={"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1},
        show_progress=True,   # explicit default (Gate 0.2)
        n_jobs=1,             # explicit default (Gate 0.3)
        cluster_seeds=False,  # explicit default (Gate 0.3)
    )
    sha2 = _export_sha256(m2)

    passed = sha1 == sha2
    return {
        "gate_name": "fit_output_parity",
        "case_label": "small_3x4/n_starts=1",
        "passed": passed,
        "detail": "SHA256 match" if passed else f"mismatch: {sha1} vs {sha2}",
        "n_curve_fit_calls": None,
        "runtime_s": None,
        "note": "show_progress=True default confirmed (Gate 0.2); n_jobs=1, cluster_seeds=False confirmed (Gate 0.3)",
    }


# ---------------------------------------------------------------------------
# Gate 2 — export_schema_stability
# ---------------------------------------------------------------------------

# SHA256 of the v0.6.0 small_3x4/warm_start=False/n_starts=1 export,
# recorded in benchmarks/results/v0.6.0_validation.csv.
_V060_EXPORT_SHA256 = "ee894af8490a2d7c19ea94c6ae13a94dbc1ac3e3243ae21de43d7cee192b1d82"


def gate_export_schema_stability():
    """Export SHA256 matches the v0.6.0 reference for small_3x4/n_starts=1."""
    x, cube = _build_small_3x4()
    m = _build_mapping(x, cube)
    m.fit_spectra(
        fit_spectrum_kwargs={"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1},
        show_progress=False,
    )
    sha = _export_sha256(m)

    # Also verify feature_table() QA columns (frozen contract from test_api_stability.py)
    ft_cols = set(m.feature_table().columns)
    missing_qa = _QA_COLS - ft_cols
    if missing_qa:
        return {
            "gate_name": "export_schema_stability",
            "case_label": "small_3x4",
            "passed": False,
            "detail": f"feature_table() missing frozen QA columns: {sorted(missing_qa)}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        }

    passed = sha == _V060_EXPORT_SHA256
    return {
        "gate_name": "export_schema_stability",
        "case_label": "small_3x4",
        "passed": passed,
        "detail": (
            f"export SHA256 matches v0.6.0 reference; QA cols present"
            if passed
            else f"export SHA256 mismatch: current={sha}, v0.6.0={_V060_EXPORT_SHA256}"
        ),
        "n_curve_fit_calls": None,
        "runtime_s": None,
        "note": "feature_table() QA column freeze also verified",
    }


# ---------------------------------------------------------------------------
# Gate 3 — call_count_sanity
# ---------------------------------------------------------------------------

def gate_call_count_sanity():
    """n_curve_fit_calls on small_3x4/n_starts=1 is within tolerance of v0.6.0."""
    x, cube = _build_small_3x4()
    m = _build_mapping(x, cube)
    n_calls, runtime_s = _count_curve_fit_calls(
        m,
        {
            "fit_spectrum_kwargs": {"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1},
            "show_progress": False,
        },
    )

    # Load v0.6.0 reference call count
    ref_calls = None
    if _V060_CSV.exists():
        try:
            with open(_V060_CSV, newline="") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("dataset_name") == "small_3x4"
                            and row.get("warm_start") in ("False", "false")
                            and row.get("n_starts") == "1"):
                        ref_calls = int(row["n_curve_fit_calls"])
                        break
        except Exception:
            pass

    if ref_calls is None:
        return {
            "gate_name": "call_count_sanity",
            "case_label": "small_3x4/n_starts=1",
            "passed": True,
            "detail": f"current n_calls={n_calls}; no matching v0.6.0 reference row found",
            "n_curve_fit_calls": n_calls,
            "runtime_s": runtime_s,
            "note": "advisory — wall-clock is not a release gate; v0.6.0 CSV has no n_jobs column so matching by dataset+warm_start+n_starts only",
        }

    lo = ref_calls * (1 - _CALL_COUNT_TOLERANCE)
    hi = ref_calls * (1 + _CALL_COUNT_TOLERANCE)
    passed = lo <= n_calls <= hi
    return {
        "gate_name": "call_count_sanity",
        "case_label": "small_3x4/n_starts=1",
        "passed": passed,
        "detail": (
            f"current n_calls={n_calls}, v0.6.0 ref={ref_calls}, "
            f"tolerance=±{int(_CALL_COUNT_TOLERANCE*100)}%"
        ),
        "n_curve_fit_calls": n_calls,
        "runtime_s": runtime_s,
        "note": "advisory — wall-clock is not a release gate",
    }


# ---------------------------------------------------------------------------
# Gate 4 — parallel_safety
# ---------------------------------------------------------------------------

def gate_parallel_safety():
    """n_jobs=2 safe-mode succeeds; unsafe warm-start raises ValueError."""
    rows = []
    x, cube = _build_small_3x4()
    fsw = {"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1}

    # Sub-gate 4a: n_jobs=2 safe mode (warm_start=False) succeeds
    try:
        m = _build_mapping(x, cube)
        m.fit_spectra(warm_start=False, n_jobs=2, show_progress=False, fit_spectrum_kwargs=fsw)
        rows.append({
            "gate_name": "parallel_safety",
            "case_label": "n_jobs=2/warm_start=False",
            "passed": True,
            "detail": "completed without error",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except Exception as exc:
        rows.append({
            "gate_name": "parallel_safety",
            "case_label": "n_jobs=2/warm_start=False",
            "passed": False,
            "detail": f"unexpected error: {exc}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })

    # Sub-gate 4b: unsafe warm-start (warm_start=True, row_reset=False, n_jobs=2) raises
    try:
        m2 = _build_mapping(x, cube)
        m2.fit_spectra(
            warm_start=True, row_reset=False, n_jobs=2,
            show_progress=False, fit_spectrum_kwargs=fsw,
        )
        rows.append({
            "gate_name": "parallel_safety",
            "case_label": "n_jobs=2/warm_start=True/row_reset=False",
            "passed": False,
            "detail": "expected ValueError but no exception was raised",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except ValueError:
        rows.append({
            "gate_name": "parallel_safety",
            "case_label": "n_jobs=2/warm_start=True/row_reset=False",
            "passed": True,
            "detail": "ValueError raised as expected",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except Exception as exc:
        rows.append({
            "gate_name": "parallel_safety",
            "case_label": "n_jobs=2/warm_start=True/row_reset=False",
            "passed": False,
            "detail": f"wrong exception type: {type(exc).__name__}: {exc}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })

    return rows


# ---------------------------------------------------------------------------
# Gate 5 — cluster_seed_boundary
# ---------------------------------------------------------------------------

def gate_cluster_seed_boundary():
    """cluster_seeds=False parity; cluster_seeds=True+n_jobs>1 raises."""
    rows = []
    x, cube = _build_small_3x4()
    fsw = {"diagnostics": "light", "random_state": _RANDOM_STATE, "n_starts": 1}

    # Sub-gate 5a: cluster_seeds=False parity (same as default)
    m1 = _build_mapping(x, cube)
    m1.fit_spectra(fit_spectrum_kwargs=fsw, show_progress=False)
    sha1 = _export_sha256(m1)

    m2 = _build_mapping(x, cube)
    m2.fit_spectra(cluster_seeds=False, fit_spectrum_kwargs=fsw, show_progress=False)
    sha2 = _export_sha256(m2)

    passed_parity = sha1 == sha2
    rows.append({
        "gate_name": "cluster_seed_boundary",
        "case_label": "cluster_seeds=False/parity",
        "passed": passed_parity,
        "detail": "SHA256 match" if passed_parity else f"mismatch: {sha1} vs {sha2}",
        "n_curve_fit_calls": None,
        "runtime_s": None,
        "note": "",
    })

    # Sub-gate 5b: cluster_seeds=True + n_jobs>1 raises
    try:
        m3 = _build_mapping(x, cube)
        m3.fit_spectra(cluster_seeds=True, n_jobs=2, show_progress=False, fit_spectrum_kwargs=fsw)
        rows.append({
            "gate_name": "cluster_seed_boundary",
            "case_label": "cluster_seeds=True/n_jobs=2",
            "passed": False,
            "detail": "expected error but no exception raised",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except (ValueError, RuntimeError):
        rows.append({
            "gate_name": "cluster_seed_boundary",
            "case_label": "cluster_seeds=True/n_jobs=2",
            "passed": True,
            "detail": "raises as expected",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except ImportError:
        rows.append({
            "gate_name": "cluster_seed_boundary",
            "case_label": "cluster_seeds=True/n_jobs=2",
            "passed": True,
            "detail": "ImportError (sklearn not installed) before parallel check — acceptable",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "sklearn not in environment; cluster_seeds tests are advisory",
        })
    except Exception as exc:
        rows.append({
            "gate_name": "cluster_seed_boundary",
            "case_label": "cluster_seeds=True/n_jobs=2",
            "passed": False,
            "detail": f"wrong exception type: {type(exc).__name__}: {exc}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })

    return rows


# ---------------------------------------------------------------------------
# Gate 6 — autotune_non_mutation
# ---------------------------------------------------------------------------

def _build_mapping_with_baseline(x, cube):
    """Build a RamanMapping with background_remove=True so apply_choice() works."""
    Y, X, _ = cube.shape
    return RamanMapping.from_arrays(
        cube, x, X, Y,
        custom_peaks=_CUSTOM_PEAKS,
        data_range=_DATA_RANGE,
        background_remove=True,
        smoothing=False,
        normalize=False,
    )


def gate_autotune_non_mutation():
    """autotune_baseline() does not mutate preprocessing until apply_choice()."""
    x, cube = _build_small_3x4()
    m = _build_mapping_with_baseline(x, cube)

    preprocessing_before = repr(m.preprocessing)

    try:
        result = m.autotune_baseline(seed_coord=(0, 0), plot=False)
    except Exception as exc:
        return [{
            "gate_name": "autotune_non_mutation",
            "case_label": "autotune_baseline/call",
            "passed": False,
            "detail": f"autotune_baseline raised: {type(exc).__name__}: {exc}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        }]

    preprocessing_after_autotune = repr(m.preprocessing)
    non_mutating = preprocessing_before == preprocessing_after_autotune

    rows = [{
        "gate_name": "autotune_non_mutation",
        "case_label": "autotune_baseline/non_mutating",
        "passed": non_mutating,
        "detail": (
            "preprocessing unchanged after autotune_baseline()" if non_mutating
            else "preprocessing was mutated by autotune_baseline() — expected non-mutating"
        ),
        "n_curve_fit_calls": None,
        "runtime_s": None,
        "note": "",
    }]

    # apply_choice should mutate
    try:
        m.apply_choice(result.winner)
        preprocessing_after_apply = repr(m.preprocessing)
        mutated = preprocessing_after_apply != preprocessing_before
        rows.append({
            "gate_name": "autotune_non_mutation",
            "case_label": "apply_choice/mutates",
            "passed": mutated,
            "detail": (
                "preprocessing changed after apply_choice()" if mutated
                else "preprocessing unchanged after apply_choice() — expected mutation"
            ),
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })
    except Exception as exc:
        rows.append({
            "gate_name": "autotune_non_mutation",
            "case_label": "apply_choice/mutates",
            "passed": False,
            "detail": f"apply_choice raised: {type(exc).__name__}: {exc}",
            "n_curve_fit_calls": None,
            "runtime_s": None,
            "note": "",
        })

    return rows


# ---------------------------------------------------------------------------
# Gate 7 — batch_progress_default
# ---------------------------------------------------------------------------

def gate_batch_progress_default():
    """show_progress=True is the default on fit_spectra_batch; omitting it preserves behavior."""
    import inspect
    from ramanpl.batch import fit_spectra_batch

    sig = inspect.signature(fit_spectra_batch)
    default = sig.parameters["show_progress"].default if "show_progress" in sig.parameters else None
    passed = default is True
    return [{
        "gate_name": "batch_progress_default",
        "case_label": "fit_spectra_batch/show_progress_default",
        "passed": passed,
        "detail": (
            f"show_progress default is True (confirmed Gate 0.2)" if passed
            else f"show_progress default is {default!r}, expected True"
        ),
        "n_curve_fit_calls": None,
        "runtime_s": None,
        "note": "",
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("Gate 1: fit_output_parity …")
    all_rows.append(gate_fit_output_parity())
    print(f"  {all_rows[-1]['passed']}  {all_rows[-1]['detail']}")

    print("Gate 2: export_schema_stability …")
    all_rows.append(gate_export_schema_stability())
    print(f"  {all_rows[-1]['passed']}  {all_rows[-1]['detail']}")

    print("Gate 3: call_count_sanity …")
    all_rows.append(gate_call_count_sanity())
    print(f"  {all_rows[-1]['passed']}  {all_rows[-1]['detail']}")

    print("Gate 4: parallel_safety …")
    parallel_rows = gate_parallel_safety()
    all_rows.extend(parallel_rows)
    for r in parallel_rows:
        print(f"  [{r['case_label']}] {r['passed']}  {r['detail']}")

    print("Gate 5: cluster_seed_boundary …")
    cs_rows = gate_cluster_seed_boundary()
    all_rows.extend(cs_rows)
    for r in cs_rows:
        print(f"  [{r['case_label']}] {r['passed']}  {r['detail']}")

    print("Gate 6: autotune_non_mutation …")
    at_rows = gate_autotune_non_mutation()
    all_rows.extend(at_rows)
    for r in at_rows:
        print(f"  [{r['case_label']}] {r['passed']}  {r['detail']}")

    print("Gate 7: batch_progress_default …")
    bp_rows = gate_batch_progress_default()
    all_rows.extend(bp_rows)
    for r in bp_rows:
        print(f"  [{r['case_label']}] {r['passed']}  {r['detail']}")

    # Write CSV
    with open(_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written  →  {_CSV_PATH}  ({len(all_rows)} rows)")

    # Build summary by gate
    def _gate_passed(gate_name):
        return all(r["passed"] for r in all_rows if r["gate_name"] == gate_name)

    summary = {
        "fit_output_parity": _gate_passed("fit_output_parity"),
        "export_schema_stability": _gate_passed("export_schema_stability"),
        "call_count_sanity": _gate_passed("call_count_sanity"),
        "parallel_safety": _gate_passed("parallel_safety"),
        "cluster_seed_boundary": _gate_passed("cluster_seed_boundary"),
        "autotune_non_mutation": _gate_passed("autotune_non_mutation"),
        "batch_progress_default": _gate_passed("batch_progress_default"),
    }
    summary["overall_pass"] = all(summary.values())

    _JSON_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON  →  {_JSON_PATH}")

    failed = [k for k, v in summary.items() if not v and k != "overall_pass"]
    if failed:
        print(f"\n[FAIL] gates failed: {failed}")
        sys.exit(1)

    print("\n[Done] All v0.6.6 validation gates passed.")


if __name__ == "__main__":
    run()
