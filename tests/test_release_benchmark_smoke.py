"""
test_release_benchmark_smoke.py
--------------------------------
Smoke-validate the benchmark harnesses for release readiness.

Calls the public benchmark entry points with tiny synthetic inputs and
asserts that the harness runs, produces output, and returns records with
the expected structural fields. No runtime thresholds are enforced.
"""

import sys
from pathlib import Path

import pytest

# Make benchmark scripts importable when tests are run from the repo root.
_BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS_DIR))


# ---------------------------------------------------------------------------
# Baseline kernel benchmark smoke
# ---------------------------------------------------------------------------

_BASELINE_REQUIRED_FIELDS = {"spectrum_len", "method", "runtime_s", "peak_mb"}


def test_baseline_kernel_benchmark_runs_and_returns_records():
    from benchmark_baseline_kernels import run_baseline_kernel_benchmarks
    rows = run_baseline_kernel_benchmarks()
    assert isinstance(rows, list), "run_baseline_kernel_benchmarks() must return a list"
    assert len(rows) > 0, "No benchmark rows returned"


def test_baseline_kernel_benchmark_record_structure():
    from benchmark_baseline_kernels import run_baseline_kernel_benchmarks
    rows = run_baseline_kernel_benchmarks()
    for row in rows:
        missing = _BASELINE_REQUIRED_FIELDS - row.keys()
        assert not missing, f"Benchmark row missing fields: {missing}  row={row}"


def test_baseline_kernel_benchmark_values_are_finite():
    from benchmark_baseline_kernels import run_baseline_kernel_benchmarks
    import math
    rows = run_baseline_kernel_benchmarks()
    for row in rows:
        assert row["runtime_s"] >= 0, f"Negative runtime: {row}"
        assert math.isfinite(row["runtime_s"]), f"Non-finite runtime: {row}"
        assert math.isfinite(row["peak_mb"]), f"Non-finite peak_mb: {row}"


# ---------------------------------------------------------------------------
# Mapping preprocessing benchmark smoke
# ---------------------------------------------------------------------------

_MAPPING_REQUIRED_FIELDS = {
    "dataset_name", "pipeline_name", "backend_requested", "backend_resolved",
    "runtime_s", "peak_memory_mb", "cube_shape", "axis_len", "axis_match",
}


def test_mapping_benchmark_builds_cases():
    from benchmark_mapping_preprocessing import build_mapping_benchmark_cases
    cases = build_mapping_benchmark_cases()
    assert isinstance(cases, list)
    assert len(cases) > 0
    name, x, cube = cases[0]
    assert isinstance(name, str)
    assert cube.ndim == 3


def test_mapping_benchmark_builds_pipelines():
    from benchmark_mapping_preprocessing import build_supported_mapping_pipelines
    pipelines = build_supported_mapping_pipelines()
    assert isinstance(pipelines, list)
    assert len(pipelines) > 0


def test_mapping_benchmark_smoke_run_native():
    from benchmark_mapping_preprocessing import (
        build_mapping_benchmark_cases,
        build_supported_mapping_pipelines,
        run_mapping_preprocess_case,
    )
    # Run only the smallest case with the first pipeline to keep CI fast.
    cases = build_mapping_benchmark_cases()
    pipelines = build_supported_mapping_pipelines()
    name, x, cube = cases[0]   # "small_3x4"
    pipeline = pipelines[0]    # crop-only

    row = run_mapping_preprocess_case(
        x=x, cube=cube, pipeline=pipeline, backend="native", dataset_name=name
    )

    missing = _MAPPING_REQUIRED_FIELDS - row.keys()
    assert not missing, f"Mapping benchmark row missing fields: {missing}"
    assert row["backend_resolved"] in ("native", "ramanspy"), (
        f"Unexpected backend_resolved: {row['backend_resolved']}"
    )
    assert row["axis_match"] is True, "Axis length mismatch after preprocessing"


# ---------------------------------------------------------------------------
# Mapping fit benchmark smoke
# ---------------------------------------------------------------------------

_FIT_REQUIRED_FIELDS = {
    "dataset_name", "cube_shape", "warm_start", "n_starts", "n_jobs", "random_state",
    "runtime_s", "n_curve_fit_calls", "success_rate", "mean_rmse_finite",
    "n_failed_pixels",
}


def test_mapping_fit_benchmark_builds_cases():
    from benchmark_mapping_fit import build_mapping_fit_benchmark_cases
    cases = build_mapping_fit_benchmark_cases()
    assert isinstance(cases, list)
    assert len(cases) > 0
    name, x, cube = cases[0]
    assert isinstance(name, str)
    assert cube.ndim == 3
    assert x.shape[0] == cube.shape[2]


def test_mapping_fit_benchmark_smoke_run():
    from benchmark_mapping_fit import (
        build_mapping_fit_benchmark_cases,
        build_fit_kwargs_variants,
        run_mapping_fit_case,
    )
    cases = build_mapping_fit_benchmark_cases()
    variants = build_fit_kwargs_variants()
    name, x, cube = cases[0]   # "small_3x4"
    fit_kwargs = variants[0]   # warm_start=False, n_starts=1

    row = run_mapping_fit_case(x=x, cube=cube, fit_kwargs=fit_kwargs, dataset_name=name)

    missing = _FIT_REQUIRED_FIELDS - row.keys()
    assert not missing, f"Fit benchmark row missing fields: {missing}"


def test_mapping_fit_benchmark_record_structure():
    from benchmark_mapping_fit import (
        build_mapping_fit_benchmark_cases,
        build_fit_kwargs_variants,
        run_mapping_fit_case,
    )
    cases = build_mapping_fit_benchmark_cases()
    variants = build_fit_kwargs_variants()
    name, x, cube = cases[0]
    fit_kwargs = variants[0]

    row = run_mapping_fit_case(x=x, cube=cube, fit_kwargs=fit_kwargs, dataset_name=name)

    assert isinstance(row["n_curve_fit_calls"], int), (
        f"n_curve_fit_calls must be int, got {type(row['n_curve_fit_calls'])}"
    )
    assert row["n_curve_fit_calls"] >= 0
    assert 0.0 <= row["success_rate"] <= 1.0
    assert row["runtime_s"] >= 0.0
