"""
test_mapping_backend_benchmark_smoke.py
----------------------------------------
Smoke tests for the benchmark harness functions in
benchmarks/benchmark_mapping_preprocessing.py.

Verifies that benchmark functions run without error on tiny fixtures,
that result records contain the required fields, and that backend
resolution behaves correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the benchmark module importable from tests/.
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_mapping_preprocessing import (
    build_supported_mapping_pipelines,
    run_mapping_preprocess_case,
    pipeline_to_benchmark_name,
    _RESULT_FIELDS,
)
from ramanpl.integration.ramanspy_adapter import has_ramanspy
from ramanpl.preprocessing import make_benchmark_pipeline_asls_baseline

# ---------------------------------------------------------------------------
# Tiny 2×2 fixture
# ---------------------------------------------------------------------------

_N_TINY = 40
_x_tiny = np.linspace(200.0, 1800.0, _N_TINY)
_cube_tiny = np.ones((2, 2, _N_TINY), dtype=float)
_pipe = build_supported_mapping_pipelines()[0]  # crop_only — always translatable


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mapping_benchmark_case_runs_native():
    """Native backend run must succeed and return a non-negative runtime."""
    result = run_mapping_preprocess_case(
        x=_x_tiny, cube=_cube_tiny, pipeline=_pipe,
        backend="native", dataset_name="smoke_2x2",
    )
    assert isinstance(result, dict)
    assert result["runtime_s"] >= 0.0
    assert result["backend_requested"] == "native"
    assert result["backend_resolved"] == "native"


def test_mapping_benchmark_case_runs_ramanspy_when_available():
    """RamanSPy backend run must succeed and report backend_resolved='ramanspy'."""
    if not has_ramanspy():
        pytest.skip("ramanspy not installed")

    result = run_mapping_preprocess_case(
        x=_x_tiny, cube=_cube_tiny, pipeline=_pipe,
        backend="ramanspy", dataset_name="smoke_2x2",
    )
    assert result["backend_resolved"] == "ramanspy"
    assert result["runtime_s"] >= 0.0


def test_mapping_benchmark_result_contains_required_fields():
    """Result dict must contain all 11 required field names."""
    result = run_mapping_preprocess_case(
        x=_x_tiny, cube=_cube_tiny, pipeline=_pipe,
        backend="native", dataset_name="smoke_2x2",
    )
    required = set(_RESULT_FIELDS)
    assert required.issubset(set(result.keys())), (
        f"Missing fields: {required - set(result.keys())}"
    )


def test_mapping_benchmark_auto_backend_records_resolved_backend():
    """'auto' backend must resolve to either 'native' or 'ramanspy'."""
    result = run_mapping_preprocess_case(
        x=_x_tiny, cube=_cube_tiny, pipeline=_pipe,
        backend="auto", dataset_name="smoke_2x2",
    )
    assert result["backend_resolved"] in ("native", "ramanspy"), (
        f"Unexpected resolved backend: {result['backend_resolved']!r}"
    )


def test_mapping_benchmark_native_iterative_baseline_smoke():
    """Native iterative baseline (asLS) benchmark case must run and record required fields."""
    pipe = make_benchmark_pipeline_asls_baseline()
    # Use a longer axis so crop step inside the pipeline has enough range to keep ≥3 points
    x = np.linspace(200.0, 1800.0, 150)
    cube = np.ones((2, 2, 150), dtype=float) + 0.1 * np.arange(150)
    result = run_mapping_preprocess_case(
        x=x, cube=cube, pipeline=pipe,
        backend="native", dataset_name="smoke_native_asls",
    )
    assert isinstance(result, dict)
    required = set(_RESULT_FIELDS)
    assert required.issubset(set(result.keys())), (
        f"Missing fields: {required - set(result.keys())}"
    )
    assert result["backend_resolved"] == "native"
    assert result["runtime_s"] >= 0.0
