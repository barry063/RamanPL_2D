"""
test_mapping_memory_runtime_smoke.py
--------------------------------------
Smoke tests for runtime and memory measurement helpers in
benchmarks/benchmark_mapping_preprocessing.py.

Guards against obvious regressions: metrics must be recorded and non-negative;
the RamanSPy path must not blow up on a tiny fixture.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_mapping_preprocessing import (
    measure_runtime_seconds,
    measure_peak_memory_mb,
    run_mapping_preprocess_case,
    build_supported_mapping_pipelines,
)
from ramanpl.integration.ramanspy_adapter import has_ramanspy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mapping_runtime_metric_is_recorded():
    """measure_runtime_seconds must return a positive float for any callable."""
    def _work():
        arr = np.zeros((200, 200), dtype=float)
        for _ in range(50):
            arr += 1.0

    t = measure_runtime_seconds(_work)
    assert isinstance(t, float)
    assert t >= 0.0


def test_mapping_peak_memory_metric_is_recorded():
    """measure_peak_memory_mb must return a positive float for an allocation."""
    def _allocate():
        return np.ones((500, 500), dtype=float)  # ~2 MB

    mem = measure_peak_memory_mb(_allocate)
    assert isinstance(mem, float)
    assert mem >= 0.0


def test_mapping_ramanspy_backend_does_not_fail_small_fixture_benchmark():
    """RamanSPy backend must run cleanly on a tiny 2×2 crop-only benchmark case."""
    if not has_ramanspy():
        pytest.skip("ramanspy not installed")

    x = np.linspace(200.0, 1800.0, 40)
    cube = np.ones((2, 2, 40), dtype=float)
    pipe = build_supported_mapping_pipelines()[0]  # crop_only

    result = run_mapping_preprocess_case(
        x=x, cube=cube, pipeline=pipe,
        backend="ramanspy", dataset_name="tiny_rspy_smoke",
    )
    assert result["runtime_s"] >= 0.0
    assert result["peak_memory_mb"] >= 0.0
    assert result["backend_resolved"] == "ramanspy"
