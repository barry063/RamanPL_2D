"""
test_mapping_backend_parity.py
-------------------------------
Parity tests: native vs RamanSPy preprocessing on synthetic Raman mapping cubes.

Entire module is skipped when RamanSPy is not installed.
"""

import numpy as np
import pytest

ramanspy = pytest.importorskip("ramanspy")  # skip entire module if absent

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.preprocessing import (
    apply_pipeline_to_mapping_cube,
    Pipeline,
    make_benchmark_pipeline_crop_only,
    make_benchmark_pipeline_savgol_only,
    make_benchmark_pipeline_poly_baseline,
    make_benchmark_pipeline_asls_baseline,
    make_benchmark_pipeline_airpls_baseline,
    make_benchmark_pipeline_arpls_baseline,
)

# ---------------------------------------------------------------------------
# Shared fixture (module-level constants)
# ---------------------------------------------------------------------------

_N = 80
_x = np.linspace(200.0, 1800.0, _N)
_Y, _X = 4, 3
_rng = np.random.default_rng(42)
_peak = np.exp(-0.5 * ((_x - 520.0) / 15.0) ** 2)
_cube = _peak[None, None, :] + _rng.uniform(0.0, 0.05, (_Y, _X, _N))


def _run(pipeline_native):
    """Run the same steps through both backends and return (native_result, ramanspy_result)."""
    p_n = Pipeline(steps=pipeline_native.steps, backend="native",   name=pipeline_native.name)
    p_r = Pipeline(steps=pipeline_native.steps, backend="ramanspy", name=pipeline_native.name)
    r_n = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_n, modality="Raman", axis_kind="raman_shift_cm-1"
    )
    r_r = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_r, modality="Raman", axis_kind="raman_shift_cm-1"
    )
    return r_n, r_r


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

def test_mapping_crop_only_native_vs_ramanspy_parity():
    """Crop is a boolean mask — output values must be bit-identical between backends."""
    r_n, r_r = _run(make_benchmark_pipeline_crop_only())

    assert r_n.x.shape == r_r.x.shape
    assert r_n.cube.shape == r_r.cube.shape
    np.testing.assert_array_equal(r_n.x, r_r.x)
    np.testing.assert_array_equal(r_n.cube, r_r.cube)


def test_mapping_savgol_native_vs_ramanspy_parity():
    """SavGol: same kernel, both backends; RMSE should be negligible (<1e-6)."""
    r_n, r_r = _run(make_benchmark_pipeline_savgol_only())

    assert r_n.cube.shape == r_r.cube.shape
    assert r_n.x.shape == r_r.x.shape
    rmse = float(np.sqrt(np.mean((r_n.cube - r_r.cube) ** 2)))
    assert rmse < 1e-6, f"SavGol parity RMSE {rmse:.2e} exceeds 1e-6"


def test_mapping_poly_baseline_native_vs_ramanspy_parity():
    """Polynomial baseline: RMSE < 1e-4 (same lstsq solve, minor fp path differences)."""
    r_n, r_r = _run(make_benchmark_pipeline_poly_baseline())

    assert r_n.cube.shape == r_r.cube.shape
    assert r_n.x.shape == r_r.x.shape
    rmse = float(np.sqrt(np.mean((r_n.cube - r_r.cube) ** 2)))
    assert rmse < 1e-4, f"Poly baseline parity RMSE {rmse:.2e} exceeds 1e-4"


def test_mapping_asls_baseline_native_vs_ramanspy_parity():
    """AsLS baseline: RMSE < 1e-3 (solver convergence may differ between implementations)."""
    r_n, r_r = _run(make_benchmark_pipeline_asls_baseline())

    assert r_n.cube.shape == r_r.cube.shape
    assert r_n.x.shape == r_r.x.shape
    rmse = float(np.sqrt(np.mean((r_n.cube - r_r.cube) ** 2)))
    assert rmse < 1e-3, f"AsLS parity RMSE {rmse:.2e} exceeds 1e-3"


def test_mapping_airpls_baseline_native_vs_ramanspy_parity():
    """airPLS baseline: RMSE < 5e-3.

    Tolerance is wider than other iterative solvers because airPLS shows
    greater convergence divergence between the native (pybaselines) and
    RamanSPy implementations at niter=10, reaching ~3.9e-3 on the synthetic
    fixture. This is an expected numeric difference, not a structural bug.
    """
    r_n, r_r = _run(make_benchmark_pipeline_airpls_baseline())

    assert r_n.cube.shape == r_r.cube.shape
    assert r_n.x.shape == r_r.x.shape
    rmse = float(np.sqrt(np.mean((r_n.cube - r_r.cube) ** 2)))
    assert rmse < 5e-3, f"airPLS parity RMSE {rmse:.2e} exceeds 5e-3"


def test_mapping_arpls_baseline_native_vs_ramanspy_parity():
    """arPLS baseline: RMSE < 1e-3."""
    r_n, r_r = _run(make_benchmark_pipeline_arpls_baseline())

    assert r_n.cube.shape == r_r.cube.shape
    assert r_n.x.shape == r_r.x.shape
    rmse = float(np.sqrt(np.mean((r_n.cube - r_r.cube) ** 2)))
    assert rmse < 1e-3, f"arPLS parity RMSE {rmse:.2e} exceeds 1e-3"
