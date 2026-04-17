"""
test_mapping_cube_consistency.py
---------------------------------
Validate cube shape invariants, axis ordering, and adapter round-trip correctness.

Native-only tests run unconditionally.  Tests that require RamanSPy call
pytest.importorskip inside the test function.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.preprocessing import (
    apply_pipeline_to_mapping_cube,
    Pipeline,
    SmoothSavGol,
    make_benchmark_pipeline_crop_only,
    make_benchmark_pipeline_asls_baseline,
)
from ramanpl.integration.ramanspy_adapter import to_ramanspy_image, from_ramanspy_image

# ---------------------------------------------------------------------------
# Shared module-level fixture
# ---------------------------------------------------------------------------

_N = 60
_x = np.linspace(200.0, 1800.0, _N)
_Y, _X = 5, 4
_rng = np.random.default_rng(7)
_peak = np.exp(-0.5 * ((_x - 520.0) / 15.0) ** 2)
_cube = _peak[None, None, :] + _rng.uniform(0.0, 0.05, (_Y, _X, _N))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mapping_cube_shape_preserved_after_native_preprocessing():
    """Output cube must be [Y, X, N_cropped] with N_cropped < N_input."""
    pipeline = make_benchmark_pipeline_crop_only()
    result = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=pipeline,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    assert result.cube.ndim == 3
    assert result.cube.shape[0] == _Y
    assert result.cube.shape[1] == _X
    assert result.cube.shape[2] == result.x.size
    assert result.cube.shape[2] < _N, "Crop step should reduce spectral axis length"


def test_mapping_cube_shape_preserved_after_ramanspy_preprocessing():
    """RamanSPy backend must return the same spatial dimensions as the native backend."""
    pytest.importorskip("ramanspy")
    p = Pipeline(
        steps=make_benchmark_pipeline_crop_only().steps,
        backend="ramanspy",
        name="crop_rspy",
    )
    result = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    assert result.cube.ndim == 3
    assert result.cube.shape[0] == _Y
    assert result.cube.shape[1] == _X
    assert result.cube.shape[2] == result.x.size
    assert result.cube.shape[2] < _N


def test_mapping_axis_order_consistent_between_backends():
    """Cropped axis values must be identical between native and RamanSPy backends."""
    pytest.importorskip("ramanspy")
    p_n = Pipeline(
        steps=make_benchmark_pipeline_crop_only().steps,
        backend="native", name="crop_n",
    )
    p_r = Pipeline(
        steps=make_benchmark_pipeline_crop_only().steps,
        backend="ramanspy", name="crop_r",
    )
    r_n = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_n,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    r_r = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_r,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    np.testing.assert_array_equal(
        r_n.x, r_r.x,
        err_msg="Cropped axis values differ between native and RamanSPy backends",
    )


def test_mapping_selected_pixel_identity_consistent_between_backends():
    """Pixel (2, 1) spectrum must be the same across backends after SavGol smoothing."""
    pytest.importorskip("ramanspy")
    steps = [SmoothSavGol(window_length=11, polyorder=3)]
    p_n = Pipeline(steps=steps, backend="native",   name="sg_n")
    p_r = Pipeline(steps=steps, backend="ramanspy", name="sg_r")

    r_n = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_n,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    r_r = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=p_r,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    np.testing.assert_allclose(
        r_n.cube[2, 1, :], r_r.cube[2, 1, :],
        rtol=1e-9, atol=1e-9,
        err_msg="Pixel (2, 1) spectrum differs between backends after SavGol",
    )


def test_mapping_asls_cube_shape_preserved_after_v046_optimisation():
    """v0.4.6: asLS baseline preprocessing via DtD-cached path must preserve cube shape."""
    pipeline = make_benchmark_pipeline_asls_baseline()
    result = apply_pipeline_to_mapping_cube(
        x=_x, cube=_cube, pipeline=pipeline,
        modality="Raman", axis_kind="raman_shift_cm-1",
    )
    assert result.cube.ndim == 3
    assert result.cube.shape[0] == _Y
    assert result.cube.shape[1] == _X
    assert result.cube.shape[2] == result.x.size
    assert np.all(np.isfinite(result.cube)), "asLS cube contains non-finite values"


def test_mapping_coordinates_remain_consistent_after_preprocessing():
    """Adapter round-trip [Y,X,N] → [X,Y,N] → [Y,X,N] must be lossless."""
    pytest.importorskip("ramanspy")
    x_rt, cube_rt = from_ramanspy_image(
        to_ramanspy_image(
            x=_x, cube_yxn=_cube, modality="Raman", axis_kind="raman_shift_cm-1"
        )
    )
    assert cube_rt.shape == _cube.shape, (
        f"Round-trip shape {cube_rt.shape} != original {_cube.shape}"
    )
    np.testing.assert_array_equal(x_rt, _x, err_msg="Spectral axis changed during adapter round-trip")
    np.testing.assert_array_equal(
        cube_rt, _cube,
        err_msg="Cube values changed during [Y,X,N] → [X,Y,N] → [Y,X,N] round-trip",
    )
