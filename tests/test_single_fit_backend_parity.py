"""
test_single_fit_backend_parity.py

Parity tests: native vs RamanSPy preprocessing on a synthetic Raman single spectrum.

Entire module is skipped when RamanSPy is not installed.
"""

import numpy as np
import pytest

ramanspy = pytest.importorskip("ramanspy")  # skip entire module if absent

from ramanpl.preprocessing import (
    SpectralDataset,
    Pipeline,
    CropByRange,
    SmoothSavGol,
    BaselineSubtract,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_N = 200
_x = np.linspace(100.0, 2000.0, _N)
_rng = np.random.default_rng(42)
_peak = 1e4 / ((_x - 520.0) ** 2 + 10.0 ** 2)
_y = np.clip(_peak + _rng.normal(0, 0.01, _N), 0, None)

_ds = SpectralDataset(
    x=_x.copy(),
    y=_y.copy(),
    modality="Raman",
    axis_kind="raman_shift_cm-1",
    meta={"x_trimmed_on_load": False},
)


def _run_single(steps):
    """Apply the same steps via native and ramanspy backends; return (y_native, y_ramanspy)."""
    p_n = Pipeline(steps=steps, backend="native")
    p_r = Pipeline(steps=steps, backend="ramanspy")
    out_n = p_n.apply(_ds)
    out_r = p_r.apply(_ds)
    return out_n.y, out_r.y


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

def test_single_fit_crop_only_native_vs_ramanspy_parity():
    """Crop is a boolean mask — output values must be bit-identical between backends."""
    y_n, y_r = _run_single([CropByRange((200, 1800))])
    assert y_n.shape == y_r.shape
    np.testing.assert_array_equal(y_n, y_r)


def test_single_fit_savgol_native_vs_ramanspy_parity():
    """SavGol: same kernel coefficients; RMSE should be negligible (<1e-6)."""
    y_n, y_r = _run_single([SmoothSavGol(window_length=11, polyorder=3)])
    assert y_n.shape == y_r.shape
    rmse = float(np.sqrt(np.mean((y_n - y_r) ** 2)))
    assert rmse < 1e-6, f"SavGol parity RMSE {rmse:.2e} exceeds 1e-6"


def test_single_fit_poly_baseline_native_vs_ramanspy_parity():
    """Polynomial baseline: RMSE < 1e-4 (same lstsq solve, minor fp path differences)."""
    steps = [
        CropByRange((200, 1800)),
        BaselineSubtract({"method": "poly", "poly_order": 3}),
    ]
    y_n, y_r = _run_single(steps)
    assert y_n.shape == y_r.shape
    rmse = float(np.sqrt(np.mean((y_n - y_r) ** 2)))
    assert rmse < 1e-4, f"Poly baseline parity RMSE {rmse:.2e} exceeds 1e-4"


def test_single_fit_asls_baseline_native_vs_ramanspy_parity():
    """AsLS baseline: RMSE < 5e-3 (iterative Whittaker, minor convergence path differences)."""
    steps = [
        CropByRange((200, 1800)),
        BaselineSubtract({"method": "asls", "lam": 1e5, "p": 0.01}),
    ]
    y_n, y_r = _run_single(steps)
    assert y_n.shape == y_r.shape
    rmse = float(np.sqrt(np.mean((y_n - y_r) ** 2)))
    assert rmse < 5e-3, f"AsLS parity RMSE {rmse:.2e} exceeds 5e-3"


def test_single_fit_airpls_baseline_native_vs_ramanspy_parity():
    """airPLS baseline: RMSE < 0.05 (widest tolerance — adaptive weights diverge slightly)."""
    steps = [
        CropByRange((200, 1800)),
        BaselineSubtract({"method": "airpls", "lam": 1e6}),
    ]
    y_n, y_r = _run_single(steps)
    assert y_n.shape == y_r.shape
    rmse = float(np.sqrt(np.mean((y_n - y_r) ** 2)))
    assert rmse < 0.05, f"airPLS parity RMSE {rmse:.2e} exceeds 0.05"


def test_single_fit_arpls_baseline_native_vs_ramanspy_parity():
    """arPLS baseline: RMSE < 5e-3."""
    steps = [
        CropByRange((200, 1800)),
        BaselineSubtract({"method": "arpls", "lam": 1e5}),
    ]
    y_n, y_r = _run_single(steps)
    assert y_n.shape == y_r.shape
    rmse = float(np.sqrt(np.mean((y_n - y_r) ** 2)))
    assert rmse < 5e-3, f"arPLS parity RMSE {rmse:.2e} exceeds 5e-3"
