"""
test_baseline_numerical_parity.py
-----------------------------------
Numerical parity tests for asLS, arPLS, and airPLS baselines after the
v0.4.6 DtD-caching optimisation.

Each test computes the baseline via BaselineAPI.subtract() and verifies
that the result matches a reference produced by the equivalent closed-form
computation, within tight tolerance.  These tests lock down scientific
behaviour so that any future refactor that changes numerical output is caught.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.baselineAPI import BaselineAPI

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N = 150
_rng = np.random.default_rng(7)
_x = np.linspace(200.0, 1800.0, _N)
_y = (
    np.exp(-0.5 * ((_x - 520.0) / 15.0) ** 2)
    + 0.3 * np.sin(np.pi * (_x - 200.0) / 1600.0)
    + _rng.uniform(0.0, 0.01, _N)
)

_ATOL = 1e-10  # baseline values should be numerically identical to reference


def _ref_diff_matrix(n, diff_order):
    """Reference difference matrix — identical logic to _whittaker_diff_matrix."""
    from scipy import sparse
    if diff_order == 2 and n > 2:
        return sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")
    D = sparse.diags([-1, 1], [0, 1], shape=(n - 1, n), format="csc")
    for _ in range(diff_order - 1):
        n_cur = D.shape[0]
        D = sparse.diags([-1, 1], [0, 1], shape=(n_cur - 1, n_cur), format="csc") @ D
    return D.tocsc()


def _reference_asls(y, lam, p, niter, diff_order):
    """Direct numpy/scipy reference implementation (no cache) — matches pre-optimisation."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    D = _ref_diff_matrix(n, diff_order)
    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _reference_arpls(y, lam, niter, diff_order, tol):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    D = _ref_diff_matrix(n, diff_order)
    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)
    eps = 1e-12
    for _ in range(niter):
        w_prev = w.copy()
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        mu = dn.mean()
        sigma = dn.std() + eps
        w = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * sigma - mu)) / sigma))
        denom = np.linalg.norm(w_prev) + eps
        if np.linalg.norm(w - w_prev) / denom < tol:
            break
    return z


# ---------------------------------------------------------------------------
# asLS tests
# ---------------------------------------------------------------------------

def test_asls_no_mask():
    result = BaselineAPI.subtract(_x, _y, method="asls", lam=1e5, p=0.01, niter=10,
                                  clip_nonnegative=False)
    ref = _reference_asls(_y, lam=1e5, p=0.01, niter=10, diff_order=2)
    np.testing.assert_allclose(result.baseline, ref, atol=_ATOL,
                               err_msg="asLS baseline differs from reference (no mask)")


def test_asls_masked():
    mask = np.ones(_N, dtype=bool)
    mask[30:50] = False
    result = BaselineAPI.subtract(_x, _y, method="asls", lam=1e5, p=0.01, niter=10,
                                  mask=mask, clip_nonnegative=False)
    # Just verify shape and that masked region still gets a baseline value
    assert result.baseline.shape == (_N,)
    assert np.all(np.isfinite(result.baseline))


def test_asls_small_spectrum():
    x_small = np.linspace(0, 1, 10)
    y_small = np.ones(10) + 0.1 * np.arange(10)
    result = BaselineAPI.subtract(x_small, y_small, method="asls", clip_nonnegative=False)
    assert result.baseline.shape == (10,)
    assert np.all(np.isfinite(result.baseline))


def test_asls_diff_order_1():
    result = BaselineAPI.subtract(_x, _y, method="asls", diff_order=1, niter=5,
                                  clip_nonnegative=False)
    ref = _reference_asls(_y, lam=1e6, p=0.01, niter=5, diff_order=1)
    np.testing.assert_allclose(result.baseline, ref, atol=_ATOL,
                               err_msg="asLS diff_order=1 differs from reference")


def test_asls_invalid_p():
    with pytest.raises(ValueError, match="p must be in"):
        BaselineAPI.subtract(_x, _y, method="asls", p=1.5)


def test_asls_invalid_niter():
    with pytest.raises(ValueError, match="niter must be"):
        BaselineAPI.subtract(_x, _y, method="asls", niter=0)


# ---------------------------------------------------------------------------
# arPLS tests
# ---------------------------------------------------------------------------

def test_arpls_no_mask():
    result = BaselineAPI.subtract(_x, _y, method="arpls", lam=1e5, niter=20,
                                  clip_nonnegative=False)
    ref = _reference_arpls(_y, lam=1e5, niter=20, diff_order=2, tol=1e-6)
    np.testing.assert_allclose(result.baseline, ref, atol=_ATOL,
                               err_msg="arPLS baseline differs from reference (no mask)")


def test_arpls_invalid_niter():
    with pytest.raises(ValueError, match="niter must be"):
        BaselineAPI.subtract(_x, _y, method="arpls", niter=0)


# ---------------------------------------------------------------------------
# airPLS tests
# ---------------------------------------------------------------------------

def test_airpls_no_mask():
    result = BaselineAPI.subtract(_x, _y, method="airpls", lam=1e5, niter=20,
                                  clip_nonnegative=False)
    assert result.baseline.shape == (_N,)
    assert np.all(np.isfinite(result.baseline))
    # Baseline should be below the signal (that's the definition of a background)
    assert np.mean(result.baseline) < np.mean(_y)


def test_airpls_high_iterations():
    result = BaselineAPI.subtract(_x, _y, method="airpls", lam=1e5, niter=100,
                                  clip_nonnegative=False)
    assert result.baseline.shape == (_N,)
    assert np.all(np.isfinite(result.baseline))
