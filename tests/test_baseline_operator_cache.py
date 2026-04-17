"""
test_baseline_operator_cache.py
---------------------------------
Tests for the _DtD_cache / _get_cached_DtD internal mechanism introduced in v0.4.6.

These are internal-behaviour tests.  They lock down cache identity, keying
correctness, and numerical agreement with direct computation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.baselineAPI import _get_cached_DtD, _whittaker_diff_matrix, _DtD_cache


# ---------------------------------------------------------------------------
# Cache identity and keying
# ---------------------------------------------------------------------------

def test_cache_same_key_returns_same_object():
    """Two calls with identical (n, diff_order) must return the exact same object."""
    _ = _get_cached_DtD(80, 2)  # ensure populated
    a = _get_cached_DtD(80, 2)
    b = _get_cached_DtD(80, 2)
    assert a is b, "Cache should return the same object for the same key"


def test_cache_different_n_different_object():
    a = _get_cached_DtD(80, 2)
    b = _get_cached_DtD(100, 2)
    assert a is not b
    assert a.shape != b.shape


def test_cache_different_diff_order_different_object():
    a = _get_cached_DtD(80, 2)
    b = _get_cached_DtD(80, 1)
    assert a is not b
    # Both are (n, n) — shapes match, but stored elements differ (higher order = more bandwidth)
    assert a.nnz != b.nnz, "DtD for diff_order=2 and diff_order=1 should have different nnz"


def test_cache_key_stored_in_dict():
    """Cache dict must contain the (n, diff_order) key after a call."""
    _get_cached_DtD(77, 2)
    assert (77, 2) in _DtD_cache


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

def test_cache_result_matches_direct_computation_order2():
    n, diff_order = 120, 2
    D = _whittaker_diff_matrix(n, diff_order)
    expected = (D.T @ D).tocsc()
    result = _get_cached_DtD(n, diff_order)
    diff = (result - expected)
    assert abs(diff).max() < 1e-14, "Cached DtD does not match direct computation (diff_order=2)"


def test_cache_result_matches_direct_computation_order1():
    n, diff_order = 60, 1
    D = _whittaker_diff_matrix(n, diff_order)
    expected = (D.T @ D).tocsc()
    result = _get_cached_DtD(n, diff_order)
    diff = (result - expected)
    assert abs(diff).max() < 1e-14, "Cached DtD does not match direct computation (diff_order=1)"


def test_cache_result_matches_direct_computation_order3():
    n, diff_order = 50, 3
    D = _whittaker_diff_matrix(n, diff_order)
    expected = (D.T @ D).tocsc()
    result = _get_cached_DtD(n, diff_order)
    diff = (result - expected)
    assert abs(diff).max() < 1e-14, "Cached DtD does not match direct computation (diff_order=3)"


def test_cached_DtD_is_symmetric():
    """D^T D is always symmetric positive semi-definite."""
    DtD = _get_cached_DtD(100, 2)
    diff = DtD - DtD.T
    assert abs(diff).max() < 1e-14, "Cached DtD is not symmetric"


# ---------------------------------------------------------------------------
# _whittaker_diff_matrix
# ---------------------------------------------------------------------------

def test_diff_matrix_order2_shape():
    D = _whittaker_diff_matrix(100, 2)
    assert D.shape == (98, 100)


def test_diff_matrix_order1_shape():
    D = _whittaker_diff_matrix(100, 1)
    assert D.shape == (99, 100)


def test_diff_matrix_order2_fast_path_matches_general():
    """The diff_order==2 fast path must produce the same matrix as the general loop."""
    n = 80
    D_fast = _whittaker_diff_matrix(n, 2)
    # General path: compose two first-order difference operators
    D1 = sparse.diags([-1, 1], [0, 1], shape=(n - 1, n), format="csc")
    D_gen = (sparse.diags([-1, 1], [0, 1], shape=(n - 2, n - 1), format="csc") @ D1).tocsc()
    diff = (D_fast - D_gen)
    assert abs(diff).max() < 1e-14
