"""
test_cluster_seed_helpers.py
----------------------------
Unit tests for ramanpl.mapping._cluster_seeds internal helpers.

All tests use synthetic arrays only; no .wdf files are loaded.
sklearn-requiring tests are skipped automatically when sklearn is absent.
"""

import sys
import importlib
from math import sqrt
from unittest.mock import patch

import numpy as np
import pytest

from ramanpl.mapping._cluster_seeds import (
    _normalise_cluster_seed_config,
    _require_sklearn_for_cluster_seeds,
    _spectral_feature_matrix,
    _build_cluster_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_domain_cube(Y=6, X=8, N=20, seed=0):
    """Return a [Y, X, N] cube where the left half has a different spectral
    shape from the right half — two clearly separable domains."""
    rng = np.random.default_rng(seed)
    cube = np.zeros((Y, X, N), dtype=np.float64)
    peak_left = np.exp(-0.5 * ((np.arange(N) - 5) / 1.5) ** 2)
    peak_right = np.exp(-0.5 * ((np.arange(N) - 15) / 1.5) ** 2)
    for j in range(Y):
        for i in range(X):
            base = peak_left if i < X // 2 else peak_right
            cube[j, i] = base + rng.normal(scale=0.02, size=N)
    return cube


def _uniform_cube(Y=4, X=5, N=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((Y, X, N))


# ---------------------------------------------------------------------------
# Step 1: config normalisation tests
# ---------------------------------------------------------------------------

def test_cluster_seed_config_false_disables():
    """False → returns None (disabled), does not trigger sklearn import."""
    sklearn_keys_before = {k for k in sys.modules if "sklearn" in k}
    result = _normalise_cluster_seed_config(False, X=5, Y=5)
    sklearn_keys_after = {k for k in sys.modules if "sklearn" in k}
    assert result is None
    assert sklearn_keys_after == sklearn_keys_before


def test_cluster_seed_config_true_defaults_are_bounded():
    """True → valid config for 1×1 and 20×20 maps."""
    cfg_small = _normalise_cluster_seed_config(True, X=1, Y=1)
    assert cfg_small is not None
    assert cfg_small["n_clusters"] >= 1
    assert cfg_small["n_components"] >= 1

    cfg_large = _normalise_cluster_seed_config(True, X=20, Y=20)
    assert cfg_large is not None
    assert cfg_large["n_clusters"] <= 8
    assert cfg_large["n_clusters"] >= 1
    assert cfg_large["n_components"] >= 1


def test_cluster_seed_config_dict_rejects_unknown_keys():
    """Dict with unknown keys raises ValueError or TypeError naming the offending key."""
    with pytest.raises((ValueError, TypeError), match="bad_key"):
        _normalise_cluster_seed_config(
            {"n_clusters": 4, "bad_key": 99}, X=5, Y=5
        )


def test_cluster_seed_config_dict_rejects_wrong_types():
    """Non-integer n_clusters raises a clear type error."""
    with pytest.raises((ValueError, TypeError)):
        _normalise_cluster_seed_config({"n_clusters": "fish"}, X=5, Y=5)


# ---------------------------------------------------------------------------
# Step 1: spectral feature matrix tests
# ---------------------------------------------------------------------------

def test_spectral_feature_matrix_preserves_pixel_order():
    """Row-major (y, x) flattening: pixel (j, i) maps to row j*X + i."""
    Y, X, N = 3, 4, 10
    cube = np.zeros((Y, X, N))
    for j in range(Y):
        for i in range(X):
            cube[j, i] = j * 100 + i  # unique per pixel
    matrix, valid_mask = _spectral_feature_matrix(cube)
    # All pixels are valid (finite)
    assert valid_mask.shape == (Y, X)
    assert matrix.shape[0] == Y * X
    # Recover (j, i) from row index
    for idx in range(Y * X):
        j, i = divmod(idx, X)
        if valid_mask[j, i]:
            expected_val = j * 100 + i
            assert np.allclose(matrix[idx], expected_val), (
                f"Pixel ({j},{i}) at row {idx}: expected {expected_val}, got {matrix[idx]}"
            )


def test_spectral_feature_matrix_handles_degenerate_shapes():
    """Cubes of shape (1,1,N), (1,N,K), and (N,1,K) must not crash."""
    rng = np.random.default_rng(7)
    for shape in [(1, 1, 10), (1, 4, 10), (3, 1, 10)]:
        cube = rng.random(shape)
        matrix, valid_mask = _spectral_feature_matrix(cube)
        assert valid_mask.shape == (shape[0], shape[1])
        assert matrix.ndim == 2
        assert matrix.shape[1] == shape[2]


def test_spectral_feature_matrix_all_invalid_pixels():
    """All-NaN cube → empty-but-defined matrix; no True values in valid_mask."""
    Y, X, N = 3, 4, 10
    cube = np.full((Y, X, N), np.nan)
    matrix, valid_mask = _spectral_feature_matrix(cube)
    assert valid_mask.shape == (Y, X)
    assert valid_mask.sum() == 0
    assert matrix.ndim == 2
    assert matrix.shape[0] == 0  # no valid pixels


# ---------------------------------------------------------------------------
# Step 1: cluster spectra and representative pixel tests (require sklearn)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)
def test_cluster_spectra_recovers_two_domains():
    """Two-domain cube gets two stable distinct labels."""
    from ramanpl.mapping._cluster_seeds import _cluster_spectra

    cube = _two_domain_cube(Y=6, X=8, N=20, seed=42)
    labels, metadata = _cluster_spectra(
        cube, n_clusters=2, n_components=3, random_state=42
    )
    assert labels.shape == (6, 8)
    unique_labels = np.unique(labels[labels >= 0])
    assert len(unique_labels) == 2

    # Left half (i < 4) and right half (i >= 4) should have different dominant labels
    left_labels = labels[:, :4].ravel()
    right_labels = labels[:, 4:].ravel()
    assert np.bincount(left_labels).argmax() != np.bincount(right_labels).argmax()


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)
def test_representative_pixels_are_in_bounds():
    """One valid (x, y) per non-empty cluster; all coordinates are in bounds."""
    from ramanpl.mapping._cluster_seeds import _cluster_spectra, _representative_pixels

    Y, X = 6, 8
    cube = _two_domain_cube(Y=Y, X=X, N=20, seed=0)
    labels, metadata = _cluster_spectra(
        cube, n_clusters=2, n_components=3, random_state=0
    )
    reps = _representative_pixels(cube, labels, metadata)
    n_non_empty = len(np.unique(labels[labels >= 0]))
    assert len(reps) == n_non_empty
    for _, (x, y) in reps:
        assert 0 <= x < X, f"x={x} out of bounds [0, {X})"
        assert 0 <= y < Y, f"y={y} out of bounds [0, {Y})"


# ---------------------------------------------------------------------------
# Step 1: missing sklearn message test
# ---------------------------------------------------------------------------

def test_missing_sklearn_message_mentions_ml_extra():
    """Missing sklearn raises ImportError mentioning the [ml] extra."""
    with patch.dict(sys.modules, {"sklearn": None, "sklearn.preprocessing": None,
                                   "sklearn.decomposition": None,
                                   "sklearn.cluster": None}):
        with pytest.raises(ImportError, match=r"\[ml\]"):
            _require_sklearn_for_cluster_seeds()


# ---------------------------------------------------------------------------
# Step 4: cluster fit schedule tests
# ---------------------------------------------------------------------------

def _make_labels(Y, X, assignment):
    """Build a [Y, X] label array from a flat assignment list (row-major)."""
    return np.array(assignment, dtype=np.intp).reshape(Y, X)


def test_build_cluster_schedule_single_cluster():
    """One cluster: one seed entry whose members cover all other pixels."""
    Y, X = 2, 3
    labels = _make_labels(Y, X, [0, 0, 0, 0, 0, 0])
    reps = [(0, (0, 0))]  # cluster_id=0, (x=0, y=0) is the representative
    schedule = _build_cluster_schedule(labels, reps)
    assert len(schedule) == 1
    entry = schedule[0]
    assert entry["cluster"] == 0
    assert entry["seed"] == (0, 0)
    # Seed excluded from members
    assert (0, 0) not in entry["members"]
    # All other 5 pixels in members
    assert len(entry["members"]) == 5
    # Total = seed + members = 6
    all_pixels = {entry["seed"]} | set(entry["members"])
    assert len(all_pixels) == Y * X


def test_build_cluster_schedule_multiple_clusters():
    """Multiple clusters: each seed excluded from its own member list; no pixel duplicated."""
    Y, X = 4, 4
    # 4x4 grid: top-left 2x4 = cluster 0, bottom-left 2x4 = cluster 1
    assignment = [0] * 8 + [1] * 8
    labels = _make_labels(Y, X, assignment)
    reps = [(0, (0, 0)), (1, (0, 2))]  # cluster_id=0 seed (x=0,y=0); cluster_id=1 seed (x=0,y=2)
    schedule = _build_cluster_schedule(labels, reps)

    assert len(schedule) == 2
    all_member_pixels = []
    for entry in schedule:
        assert entry["seed"] not in entry["members"]
        all_member_pixels.extend(entry["members"])
        all_member_pixels.append(entry["seed"])

    # No pixel duplicated across clusters
    assert len(all_member_pixels) == len(set(all_member_pixels))
    # All 16 pixels covered
    assert len(all_member_pixels) == Y * X


def test_build_cluster_schedule_invalid_pixels_excluded():
    """Pixels with label -1 (invalid) do not appear in any schedule entry."""
    Y, X = 3, 3
    # Centre pixel is invalid (-1)
    labels = _make_labels(Y, X, [0, 0, 0, 0, -1, 0, 0, 0, 0])
    reps = [(0, (0, 0))]
    schedule = _build_cluster_schedule(labels, reps)

    all_pixels = set()
    for entry in schedule:
        all_pixels.add(entry["seed"])
        all_pixels.update(entry["members"])

    centre = (1, 1)  # x=1, y=1
    assert centre not in all_pixels


def test_build_cluster_schedule_empty_cluster_ignored():
    """A cluster label that has no assigned pixels produces no schedule entry."""
    Y, X = 2, 2
    # All pixels in cluster 0; cluster 1 has no pixels
    labels = _make_labels(Y, X, [0, 0, 0, 0])
    # Representative for cluster 1 points to a valid coord but no pixels belong to it
    reps = [(0, (0, 0)), (1, (1, 1))]  # rep for non-existent cluster 1 shouldn't matter
    schedule = _build_cluster_schedule(labels, reps)
    cluster_ids = [e["cluster"] for e in schedule]
    assert 1 not in cluster_ids  # cluster 1 absent from schedule


def test_build_cluster_schedule_member_order_is_row_major():
    """Members within each cluster are in row-major (y, x) order."""
    Y, X = 3, 4
    labels = np.zeros((Y, X), dtype=np.intp)
    reps = [(0, (0, 0))]  # cluster_id=0, seed at (x=0, y=0)
    schedule = _build_cluster_schedule(labels, reps)
    members = schedule[0]["members"]
    # All pixels except seed, in row-major order
    expected = [(i, j) for j in range(Y) for i in range(X) if (i, j) != (0, 0)]
    assert members == expected
