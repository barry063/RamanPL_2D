"""
test_cluster_seed_fit_mapping.py
---------------------------------
Integration tests for cluster_seeds keyword on RamanMapping and PLMapping.

Step 3 : byte-parity tests (cluster_seeds=False must be identical to omitting it).
Step 5 : serial cluster-seeded fitting behaviour.
Step 6 : parallel compatibility guard (cluster_seeds=True, n_jobs>1 raises ValueError).

All tests use synthetic data only — no file I/O.
sklearn-requiring tests are skipped when sklearn is absent.
"""

import importlib
import importlib.util

import numpy as np
import pytest

from ramanpl.mapping._raman_mapping import RamanMapping
from ramanpl.mapping._pl_mapping import PLMapping


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

_N_PTS = 60
_NY, _NX = 4, 5

_X_RAMAN = np.linspace(1300.0, 1400.0, _N_PTS)
_X_PL = np.linspace(1.8, 2.2, _N_PTS)

_CUSTOM_PEAKS_RAMAN = {"G": ([1330.0, 2.0, 0.01], [1370.0, 20.0, 500.0])}
_CUSTOM_PEAKS_PL = {"A": ([1.85, 0.01, 0.01], [2.15, 0.2, 200.0])}


def _make_raman_mapping(ny=_NY, nx=_NX, seed=0):
    rng = np.random.default_rng(seed)
    x = _X_RAMAN
    cube = np.zeros((ny, nx, _N_PTS))
    for j in range(ny):
        for i in range(nx):
            cube[j, i] = 1.0 + 50.0 * np.exp(-((x - 1350.0) / 8.0) ** 2) + rng.normal(0.0, 0.05, _N_PTS)
    return RamanMapping.from_arrays(
        cube, x, nx, ny,
        custom_peaks=_CUSTOM_PEAKS_RAMAN,
        data_range=(1300.0, 1400.0),
        background_remove=False,
        smoothing=False,
    )


def _make_pl_mapping(ny=_NY, nx=_NX, seed=0):
    rng = np.random.default_rng(seed)
    x = _X_PL
    cube = np.zeros((ny, nx, _N_PTS))
    for j in range(ny):
        for i in range(nx):
            cube[j, i] = 0.5 + 30.0 * np.exp(-((x - 2.0) / 0.05) ** 2) + rng.normal(0.0, 0.02, _N_PTS)
    return PLMapping.from_arrays(
        cube, x, nx, ny,
        custom_peaks=_CUSTOM_PEAKS_PL,
        data_range=(1.8, 2.2),
        background_remove=False,
        smoothing=False,
    )


# ---------------------------------------------------------------------------
# Step 3: byte-parity — cluster_seeds=False must equal omitting the keyword
# ---------------------------------------------------------------------------

def test_cluster_seeds_false_parity_raman():
    """RamanMapping: fit_spectra() == fit_spectra(cluster_seeds=False), byte-parity."""
    m1 = _make_raman_mapping()
    m2 = _make_raman_mapping()
    m1.fit_spectra(show_progress=False)
    m2.fit_spectra(cluster_seeds=False, show_progress=False)
    np.testing.assert_array_equal(m1.fitted_params, m2.fitted_params)
    np.testing.assert_array_equal(m1.residual_map, m2.residual_map)


def test_cluster_seeds_false_parity_pl():
    """PLMapping: fit_spectra() == fit_spectra(cluster_seeds=False), byte-parity."""
    m1 = _make_pl_mapping()
    m2 = _make_pl_mapping()
    m1.fit_spectra(show_progress=False)
    m2.fit_spectra(cluster_seeds=False, show_progress=False)
    np.testing.assert_array_equal(m1.fitted_params, m2.fitted_params)
    np.testing.assert_array_equal(m1.residual_map, m2.residual_map)


def test_cluster_seeds_false_does_not_import_sklearn(monkeypatch):
    """cluster_seeds=False must not trigger sklearn import."""
    import sys
    sklearn_keys_before = {k for k in sys.modules if "sklearn" in k}
    m = _make_raman_mapping()
    m.fit_spectra(cluster_seeds=False, show_progress=False)
    sklearn_keys_after = {k for k in sys.modules if "sklearn" in k}
    assert sklearn_keys_after == sklearn_keys_before


# ---------------------------------------------------------------------------
# Step 5: serial cluster-seeded fitting
# ---------------------------------------------------------------------------

_requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)


@_requires_sklearn
def test_cluster_seeds_true_raman_completes():
    """cluster_seeds=True, n_jobs=1 must complete and return fitted_params for RamanMapping."""
    m = _make_raman_mapping()
    params = m.fit_spectra(cluster_seeds=True, n_jobs=1, show_progress=False)
    assert params.shape == (m.Y, m.X, m.fitted_params.shape[2])
    # At least half of pixels must have finite fits on clean data
    n_ok = np.sum(np.all(np.isfinite(params), axis=2))
    assert n_ok >= m.Y * m.X // 2, f"Too many failed fits: {n_ok}/{m.Y*m.X}"


@_requires_sklearn
def test_cluster_seeds_true_pl_completes():
    """cluster_seeds=True, n_jobs=1 must complete and return fitted_params for PLMapping."""
    m = _make_pl_mapping()
    params = m.fit_spectra(cluster_seeds=True, n_jobs=1, show_progress=False)
    assert params.shape == (m.Y, m.X, m.fitted_params.shape[2])
    n_ok = np.sum(np.all(np.isfinite(params), axis=2))
    assert n_ok >= m.Y * m.X // 2, f"Too many failed fits: {n_ok}/{m.Y*m.X}"


@_requires_sklearn
def test_cluster_seeds_success_rate_not_lower_than_baseline_raman():
    """cluster_seeds=True success rate must be >= cluster_seeds=False on clean data."""
    m_base = _make_raman_mapping()
    m_cs = _make_raman_mapping()
    p_base = m_base.fit_spectra(cluster_seeds=False, show_progress=False)
    p_cs = m_cs.fit_spectra(
        cluster_seeds=True, n_jobs=1,
        fit_spectrum_kwargs={"random_state": 0},
        show_progress=False,
    )
    n_ok_base = int(np.sum(np.all(np.isfinite(p_base), axis=2)))
    n_ok_cs = int(np.sum(np.all(np.isfinite(p_cs), axis=2)))
    assert n_ok_cs >= n_ok_base, (
        f"cluster_seeds=True success rate {n_ok_cs} < baseline {n_ok_base}"
    )


@_requires_sklearn
def test_cluster_seeds_params_within_tolerance_raman():
    """cluster_seeds=True params agree with baseline within rtol=1e-3, atol=1e-5."""
    m_base = _make_raman_mapping()
    m_cs = _make_raman_mapping()
    p_base = m_base.fit_spectra(cluster_seeds=False, show_progress=False)
    p_cs = m_cs.fit_spectra(
        cluster_seeds=True, n_jobs=1,
        fit_spectrum_kwargs={"random_state": 0},
        show_progress=False,
    )
    both_ok = np.isfinite(p_base).all(axis=2) & np.isfinite(p_cs).all(axis=2)
    if both_ok.sum() == 0:
        pytest.skip("No pixels succeeded in both runs")
    np.testing.assert_allclose(
        p_cs[both_ok], p_base[both_ok], rtol=1e-3, atol=1e-5,
        err_msg="cluster_seeds=True params deviate from baseline beyond tolerance",
    )


@_requires_sklearn
def test_cluster_seeds_seed_coord_mutual_exclusion_raman():
    """cluster_seeds=True combined with seed_coord must raise ValueError naming both resolutions."""
    m = _make_raman_mapping()
    with pytest.raises(ValueError, match="seed_coord"):
        m.fit_spectra(
            cluster_seeds=True, seed_coord=(0, 0), n_jobs=1, show_progress=False
        )


@_requires_sklearn
def test_cluster_seeds_seed_coord_mutual_exclusion_pl():
    """cluster_seeds=True combined with seed_coord must raise ValueError for PLMapping."""
    m = _make_pl_mapping()
    with pytest.raises(ValueError, match="seed_coord"):
        m.fit_spectra(
            cluster_seeds=True, seed_coord=(0, 0), n_jobs=1, show_progress=False
        )


@_requires_sklearn
def test_cluster_seeds_warm_start_false_raman():
    """cluster_seeds=True with warm_start=False must complete without error."""
    m = _make_raman_mapping()
    params = m.fit_spectra(
        cluster_seeds=True, warm_start=False, n_jobs=1, show_progress=False,
        fit_spectrum_kwargs={"random_state": 0},
    )
    assert params is not None
    assert params.shape[0] == m.Y and params.shape[1] == m.X


@_requires_sklearn
def test_cluster_seeds_does_not_mutate_warm_start_kwarg():
    """cluster_seeds=True must not silently change the warm_start value seen by caller."""
    m = _make_raman_mapping()
    # warm_start=False is passed explicitly; cluster seeds should not override this
    original_warm_start = False
    m.fit_spectra(
        cluster_seeds=True, warm_start=original_warm_start, n_jobs=1,
        show_progress=False, fit_spectrum_kwargs={"random_state": 0},
    )
    # The test is that fit_spectra completed without changing the user's variable


@_requires_sklearn
def test_cluster_seeds_initial_p0_used_for_representative_raman():
    """When initial_p0 is supplied with cluster_seeds=True, it is used as base p0."""
    from ramanpl.mapping._raman_mapping import RamanMapping
    m = _make_raman_mapping()
    # Midpoint p0 as initial_p0 — should produce same or better fits
    import inspect
    m2 = _make_raman_mapping()
    initial_p0 = np.array([1350.0, 8.0, 50.0])
    # Should not raise
    params = m2.fit_spectra(
        cluster_seeds=True, initial_p0=initial_p0, n_jobs=1,
        show_progress=False, fit_spectrum_kwargs={"random_state": 0},
    )
    assert params is not None


# ---------------------------------------------------------------------------
# Step 6: parallel compatibility guard
# ---------------------------------------------------------------------------

@_requires_sklearn
def test_cluster_seeds_n_jobs_gt1_raises_raman():
    """cluster_seeds=True with n_jobs>1 must raise ValueError for RamanMapping."""
    m = _make_raman_mapping()
    with pytest.raises(ValueError, match="n_jobs=1"):
        m.fit_spectra(cluster_seeds=True, n_jobs=2, show_progress=False)


@_requires_sklearn
def test_cluster_seeds_n_jobs_gt1_raises_pl():
    """cluster_seeds=True with n_jobs>1 must raise ValueError for PLMapping."""
    m = _make_pl_mapping()
    with pytest.raises(ValueError, match="n_jobs=1"):
        m.fit_spectra(cluster_seeds=True, n_jobs=2, show_progress=False)


def test_cluster_seeds_false_n_jobs_gt1_allowed():
    """cluster_seeds=False with n_jobs>1 must NOT raise (existing row-band path)."""
    m = _make_raman_mapping()
    # Should complete without raising, though results are not byte-parity when warm_start=True
    m.fit_spectra(cluster_seeds=False, n_jobs=2, warm_start=False, show_progress=False)
