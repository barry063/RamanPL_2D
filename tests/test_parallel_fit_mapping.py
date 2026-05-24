"""
Tests for n_jobs parallel fitting in RamanMapping and PLMapping (v0.6.4).

Uses synthetic data only — no file I/O.
All tests assume the loky backend is available (Windows/Linux/Mac).
"""
import numpy as np
import pytest
import warnings

from ramanpl.mapping._raman_mapping import RamanMapping
from ramanpl.mapping._pl_mapping import PLMapping
from ramanpl.mapping._parallel import _split_rows, _validate_parallel_kwargs


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_raman_mapping(ny=4, nx=3, n_pts=60, seed=42):
    """Return a RamanMapping on clean synthetic data (single Lorentzian peak)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1300.0, 1400.0, n_pts)
    cube = np.zeros((ny, nx, n_pts))
    for j in range(ny):
        for i in range(nx):
            noise = rng.normal(0.0, 0.05, n_pts)
            cube[j, i] = 1.0 + 50.0 * np.exp(-((x - 1350.0) / 8.0) ** 2) + noise
    peak = {"G": ([1330.0, 2.0, 0.01], [1370.0, 20.0, 500.0])}
    return RamanMapping.from_arrays(
        cube, x, nx, ny,
        custom_peaks=peak,
        data_range=(1300.0, 1400.0),
        background_remove=False,
        smoothing=False,
    )


def _make_pl_mapping(ny=4, nx=3, n_pts=60, seed=42):
    """Return a PLMapping on clean synthetic data (single Lorentzian peak)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1.8, 2.2, n_pts)
    cube = np.zeros((ny, nx, n_pts))
    for j in range(ny):
        for i in range(nx):
            noise = rng.normal(0.0, 0.02, n_pts)
            cube[j, i] = 0.5 + 30.0 * np.exp(-((x - 2.0) / 0.05) ** 2) + noise
    peak = {"A": ([1.85, 0.01, 0.01], [2.15, 0.2, 200.0])}
    return PLMapping.from_arrays(
        cube, x, nx, ny,
        custom_peaks=peak,
        data_range=(1.8, 2.2),
        background_remove=False,
        smoothing=False,
    )


# ---------------------------------------------------------------------------
# Test 1 — n_jobs=1 byte-parity (serial path unchanged)
# ---------------------------------------------------------------------------

def test_byte_parity_n_jobs_1():
    """n_jobs=1 must produce identical residual_map as the implicit default."""
    m1 = _make_raman_mapping()
    m2 = _make_raman_mapping()
    m1.fit_spectra(show_progress=False)
    m2.fit_spectra(n_jobs=1, show_progress=False)
    np.testing.assert_array_equal(m1.residual_map, m2.residual_map)
    np.testing.assert_array_equal(m1.fitted_params, m2.fitted_params)


# ---------------------------------------------------------------------------
# Test 2 — n_jobs=2 default mode (warm_start=False) byte-parity
# ---------------------------------------------------------------------------

def test_byte_parity_n_jobs_2_default_mode():
    """n_jobs=2 with warm_start=False must give same results as n_jobs=1."""
    m1 = _make_raman_mapping()
    m2 = _make_raman_mapping()
    m1.fit_spectra(warm_start=False, show_progress=False)
    m2.fit_spectra(warm_start=False, n_jobs=2, show_progress=False)
    np.testing.assert_array_almost_equal(m1.residual_map, m2.residual_map, decimal=10)
    np.testing.assert_array_almost_equal(m1.fitted_params, m2.fitted_params, decimal=10)


# ---------------------------------------------------------------------------
# Test 3 — n_jobs=2 warm_start + row_reset byte-parity
# ---------------------------------------------------------------------------

def test_byte_parity_n_jobs_2_warm_start_row_reset():
    """n_jobs=2 with warm_start=True, row_reset=True must give same results as n_jobs=1."""
    m1 = _make_raman_mapping()
    m2 = _make_raman_mapping()
    m1.fit_spectra(warm_start=True, row_reset=True, show_progress=False)
    m2.fit_spectra(warm_start=True, row_reset=True, n_jobs=2, show_progress=False)
    np.testing.assert_array_almost_equal(m1.residual_map, m2.residual_map, decimal=10)
    np.testing.assert_array_almost_equal(m1.fitted_params, m2.fitted_params, decimal=10)


# ---------------------------------------------------------------------------
# Test 4 — unsafe mode raises ValueError
# ---------------------------------------------------------------------------

def test_unsafe_mode_raises():
    """n_jobs > 1 with warm_start=True and row_reset=False must raise ValueError."""
    m = _make_raman_mapping()
    with pytest.raises(ValueError, match="row_reset=True"):
        m.fit_spectra(warm_start=True, row_reset=False, n_jobs=2, show_progress=False)
    with pytest.raises(ValueError, match="n_jobs=1"):
        m.fit_spectra(warm_start=True, row_reset=False, n_jobs=2, show_progress=False)


# ---------------------------------------------------------------------------
# Test 5 — curve_fit call count is invariant under n_jobs
# ---------------------------------------------------------------------------

def test_curve_fit_call_count_invariant():
    """Number of finite residuals (= successful fits) must be the same for any n_jobs."""
    m1 = _make_raman_mapping()
    m2 = _make_raman_mapping()
    m1.fit_spectra(warm_start=False, show_progress=False)
    m2.fit_spectra(warm_start=False, n_jobs=2, show_progress=False)
    n_finite_1 = np.sum(np.isfinite(m1.residual_map))
    n_finite_2 = np.sum(np.isfinite(m2.residual_map))
    assert n_finite_1 == n_finite_2, (
        f"Success count differs: n_jobs=1 → {n_finite_1}, n_jobs=2 → {n_finite_2}"
    )


# ---------------------------------------------------------------------------
# Test 6 — invalid n_jobs raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_n_jobs_raises():
    m = _make_raman_mapping()
    with pytest.raises(ValueError):
        m.fit_spectra(n_jobs=0, show_progress=False)
    with pytest.raises(ValueError):
        m.fit_spectra(n_jobs=-1, show_progress=False)
    with pytest.raises(ValueError):
        m.fit_spectra(n_jobs=1.5, show_progress=False)


# ---------------------------------------------------------------------------
# Test 7 — n_jobs > Y clamps and emits UserWarning
# ---------------------------------------------------------------------------

def test_n_jobs_exceeds_Y_clamps():
    """n_jobs larger than Y must clamp to Y and emit a UserWarning."""
    m = _make_raman_mapping(ny=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit_spectra(n_jobs=10, show_progress=False)
    clamped_warnings = [x for x in w if issubclass(x.category, UserWarning)
                        and "clamping" in str(x.message).lower()]
    assert len(clamped_warnings) >= 1
    assert np.sum(np.isfinite(m.residual_map)) > 0


# ---------------------------------------------------------------------------
# Test 8 — seed_coord works with parallel (requires row_reset=True)
# ---------------------------------------------------------------------------

def test_seed_coord_works_with_parallel():
    """fit_spectra(seed_coord=..., row_reset=True, n_jobs=2) should complete without error."""
    m = _make_raman_mapping(ny=4, nx=3)
    m.fit_spectra(
        seed_coord=(0, 0),
        row_reset=True,
        n_jobs=2,
        show_progress=False,
    )
    assert np.sum(np.isfinite(m.residual_map)) > 0


# ---------------------------------------------------------------------------
# _split_rows and _validate_parallel_kwargs unit tests
# ---------------------------------------------------------------------------

def test_split_rows_coverage():
    """All rows covered, no overlap, correct band count."""
    bands = _split_rows(7, 3)
    assert len(bands) == 3
    flat = []
    for j_start, j_end in bands:
        flat.extend(range(j_start, j_end))
    assert sorted(flat) == list(range(7))


def test_validate_parallel_kwargs_clamping():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_parallel_kwargs(n_jobs=10, warm_start=False, row_reset=False, Y=4)
    assert result == 4
    assert any(issubclass(x.category, UserWarning) for x in w)


def test_validate_parallel_kwargs_unsafe_raises():
    with pytest.raises(ValueError, match="row_reset=True"):
        _validate_parallel_kwargs(n_jobs=2, warm_start=True, row_reset=False, Y=4)
