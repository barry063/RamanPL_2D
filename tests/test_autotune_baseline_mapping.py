"""
Tests for autotune_baseline / apply_choice on mapping objects (v0.6.3).

Uses synthetic data only — no file I/O.
"""

import numpy as np
import pytest
import warnings
import matplotlib
matplotlib.use("Agg")  # headless

from ramanpl.mapping._raman_mapping import RamanMapping
from ramanpl._autotune import BaselineAutotuneResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_cube(n_pts=80, ny=2, nx=2, peak_centre=400.0):
    """Return (cube, x, peak_dict) — a clean Lorentzian on a flat baseline."""
    x = np.linspace(300.0, 500.0, n_pts)
    cube = np.zeros((ny, nx, n_pts))
    for j in range(ny):
        for i in range(nx):
            cube[j, i] = 5.0 + 80.0 * np.exp(-((x - peak_centre) / 8.0) ** 2)
    peak = {"A": ([390.0, 2.0, 0.01], [410.0, 20.0, 200.0])}
    return cube, x, peak


@pytest.fixture
def mapping_with_baseline():
    """2×2 RamanMapping with poly baseline enabled."""
    cube, x, peak = _make_synthetic_cube()
    return RamanMapping.from_arrays(
        cube, x, 2, 2,
        custom_peaks=peak,
        data_range=(300.0, 500.0),
        background_remove=True,
        baseline_method={"method": "poly", "poly_order": 1},
        smoothing=False,
    )


@pytest.fixture
def mapping_no_baseline():
    """2×2 RamanMapping with background_remove=False."""
    cube, x, peak = _make_synthetic_cube()
    return RamanMapping.from_arrays(
        cube, x, 2, 2,
        custom_peaks=peak,
        data_range=(300.0, 500.0),
        background_remove=False,
        smoothing=False,
    )


# ---------------------------------------------------------------------------
# Test 1 — ranking is sorted ascending and winner matches ranking[0]
# ---------------------------------------------------------------------------

def test_autotune_returns_ranking_sorted_ascending(mapping_with_baseline):
    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}, "gaussian": {"gaussian_sigma": [5, 10, 20, 50, 100]}},
        plot=False,
    )
    assert isinstance(result, BaselineAutotuneResult)
    rmses = [r["rmse"] for r in result.ranking]
    assert rmses == sorted(rmses, key=lambda v: (not np.isfinite(v), v))
    assert result.winner == {"method": result.ranking[0]["method"], **result.ranking[0]["kwargs"]}
    finite = [v for v in rmses if np.isfinite(v)]
    assert len(finite) > 0, "All candidates scored inf — sum_peaks stride or pipeline issue"


# ---------------------------------------------------------------------------
# Test 2 — autotune does NOT mutate baseline_method
# ---------------------------------------------------------------------------

def test_autotune_does_not_mutate_baseline_method(mapping_with_baseline):
    before = dict(mapping_with_baseline.baseline_method)
    mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    assert mapping_with_baseline.baseline_method == before


# ---------------------------------------------------------------------------
# Test 3 — autotune does NOT invalidate the preprocessed cube cache
# ---------------------------------------------------------------------------

def test_autotune_does_not_invalidate_cube_cache(mapping_with_baseline):
    # Warm the cache
    _ = mapping_with_baseline._get_processed_mapping_cube()
    cache_before = mapping_with_baseline._preprocessed_cube_cache
    assert cache_before is not None

    mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    # Cache should be identical object (untouched)
    assert mapping_with_baseline._preprocessed_cube_cache is cache_before


# ---------------------------------------------------------------------------
# Test 4 — plot=False returns no figure
# ---------------------------------------------------------------------------

def test_autotune_plot_false_returns_no_figure(mapping_with_baseline):
    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    assert result.figure is None


# ---------------------------------------------------------------------------
# Test 5 — plot=True returns a matplotlib Figure
# ---------------------------------------------------------------------------

def test_autotune_plot_true_returns_figure(mapping_with_baseline):
    import matplotlib.figure
    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}, "gaussian": {"gaussian_sigma": [5, 10, 20, 50, 100]}},
        plot=True,
    )
    assert isinstance(result.figure, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 6 — apply_choice updates the pipeline's BaselineSubtract step
# ---------------------------------------------------------------------------

def test_apply_choice_updates_pipeline(mapping_with_baseline):
    from ramanpl.preprocessing import BaselineSubtract

    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    mapping_with_baseline.apply_choice(result.winner)

    baseline_steps = [
        s for s in mapping_with_baseline.preprocessing.steps
        if isinstance(s, BaselineSubtract)
    ]
    assert len(baseline_steps) == 1
    committed_spec = baseline_steps[0].baseline_spec
    assert committed_spec.get("method") == result.winner.get("method")


# ---------------------------------------------------------------------------
# Test 7 — apply_choice updates legacy baseline attributes
# ---------------------------------------------------------------------------

def test_apply_choice_updates_legacy_attributes(mapping_with_baseline):
    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    winner_method = result.winner["method"]
    mapping_with_baseline.apply_choice(result.winner)

    assert mapping_with_baseline.baseline_method.get("method") == winner_method
    assert mapping_with_baseline._baseline_method == winner_method


# ---------------------------------------------------------------------------
# Test 8 — apply_choice invalidates the preprocessed cube cache
# ---------------------------------------------------------------------------

def test_apply_choice_invalidates_cube_cache(mapping_with_baseline):
    # Warm the cache
    _ = mapping_with_baseline._get_processed_mapping_cube()
    assert mapping_with_baseline._preprocessed_cube_cache is not None

    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    mapping_with_baseline.apply_choice(result.winner)

    assert mapping_with_baseline._preprocessed_cube_cache is None
    assert mapping_with_baseline._preprocessed_x_cache is None
    assert mapping_with_baseline._preprocess_meta == {}


# ---------------------------------------------------------------------------
# Test 9 — apply_choice raises when there is no BaselineSubtract step
# ---------------------------------------------------------------------------

def test_apply_choice_raises_when_no_baseline_step(mapping_no_baseline):
    with pytest.raises(ValueError, match="BaselineSubtract"):
        mapping_no_baseline.apply_choice({"method": "poly", "poly_order": 2})


# ---------------------------------------------------------------------------
# Test 10 — export includes baseline_autotune block after apply+fit+export
# ---------------------------------------------------------------------------

def test_export_includes_autotune_block(tmp_path, mapping_with_baseline):
    import io

    result = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    mapping_with_baseline.apply_choice(result.winner)
    mapping_with_baseline.fit_spectra()

    # Export to a temp file and read back
    out_file = str(tmp_path / "export_test.txt")
    mapping_with_baseline.export_fit_map(out_file)

    with open(out_file, "r") as f:
        content = f.read()

    assert "baseline_autotune" in content


# ---------------------------------------------------------------------------
# Test 11 — autotune raises for pipelines with >1 BaselineSubtract steps
# (regression for P2: broad except ValueError silently used wrong pipeline)
# ---------------------------------------------------------------------------

def test_autotune_raises_for_multiple_baseline_steps():
    from ramanpl.preprocessing import Pipeline, BaselineSubtract
    from ramanpl.schema import normalise_baseline_spec

    cube, x, peak = _make_synthetic_cube()
    spec = normalise_baseline_spec({"method": "poly", "poly_order": 1})
    pipe = Pipeline(
        steps=[
            BaselineSubtract(baseline_spec=spec),
            BaselineSubtract(baseline_spec=spec),  # duplicate — invalid
        ],
        backend="native",
        name="double_baseline",
    )
    mapping = RamanMapping.from_arrays(
        cube, x, 2, 2,
        custom_peaks=peak,
        data_range=(300.0, 500.0),
        preprocessing=pipe,
    )
    with pytest.raises(ValueError, match="BaselineSubtract"):
        mapping.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
            plot=False,
        )


# ---------------------------------------------------------------------------
# v0.6.3 — method_grids API tests
# ---------------------------------------------------------------------------

def test_method_grids_and_methods_are_mutually_exclusive(mapping_with_baseline):
    with pytest.raises(TypeError, match="not both"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={"poly": {"poly_order": [1]}},
            methods=["poly"],
            plot=False,
        )


def test_deprecated_methods_kwarg_emits_warning(mapping_with_baseline):
    with pytest.warns(DeprecationWarning, match="method_grids"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            methods=["poly"],
            plot=False,
        )


def test_deprecated_kwargs_produce_same_ranking_as_method_grids(mapping_with_baseline):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r_old = mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            methods=["airpls"],
            plot=False,
        )
    r_new = mapping_with_baseline.autotune_baseline(
        seed_coord=(0, 0),
        method_grids={"airpls": {"lam": [1e3, 1e4, 1e5, 1e6], "niter": [50]}},
        plot=False,
    )
    assert [e["rmse"] for e in r_old.ranking] == [e["rmse"] for e in r_new.ranking]


def test_method_grids_empty_dict_raises(mapping_with_baseline):
    with pytest.raises(ValueError, match="empty"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={},
            plot=False,
        )


def test_method_grids_unknown_method_raises(mapping_with_baseline):
    with pytest.raises(ValueError, match="Unknown method"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={"polynomial": {"poly_order": [1]}},
            plot=False,
        )


def test_method_grids_unknown_param_raises(mapping_with_baseline):
    with pytest.raises(ValueError, match="Unknown parameter"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={"poly": {"order": [1]}},
            plot=False,
        )


def test_method_grids_empty_param_list_raises(mapping_with_baseline):
    with pytest.raises(ValueError, match="empty"):
        mapping_with_baseline.autotune_baseline(
            seed_coord=(0, 0),
            method_grids={"poly": {"poly_order": []}},
            plot=False,
        )
