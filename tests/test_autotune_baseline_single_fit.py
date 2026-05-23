"""
Tests for autotune_baseline / apply_choice on single-fit objects (v0.6.3).

Uses synthetic data only — no file I/O.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless

from ramanpl.single_fit.RamanFit import RamanFit
from ramanpl._autotune import BaselineAutotuneResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raman_fit(background_remove=True, baseline_method=None):
    """Return a RamanFit with a single Lorentzian peak on a flat background."""
    x = np.linspace(300.0, 500.0, 100)
    y = 5.0 + 80.0 * np.exp(-((x - 400.0) / 8.0) ** 2)

    if baseline_method is None:
        baseline_method = {"method": "poly", "poly_order": 1}

    custom_peaks = {"A": ([390.0, 2.0, 0.01], [410.0, 20.0, 200.0])}
    return RamanFit(
        spectra=y,
        wavenumber=x,
        custom_peaks=custom_peaks,
        background_remove=background_remove,
        baseline_method=baseline_method,
        smoothing=False,
        normalize=False,
    )


@pytest.fixture
def raman_fit():
    return _make_raman_fit(background_remove=True)


@pytest.fixture
def raman_fit_no_baseline():
    return _make_raman_fit(background_remove=False)


# ---------------------------------------------------------------------------
# Test 1 — ranking is sorted ascending and winner matches ranking[0]
# ---------------------------------------------------------------------------

def test_autotune_returns_ranking_sorted_ascending(raman_fit):
    result = raman_fit.autotune_baseline(
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

def test_autotune_does_not_mutate_baseline_method(raman_fit):
    before = dict(raman_fit.baseline_method)
    raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    assert raman_fit.baseline_method == before


# ---------------------------------------------------------------------------
# Test 3 — autotune does NOT mutate processed_spectra
# ---------------------------------------------------------------------------

def test_autotune_does_not_mutate_processed_spectra(raman_fit):
    proc_before = raman_fit.processed_spectra.copy()
    raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    np.testing.assert_array_equal(raman_fit.processed_spectra, proc_before)


# ---------------------------------------------------------------------------
# Test 4 — plot=False returns no figure
# ---------------------------------------------------------------------------

def test_autotune_plot_false_returns_no_figure(raman_fit):
    result = raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    assert result.figure is None


# ---------------------------------------------------------------------------
# Test 5 — plot=True returns a matplotlib Figure
# ---------------------------------------------------------------------------

def test_autotune_plot_true_returns_figure(raman_fit):
    import matplotlib.figure
    result = raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}, "gaussian": {"gaussian_sigma": [5, 10, 20, 50, 100]}},
        plot=True,
    )
    assert isinstance(result.figure, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 6 — apply_choice updates the pipeline's BaselineSubtract step
# ---------------------------------------------------------------------------

def test_apply_choice_updates_pipeline(raman_fit):
    from ramanpl.preprocessing import BaselineSubtract

    result = raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    raman_fit.apply_choice(result.winner)

    baseline_steps = [
        s for s in raman_fit.preprocessing.steps
        if isinstance(s, BaselineSubtract)
    ]
    assert len(baseline_steps) == 1
    committed_spec = baseline_steps[0].baseline_spec
    assert committed_spec.get("method") == result.winner.get("method")


# ---------------------------------------------------------------------------
# Test 7 — apply_choice updates legacy baseline attributes
# ---------------------------------------------------------------------------

def test_apply_choice_updates_legacy_attributes(raman_fit):
    result = raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    winner_method = result.winner["method"]
    raman_fit.apply_choice(result.winner)

    assert raman_fit.baseline_method.get("method") == winner_method
    assert raman_fit._baseline_method == winner_method


# ---------------------------------------------------------------------------
# Test 8 — apply_choice with a DIFFERENT baseline changes processed_spectra
# ---------------------------------------------------------------------------

def test_apply_choice_changes_processed_spectra(raman_fit):
    proc_before = raman_fit.processed_spectra.copy()

    # Apply a notably different baseline (gaussian vs the current poly)
    raman_fit.apply_choice({"method": "gaussian", "gaussian_sigma": 20})

    # Processed spectra must change (different baseline method)
    assert not np.allclose(raman_fit.processed_spectra, proc_before)


# ---------------------------------------------------------------------------
# Test 9 — apply_choice raises when there is no BaselineSubtract step
# ---------------------------------------------------------------------------

def test_apply_choice_raises_when_no_baseline_step(raman_fit_no_baseline):
    with pytest.raises(ValueError, match="BaselineSubtract"):
        raman_fit_no_baseline.apply_choice({"method": "poly", "poly_order": 2})


# ---------------------------------------------------------------------------
# Test 10 — export includes baseline_autotune block
# ---------------------------------------------------------------------------

def test_export_includes_autotune_block(tmp_path, raman_fit):
    result = raman_fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}},
        plot=False,
    )
    raman_fit.apply_choice(result.winner)
    raman_fit.fit_spectrum()

    out_file = str(tmp_path / "export_test.txt")
    raman_fit.export_fit(out_file)

    with open(out_file, "r") as f:
        content = f.read()

    assert "baseline_autotune" in content


# ---------------------------------------------------------------------------
# Test 11 — apply_choice re-applies the pipeline (processed_spectra refreshed)
# ---------------------------------------------------------------------------

def test_single_fit_apply_choice_reapplies_pipeline(raman_fit):
    """apply_choice with a different baseline produces different processed_spectra."""
    proc_poly = raman_fit.processed_spectra.copy()

    # Switch to gaussian baseline
    raman_fit.apply_choice({"method": "gaussian", "gaussian_sigma": 10})
    proc_gaussian = raman_fit.processed_spectra.copy()

    assert not np.allclose(proc_poly, proc_gaussian), (
        "processed_spectra should differ after switching baseline method"
    )


# ---------------------------------------------------------------------------
# Test 12 — apply_choice refreshes all 7 derived attributes
# ---------------------------------------------------------------------------

def test_single_fit_apply_choice_refreshes_derived_attrs():
    """All seven derived attributes are non-None and consistent after apply_choice."""
    fit = _make_raman_fit(background_remove=True)

    # Force a different baseline to ensure the attributes change
    fit.apply_choice({"method": "gaussian", "gaussian_sigma": 15})

    # 1. _baseline: set by BaselineSubtract step
    assert fit._baseline is not None, "_baseline should be set"

    # 2. _corrected_spectra: set when a baseline or smoothing was applied
    assert fit._corrected_spectra is not None, "_corrected_spectra should be set"

    # 3. peak_intensity: max of processed_spectra (positive)
    assert np.isfinite(fit.peak_intensity) and fit.peak_intensity > 0

    # 4. intensity_normal: processed_spectra / peak_intensity
    np.testing.assert_allclose(
        fit.intensity_normal,
        fit.processed_spectra / fit.peak_intensity,
        rtol=1e-12,
    )

    # 5. preprocessing_backend_resolved: set by Pipeline.apply
    assert fit.preprocessing_backend_resolved is not None

    # 6. preprocessing_backend_info: dict with backend resolution info
    assert isinstance(fit.preprocessing_backend_info, dict)

    # 7. _backend_outcome: same object as preprocessing_backend_info
    assert fit._backend_outcome is fit.preprocessing_backend_info


# ---------------------------------------------------------------------------
# Test 13 — autotune scoring is valid when pipeline contains a crop step
# (regression: x was taken from the post-crop axis while y_raw was the
# full pristine spectrum, causing a length mismatch → all rmse=inf)
# ---------------------------------------------------------------------------

def test_autotune_works_with_crop_in_pipeline():
    """Ranking must contain at least one finite RMSE when a CropByRange step
    is present in the pipeline (i.e. the spectral axis is shorter than the
    raw spectrum passed at construction)."""
    from ramanpl.preprocessing import Pipeline, CropByRange, BaselineSubtract
    from ramanpl.schema import normalise_baseline_spec

    x_full = np.linspace(250.0, 550.0, 200)
    y_full = 5.0 + 80.0 * np.exp(-((x_full - 400.0) / 8.0) ** 2)

    custom_peaks = {"A": ([390.0, 2.0, 0.01], [410.0, 20.0, 200.0])}

    pipe = Pipeline(
        steps=[
            CropByRange(data_range=(300.0, 500.0)),
            BaselineSubtract(baseline_spec=normalise_baseline_spec({"method": "poly", "poly_order": 1})),
        ],
        backend="native",
        name="crop_then_baseline",
    )

    fit = RamanFit(
        spectra=y_full,
        wavenumber=x_full,
        custom_peaks=custom_peaks,
        preprocessing=pipe,
        normalize=False,
    )

    # The cropped axis is shorter than the pristine spectrum — this is exactly
    # the condition that triggered the bug.
    assert len(fit.wavenumber) < len(fit._x_axis_pristine)

    result = fit.autotune_baseline(
        method_grids={"poly": {"poly_order": [1, 2, 3, 4, 5]}, "gaussian": {"gaussian_sigma": [5, 10, 20, 50, 100]}},
        plot=False,
    )

    finite_rmses = [r["rmse"] for r in result.ranking if np.isfinite(r["rmse"])]
    assert len(finite_rmses) > 0, (
        "All candidates scored inf — x/y_raw length mismatch detected. "
        "Pristine x-axis must be used for scoring, not the post-crop axis."
    )


# ---------------------------------------------------------------------------
# v0.6.3 — method_grids API tests (mirrored from mapping tests)
# ---------------------------------------------------------------------------

def test_method_grids_and_methods_are_mutually_exclusive(raman_fit):
    with pytest.raises(TypeError, match="not both"):
        raman_fit.autotune_baseline(
            method_grids={"poly": {"poly_order": [1]}},
            methods=["poly"],
            plot=False,
        )


def test_deprecated_methods_kwarg_emits_warning(raman_fit):
    with pytest.warns(DeprecationWarning, match="method_grids"):
        raman_fit.autotune_baseline(
            methods=["poly"],
            plot=False,
        )
