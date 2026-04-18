"""Tests confirming hard removal of deprecated APIs in v0.4.7."""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# poly_degree parameter removed from RamanFit and PLfit
# ---------------------------------------------------------------------------

def test_ramanfit_poly_degree_raises_type_error():
    from ramanpl.single_fit.RamanFit import RamanFit
    with pytest.raises(TypeError, match="poly_degree"):
        RamanFit(poly_degree=3)


def test_plfit_poly_degree_raises_type_error():
    from ramanpl.single_fit.PLfit import PLfit
    with pytest.raises(TypeError, match="poly_degree"):
        PLfit(poly_degree=3)


# ---------------------------------------------------------------------------
# BaselineAPI.parse_spec() removed
# ---------------------------------------------------------------------------

def test_baseline_api_parse_spec_does_not_exist():
    from ramanpl.baselineAPI import BaselineAPI
    assert not hasattr(BaselineAPI, "parse_spec"), (
        "parse_spec() was hard-removed in v0.4.7 and should no longer exist on BaselineAPI"
    )


# ---------------------------------------------------------------------------
# resolve_baseline_method_with_deprecation removed from _single_fit_core
# ---------------------------------------------------------------------------

def test_resolve_baseline_method_with_deprecation_does_not_exist():
    import ramanpl.single_fit._single_fit_core as core
    assert not hasattr(core, "resolve_baseline_method_with_deprecation"), (
        "resolve_baseline_method_with_deprecation was removed in v0.4.7"
    )


def test_poly_degree_sentinel_does_not_exist():
    import ramanpl.single_fit._single_fit_core as core
    assert not hasattr(core, "POLY_DEGREE_SENTINEL"), (
        "POLY_DEGREE_SENTINEL was removed in v0.4.7"
    )


# ---------------------------------------------------------------------------
# poly_degree not present in exported metadata
# ---------------------------------------------------------------------------

def test_ramanfit_export_meta_does_not_include_poly_degree():
    from ramanpl.single_fit._single_fit_core import build_single_fit_export_metadata
    from ramanpl.single_fit.RamanFit import RamanFit

    x = np.linspace(100.0, 300.0, 100)
    y = np.exp(-((x - 200.0) / 10.0) ** 2)

    fitter = RamanFit(
        spectra=y,
        wavenumber=x,
        baseline_method={"method": "poly", "poly_order": 1},
        background_remove=True,
        smoothing=False,
        custom_peaks={"peak1": ([180.0, 5.0, 0.1], [220.0, 30.0, 10.0])},
    )

    meta = build_single_fit_export_metadata(fitter)
    assert "poly_degree" not in meta


# ---------------------------------------------------------------------------
# Tuple baseline_method still normalised (not hard-removed at schema level)
# ---------------------------------------------------------------------------

def test_tuple_baseline_spec_still_normalises_poly():
    from ramanpl.schema import normalise_baseline_spec
    spec = normalise_baseline_spec(("poly", 4))
    assert spec["method"] == "poly"
    assert spec.get("poly_order") == 4
