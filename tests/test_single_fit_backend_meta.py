"""
test_single_fit_backend_meta.py

Verify that single-spectrum fitter objects (RamanFit / PLfit) record a canonical
_backend_outcome attribute after preprocessing, and that build_single_fit_export_metadata()
uses the shared serialise_backend_provenance() helper.
"""
import numpy as np
import pytest

from ramanpl.preprocessing import (
    Pipeline,
    CropByRange,
    SmoothSavGol,
    BaselineSubtract,
)


def _make_raman_fitter(backend="native"):
    """Return a fitted RamanFit instance using a simple synthetic spectrum."""
    import ramanpl.RamanFit as rf_mod

    x = np.linspace(100, 2000, 200)
    # Lorentzian peak at 520 cm⁻¹ + noise
    y = 1 / ((x - 520) ** 2 + 10 ** 2) * 1e4 + np.random.default_rng(0).normal(0, 0.01, x.size)
    y = np.clip(y, 0, None)

    pipe = Pipeline(
        steps=[CropByRange((200, 1800)), SmoothSavGol(window_length=11, polyorder=3)],
        backend=backend,
    )

    # custom_peaks format: {name: ([centre_lb, scale_lb, amp_lb], [centre_ub, scale_ub, amp_ub])}
    fitter = rf_mod.RamanFit(
        spectra=y,
        wavenumber=x,
        custom_peaks={"Si": ([490.0, 1.0, 0.01], [560.0, 50.0, 1e6])},
        preprocessing=pipe,
    )
    fitter.fit_spectrum()
    return fitter


def test_raman_fitter_records_backend_outcome_after_fit():
    fitter = _make_raman_fitter(backend="native")
    assert hasattr(fitter, "_backend_outcome")
    assert fitter._backend_outcome is not None
    assert isinstance(fitter._backend_outcome, dict)


def test_raman_fitter_backend_outcome_is_canonical():
    """_backend_outcome must contain the required canonical keys."""
    from ramanpl.schema import validate_backend_outcome_keys
    fitter = _make_raman_fitter(backend="native")
    # Should not raise
    validate_backend_outcome_keys(fitter._backend_outcome)


def test_raman_fitter_native_backend_outcome_fields():
    fitter = _make_raman_fitter(backend="native")
    outcome = fitter._backend_outcome
    assert outcome["requested_backend"] == "native"
    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is False
    assert outcome["fallback_reason"] is None


def test_build_single_fit_export_metadata_includes_backend_provenance():
    from ramanpl.single_fit._single_fit_core import build_single_fit_export_metadata
    fitter = _make_raman_fitter(backend="native")
    meta = build_single_fit_export_metadata(fitter)
    assert "preprocessing_backend_requested" in meta
    assert "preprocessing_backend_resolved" in meta
    assert meta["preprocessing_backend_requested"] == "native"
    assert meta["preprocessing_backend_resolved"] == "native"


def test_build_single_fit_export_metadata_no_fallback_reason_for_native():
    from ramanpl.single_fit._single_fit_core import build_single_fit_export_metadata
    fitter = _make_raman_fitter(backend="native")
    meta = build_single_fit_export_metadata(fitter)
    assert "preprocessing_backend_fallback_reason" not in meta


def test_single_fit_forced_ramanspy_raises_for_unsupported_pipeline():
    """Forced ramanspy with a gaussian-baseline pipeline raises NotImplementedError."""
    import ramanpl.RamanFit as rf_mod

    x = np.linspace(100, 2000, 200)
    y = np.ones_like(x)

    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="ramanspy",
    )

    with pytest.raises(NotImplementedError) as exc:
        rf_mod.RamanFit(
            spectra=y,
            wavenumber=x,
            custom_peaks={"Si": ([490.0, 1.0, 0.01], [560.0, 50.0, 1e6])},
            preprocessing=pipe,
        )

    msg = str(exc.value)
    assert "RamanSPy translation is currently implemented for" in msg or "not installed" in msg
