"""Tests for provenance serialisation helpers in exporter.py."""
import pytest

from ramanpl.exporter import serialise_backend_provenance, serialize_preprocessing_provenance


# ---------------------------------------------------------------------------
# serialise_backend_provenance
# ---------------------------------------------------------------------------

def _native_outcome():
    return {
        "requested_backend": "native",
        "resolved_backend": "native",
        "execution_ready": True,
        "ramanspy_available": False,
        "supported_for_input": False,
        "translation_supported": True,
        "unsupported_steps": [],
        "fallback_used": False,
        "fallback_reason": None,
        "failure_mode": None,
        "modality": "Raman",
        "axis_kind": "raman_shift_cm-1",
    }


def _fallback_outcome():
    return {
        "requested_backend": "auto",
        "resolved_backend": "native",
        "execution_ready": True,
        "ramanspy_available": False,
        "supported_for_input": False,
        "translation_supported": True,
        "unsupported_steps": [],
        "fallback_used": True,
        "fallback_reason": "RamanSPy not installed",
        "failure_mode": None,
        "modality": "Raman",
        "axis_kind": "raman_shift_cm-1",
    }


def test_native_outcome_has_requested_and_resolved():
    prov = serialise_backend_provenance(_native_outcome())
    assert prov["preprocessing_backend_requested"] == "native"
    assert prov["preprocessing_backend_resolved"] == "native"


def test_native_outcome_has_no_fallback_reason_key():
    prov = serialise_backend_provenance(_native_outcome())
    assert "preprocessing_backend_fallback_reason" not in prov


def test_fallback_outcome_has_all_three_keys():
    prov = serialise_backend_provenance(_fallback_outcome())
    assert "preprocessing_backend_requested" in prov
    assert "preprocessing_backend_resolved" in prov
    assert "preprocessing_backend_fallback_reason" in prov
    assert prov["preprocessing_backend_fallback_reason"] == "RamanSPy not installed"


def test_none_outcome_returns_empty_dict():
    assert serialise_backend_provenance(None) == {}


def test_empty_dict_outcome_returns_empty_dict():
    assert serialise_backend_provenance({}) == {}


# ---------------------------------------------------------------------------
# serialize_preprocessing_provenance
# ---------------------------------------------------------------------------

def test_extracts_public_keys():
    meta = {
        "pipeline": [{"step": "smooth"}],
        "smoothing": {"window_length": 5},
        "baseline": {"method": "poly"},
        "crop": [100.0, 200.0],
        "_smoothed_last": True,
        "_baseline_last": True,
        "execution_ready": True,
    }
    prov = serialize_preprocessing_provenance(meta)
    assert "pipeline" in prov
    assert "smoothing" in prov
    assert "baseline" in prov
    assert "crop" in prov
    assert "_smoothed_last" not in prov
    assert "_baseline_last" not in prov
    assert "execution_ready" not in prov


def test_none_meta_returns_empty_dict():
    assert serialize_preprocessing_provenance(None) == {}


def test_empty_meta_returns_empty_dict():
    assert serialize_preprocessing_provenance({}) == {}


def test_partial_meta_only_returns_present_keys():
    meta = {"pipeline": [{"step": "crop", "range": [100.0, 200.0]}]}
    prov = serialize_preprocessing_provenance(meta)
    assert "pipeline" in prov
    assert "smoothing" not in prov
    assert "baseline" not in prov


# ---------------------------------------------------------------------------
# Key-set consistency across workflow types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested,resolved,fallback_used,fallback_reason", [
    ("native", "native", False, None),
    ("auto", "native", True, "RamanSPy not installed"),
    ("auto", "ramanspy", False, None),
])
def test_provenance_key_set_consistent(requested, resolved, fallback_used, fallback_reason):
    outcome = {
        "requested_backend": requested,
        "resolved_backend": resolved,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }
    prov = serialise_backend_provenance(outcome)
    assert "preprocessing_backend_requested" in prov
    assert "preprocessing_backend_resolved" in prov
    if fallback_used:
        assert "preprocessing_backend_fallback_reason" in prov
    else:
        assert "preprocessing_backend_fallback_reason" not in prov
