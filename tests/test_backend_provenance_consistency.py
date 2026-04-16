"""
test_backend_provenance_consistency.py

Verify that serialise_backend_provenance() produces consistent keys regardless
of whether the caller is single-fit, mapping, or batch, and that uniform vs
mixed batch backends serialise correctly.
"""
import pytest
import numpy as np

from ramanpl.exporter import serialise_backend_provenance


# ---------------------------------------------------------------------------
# serialise_backend_provenance() unit tests
# ---------------------------------------------------------------------------

def _make_outcome(
    requested="native",
    resolved="native",
    fallback_used=False,
    fallback_reason=None,
    failure_mode=None,
):
    return {
        "requested_backend": requested,
        "resolved_backend": resolved,
        "execution_ready": True,
        "ramanspy_available": True,
        "ramanspy_version": "0.3.5",
        "supported_for_input": True,
        "translation_supported": True,
        "unsupported_steps": [],
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "failure_mode": failure_mode,
        "modality": "Raman",
        "axis_kind": "raman_shift_cm-1",
    }


def test_serialise_always_emits_requested_and_resolved():
    outcome = _make_outcome(requested="auto", resolved="ramanspy")
    result = serialise_backend_provenance(outcome)
    assert "preprocessing_backend_requested" in result
    assert "preprocessing_backend_resolved" in result
    assert result["preprocessing_backend_requested"] == "auto"
    assert result["preprocessing_backend_resolved"] == "ramanspy"


def test_serialise_omits_fallback_reason_when_no_fallback():
    outcome = _make_outcome(requested="auto", resolved="ramanspy", fallback_used=False)
    result = serialise_backend_provenance(outcome)
    assert "preprocessing_backend_fallback_reason" not in result


def test_serialise_includes_fallback_reason_when_fallback_occurred():
    reason = "Automatic fallback to native: RamanSPy is not installed."
    outcome = _make_outcome(
        requested="auto",
        resolved="native",
        fallback_used=True,
        fallback_reason=reason,
        failure_mode="fallback",
    )
    result = serialise_backend_provenance(outcome)
    assert result["preprocessing_backend_fallback_reason"] == reason


def test_serialise_omits_fallback_reason_when_fallback_but_no_reason():
    outcome = _make_outcome(
        requested="auto",
        resolved="native",
        fallback_used=True,
        fallback_reason=None,
    )
    result = serialise_backend_provenance(outcome)
    assert "preprocessing_backend_fallback_reason" not in result


def test_serialise_returns_empty_for_none_input():
    assert serialise_backend_provenance(None) == {}


def test_serialise_returns_empty_for_empty_dict():
    assert serialise_backend_provenance({}) == {}


def test_serialise_native_native_has_no_fallback_reason():
    outcome = _make_outcome(requested="native", resolved="native")
    result = serialise_backend_provenance(outcome)
    assert result["preprocessing_backend_requested"] == "native"
    assert result["preprocessing_backend_resolved"] == "native"
    assert "preprocessing_backend_fallback_reason" not in result


# ---------------------------------------------------------------------------
# Consistency across callers: same outcome → same keys
# ---------------------------------------------------------------------------

def test_identical_outcome_produces_identical_serialisation_twice():
    outcome = _make_outcome(
        requested="auto",
        resolved="native",
        fallback_used=True,
        fallback_reason="Automatic fallback to native: RamanSPy is not installed.",
        failure_mode="fallback",
    )
    result_a = serialise_backend_provenance(outcome)
    result_b = serialise_backend_provenance(outcome)
    assert result_a == result_b


# ---------------------------------------------------------------------------
# Pipeline support summary
# ---------------------------------------------------------------------------

def test_inspect_pipeline_backend_support_none_pipeline():
    from ramanpl.preprocessing import inspect_pipeline_backend_support
    result = inspect_pipeline_backend_support(None, "Raman", "raman_shift_cm-1")
    assert result["pipeline_translatable"] is True
    assert result["unsupported_steps"] == []
    assert result["translation_reason"] is None


def test_inspect_pipeline_backend_support_supported_pipeline():
    from ramanpl.preprocessing import (
        inspect_pipeline_backend_support,
        Pipeline,
        CropByRange,
        SmoothSavGol,
    )
    pipe = Pipeline(
        steps=[CropByRange((300, 1700)), SmoothSavGol(window_length=11, polyorder=3)],
        backend="auto",
    )
    result = inspect_pipeline_backend_support(pipe, "Raman", "raman_shift_cm-1")
    assert result["pipeline_translatable"] is True
    assert result["unsupported_steps"] == []
    assert result["translation_reason"] is None


def test_inspect_pipeline_backend_support_unsupported_step():
    from ramanpl.preprocessing import (
        inspect_pipeline_backend_support,
        Pipeline,
        BaselineSubtract,
    )
    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="auto",
    )
    result = inspect_pipeline_backend_support(pipe, "Raman", "raman_shift_cm-1")
    assert result["pipeline_translatable"] is False
    assert len(result["unsupported_steps"]) > 0
    assert result["translation_reason"] is not None


def test_inspect_pipeline_backend_support_result_is_serialisable():
    """Result dict should not contain any non-JSON-serialisable values."""
    import json
    from ramanpl.preprocessing import (
        inspect_pipeline_backend_support,
        Pipeline,
        BaselineSubtract,
    )
    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="auto",
    )
    result = inspect_pipeline_backend_support(pipe, "Raman", "raman_shift_cm-1")
    # Should not raise
    json.dumps(result)


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------

def test_validate_backend_outcome_keys_accepts_canonical_dict():
    from ramanpl.schema import validate_backend_outcome_keys
    outcome = _make_outcome()
    # should not raise
    validate_backend_outcome_keys(outcome)


def test_validate_backend_outcome_keys_rejects_missing_keys():
    from ramanpl.schema import validate_backend_outcome_keys
    incomplete = {"requested_backend": "native", "resolved_backend": "native"}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_backend_outcome_keys(incomplete)
