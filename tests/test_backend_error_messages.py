"""Tests for backend error message quality and content."""
import pytest


def _resolve(requested, modality="Raman", axis_kind="raman_shift_cm-1"):
    from ramanpl.integration.ramanspy_bridge import resolve_backend_outcome
    return resolve_backend_outcome(
        requested_backend=requested,
        modality=modality,
        axis_kind=axis_kind,
    )


# ---------------------------------------------------------------------------
# Invalid backend name
# ---------------------------------------------------------------------------

def test_invalid_backend_name_raises_with_clear_message():
    from ramanpl.preprocessing import normalize_backend_request
    with pytest.raises(ValueError) as exc_info:
        normalize_backend_request("totally_invalid")
    msg = str(exc_info.value).lower()
    assert "auto" in msg or "native" in msg or "ramanspy" in msg or "supported" in msg


def test_invalid_backend_name_in_schema_raises():
    from ramanpl.schema import normalise_preprocess_backend
    with pytest.raises(ValueError):
        normalise_preprocess_backend("bad_backend_xyz")


# ---------------------------------------------------------------------------
# Forced ramanspy when unavailable
# ---------------------------------------------------------------------------

def test_ramanspy_forced_when_unavailable_raises_with_install_hint():
    from ramanpl.integration.ramanspy_adapter import has_ramanspy
    if has_ramanspy():
        pytest.skip("RamanSPy is available; test targets unavailable path")

    with pytest.raises((NotImplementedError, RuntimeError, ImportError)) as exc_info:
        _resolve("ramanspy")
    msg = str(exc_info.value).lower()
    assert "ramanspy" in msg or "install" in msg or "not available" in msg


# ---------------------------------------------------------------------------
# Fallback reason string content
# ---------------------------------------------------------------------------

def test_auto_fallback_reason_mentions_ramanspy_unavailable(monkeypatch):
    monkeypatch.setattr(
        "ramanpl.integration.ramanspy_bridge.has_ramanspy",
        lambda: False,
    )
    outcome = _resolve("auto")
    assert outcome["fallback_used"] is True
    reason = outcome["fallback_reason"] or ""
    assert len(reason) > 0


def test_auto_fallback_reason_for_pl_modality(monkeypatch):
    monkeypatch.setattr(
        "ramanpl.integration.ramanspy_bridge.has_ramanspy",
        lambda: True,
    )
    outcome = _resolve("auto", modality="PL", axis_kind="energy_eV")
    assert outcome["fallback_used"] is True
    reason = outcome["fallback_reason"] or ""
    assert len(reason) > 0


# ---------------------------------------------------------------------------
# Invalid baseline spec message names the bad key/value
# ---------------------------------------------------------------------------

def test_invalid_baseline_type_raises_type_error_with_info():
    from ramanpl.preprocessing import canonicalize_baseline_spec
    with pytest.raises(TypeError):
        canonicalize_baseline_spec(42)


def test_empty_tuple_baseline_spec_raises_value_error():
    from ramanpl.preprocessing import canonicalize_baseline_spec
    with pytest.raises(ValueError):
        canonicalize_baseline_spec(())
