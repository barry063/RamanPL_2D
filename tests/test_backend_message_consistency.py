"""
test_backend_message_consistency.py

Verify that fallback/error wording from resolve_backend_outcome() is stable and
consistent, independent of whether it is reached via single-fit, mapping, or batch.
These tests operate at the resolve_backend_outcome() level directly.
"""
import pytest

from ramanpl.integration.ramanspy_bridge import (
    resolve_backend_outcome,
    format_backend_fallback_reason,
    format_backend_error_message,
)


# ---------------------------------------------------------------------------
# auto fallback cases
# ---------------------------------------------------------------------------

def test_auto_fallback_reason_when_ramanspy_unavailable(monkeypatch):
    """auto + RamanSPy unavailable → native fallback with clear 'not installed' reason."""
    from ramanpl.integration import ramanspy_bridge
    from ramanpl.integration.ramanspy_adapter import can_use_ramanspy, get_ramanspy_version

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: False)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: None)
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (False, "RamanSPy not installed"))

    outcome = resolve_backend_outcome(
        requested_backend="auto",
        modality="Raman",
        axis_kind="raman_shift_cm-1",
        translation_supported=True,
        unsupported_steps=[],
    )

    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is True
    assert outcome["failure_mode"] == "fallback"
    assert outcome["fallback_reason"] is not None
    assert "not installed" in outcome["fallback_reason"].lower() or "unavailable" in outcome["fallback_reason"].lower() or "native" in outcome["fallback_reason"].lower()
    assert format_backend_fallback_reason(outcome) == outcome["fallback_reason"]


def test_auto_fallback_reason_when_pipeline_not_translatable(monkeypatch):
    """auto + untranslatable pipeline → native fallback with 'not yet implemented' reason."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: True)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: "0.3.5")
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (True, None))

    outcome = resolve_backend_outcome(
        requested_backend="auto",
        modality="Raman",
        axis_kind="raman_shift_cm-1",
        translation_supported=False,
        unsupported_steps=["BaselineSubtract(method='gaussian')"],
    )

    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is True
    assert outcome["failure_mode"] == "fallback"
    assert "not yet implemented" in outcome["fallback_reason"]
    assert "gaussian" in outcome["fallback_reason"].lower()


def test_auto_no_fallback_when_fully_supported(monkeypatch):
    """auto + fully supported → ramanspy, no fallback."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: True)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: "0.3.5")
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (True, None))

    outcome = resolve_backend_outcome(
        requested_backend="auto",
        modality="Raman",
        axis_kind="raman_shift_cm-1",
        translation_supported=True,
        unsupported_steps=[],
    )

    assert outcome["resolved_backend"] == "ramanspy"
    assert outcome["fallback_used"] is False
    assert outcome["fallback_reason"] is None
    assert outcome["failure_mode"] is None


def test_native_backend_never_has_fallback(monkeypatch):
    """native backend → resolved native, no fallback metadata."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: True)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: "0.3.5")
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (True, None))

    outcome = resolve_backend_outcome(
        requested_backend="native",
        modality="Raman",
        axis_kind="raman_shift_cm-1",
        translation_supported=False,
        unsupported_steps=["some step"],
    )

    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is False
    assert outcome["fallback_reason"] is None
    assert outcome["failure_mode"] is None


# ---------------------------------------------------------------------------
# forced ramanspy error cases
# ---------------------------------------------------------------------------

def test_forced_ramanspy_raises_when_unavailable(monkeypatch):
    """forced ramanspy + unavailable → NotImplementedError with consistent shape."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: False)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: None)
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (False, "not installed"))

    with pytest.raises(NotImplementedError) as exc:
        resolve_backend_outcome(
            requested_backend="ramanspy",
            modality="Raman",
            axis_kind="raman_shift_cm-1",
            translation_supported=True,
        )

    msg = str(exc.value).lower()
    assert "not installed" in msg


def test_forced_ramanspy_raises_for_unsupported_input(monkeypatch):
    """forced ramanspy + unsupported axis → NotImplementedError."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: True)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: "0.3.5")
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (False, "PL is not supported by RamanSPy backend"))

    with pytest.raises(NotImplementedError) as exc:
        resolve_backend_outcome(
            requested_backend="ramanspy",
            modality="PL",
            axis_kind="energy_eV",
            translation_supported=True,
        )

    msg = str(exc.value)
    assert len(msg) > 0


def test_forced_ramanspy_raises_for_untranslatable_pipeline(monkeypatch):
    """forced ramanspy + untranslatable pipeline → NotImplementedError with step names."""
    from ramanpl.integration import ramanspy_bridge

    monkeypatch.setattr(ramanspy_bridge, "has_ramanspy", lambda: True)
    monkeypatch.setattr(ramanspy_bridge, "get_ramanspy_version", lambda: "0.3.5")
    monkeypatch.setattr(ramanspy_bridge, "can_use_ramanspy", lambda modality, axis_kind: (True, None))

    with pytest.raises(NotImplementedError) as exc:
        resolve_backend_outcome(
            requested_backend="ramanspy",
            modality="Raman",
            axis_kind="raman_shift_cm-1",
            translation_supported=False,
            unsupported_steps=["BaselineSubtract(method='gaussian')"],
        )

    msg = str(exc.value)
    assert "RamanSPy translation is currently implemented for" in msg
    assert "Unsupported steps" in msg


# ---------------------------------------------------------------------------
# format_backend_error_message helper
# ---------------------------------------------------------------------------

def test_format_backend_error_message_for_missing_installation():
    outcome = {
        "requested_backend": "ramanspy",
        "ramanspy_available": False,
        "supported_for_input": True,
        "translation_supported": True,
        "unsupported_steps": [],
        "reason": None,
    }
    msg = format_backend_error_message(outcome)
    assert "not installed" in msg.lower()
    assert "ramanspy" in msg.lower()


def test_format_backend_error_message_for_unsupported_input():
    outcome = {
        "requested_backend": "ramanspy",
        "ramanspy_available": True,
        "supported_for_input": False,
        "translation_supported": True,
        "unsupported_steps": [],
        "reason": "PL is not supported by RamanSPy backend",
    }
    msg = format_backend_error_message(outcome)
    assert "ramanspy" in msg.lower()


def test_format_backend_error_message_for_untranslatable_pipeline():
    outcome = {
        "requested_backend": "ramanspy",
        "ramanspy_available": True,
        "supported_for_input": True,
        "translation_supported": False,
        "unsupported_steps": ["BaselineSubtract(method='gaussian')"],
        "reason": None,
    }
    msg = format_backend_error_message(outcome)
    assert "translation" in msg.lower()
    assert "gaussian" in msg.lower()
