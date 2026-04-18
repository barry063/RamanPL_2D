"""Tests for backend resolution contract in ramanspy_bridge.py."""
import pytest

from ramanpl.schema import BACKEND_OUTCOME_REQUIRED_KEYS


def _resolve(requested, modality="Raman", axis_kind="raman_shift_cm-1"):
    from ramanpl.integration.ramanspy_bridge import resolve_backend_outcome
    return resolve_backend_outcome(
        requested_backend=requested,
        modality=modality,
        axis_kind=axis_kind,
    )


# ---------------------------------------------------------------------------
# All backends return a dict with all required keys
# ---------------------------------------------------------------------------

def test_native_backend_has_all_required_keys():
    outcome = _resolve("native")
    assert BACKEND_OUTCOME_REQUIRED_KEYS.issubset(outcome.keys())


def test_auto_backend_has_all_required_keys(monkeypatch):
    monkeypatch.setattr(
        "ramanpl.integration.ramanspy_bridge.has_ramanspy",
        lambda: False,
    )
    outcome = _resolve("auto")
    assert BACKEND_OUTCOME_REQUIRED_KEYS.issubset(outcome.keys())


def test_ramanspy_backend_has_all_required_keys_when_unavailable():
    from ramanpl.integration.ramanspy_bridge import resolve_backend_outcome
    from ramanpl.integration.ramanspy_adapter import has_ramanspy
    if has_ramanspy():
        pytest.skip("RamanSPy is available; test targets unavailable path")
    with pytest.raises((NotImplementedError, RuntimeError)):
        _resolve("ramanspy")


# ---------------------------------------------------------------------------
# native backend behaviour
# ---------------------------------------------------------------------------

def test_native_resolves_to_native():
    outcome = _resolve("native")
    assert outcome["resolved_backend"] == "native"
    assert outcome["execution_ready"] is True
    assert outcome["fallback_used"] is False


def test_native_modality_and_axis_passthrough():
    outcome = _resolve("native", modality="Photoluminescence", axis_kind="energy_eV")
    assert outcome["modality"] in ("Photoluminescence", "PL")
    assert outcome["axis_kind"] == "energy_eV"


# ---------------------------------------------------------------------------
# auto backend behaviour
# ---------------------------------------------------------------------------

def test_auto_falls_back_to_native_when_ramanspy_unavailable(monkeypatch):
    monkeypatch.setattr(
        "ramanpl.integration.ramanspy_bridge.has_ramanspy",
        lambda: False,
    )
    outcome = _resolve("auto")
    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is True
    assert outcome["fallback_reason"] is not None


def test_auto_falls_back_to_native_for_pl_modality(monkeypatch):
    monkeypatch.setattr(
        "ramanpl.integration.ramanspy_bridge.has_ramanspy",
        lambda: True,
    )
    outcome = _resolve("auto", modality="Photoluminescence", axis_kind="energy_eV")
    assert outcome["resolved_backend"] == "native"
    assert outcome["fallback_used"] is True


# ---------------------------------------------------------------------------
# Field value contracts
# ---------------------------------------------------------------------------

def test_native_never_sets_fallback_reason():
    outcome = _resolve("native")
    assert outcome["fallback_reason"] is None


def test_native_execution_ready_is_true():
    outcome = _resolve("native")
    assert outcome["execution_ready"] is True
