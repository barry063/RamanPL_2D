import numpy as np
import pytest

import ramanpl.preprocessing as pre
from ramanpl.preprocessing import (
    SpectralDataset,
    Pipeline,
    CropByRange,
    SmoothSavGol,
    BaselineSubtract,
)


def _mock_bridge_result(
    requested_backend,
    modality,
    axis_kind,
    *,
    ramanspy_available=True,
    supported_for_input=True,
    resolved_backend=None,
    reason=None,
):
    """
    Minimal mock payload matching the contract used by
    _resolve_pipeline_backend_for_input().
    """
    return {
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend or requested_backend,
        "modality": modality,
        "axis_kind": axis_kind,
        "ramanspy_available": ramanspy_available,
        "ramanspy_version": "mock",
        "supported_for_input": supported_for_input,
        "execution_ready": True,
        "reason": reason,
    }


def test_native_backend_always_stays_native(monkeypatch):
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=True,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return False, ["BaselineSubtract(method='gaussian')"]

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="native",
    )

    info = pre._resolve_pipeline_backend_for_input(
        pipe,
        modality="Raman",
        axis_kind="raman_shift_cm-1",
    )

    assert info["requested_backend"] == "native"
    assert info["resolved_backend"] == "native"
    assert info["execution_ready"] is True
    assert info["reason"] is None
    assert info["translation_supported"] is False
    assert info["unsupported_steps"] == ["BaselineSubtract(method='gaussian')"]


def test_auto_promotes_to_ramanspy_only_when_fully_supported(monkeypatch):
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=True,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    pipe = Pipeline(
        steps=[
            CropByRange((100, 500)),
            SmoothSavGol(window_length=5, polyorder=2),
            BaselineSubtract({"method": "asls", "lam": 1e5, "p": 0.01}),
        ],
        backend="auto",
    )

    info = pre._resolve_pipeline_backend_for_input(
        pipe,
        modality="Raman",
        axis_kind="raman_shift_cm-1",
    )

    assert info["requested_backend"] == "auto"
    assert info["resolved_backend"] == "ramanspy"
    assert info["execution_ready"] is True
    assert info["reason"] is None
    assert info["translation_supported"] is True
    assert info["unsupported_steps"] == []


def test_auto_falls_back_to_native_when_translation_is_not_supported(monkeypatch):
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=True,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return False, ["BaselineSubtract(method='gaussian')"]

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="auto",
    )

    info = pre._resolve_pipeline_backend_for_input(
        pipe,
        modality="Raman",
        axis_kind="raman_shift_cm-1",
    )

    assert info["requested_backend"] == "auto"
    assert info["resolved_backend"] == "native"
    assert info["execution_ready"] is True
    assert info["translation_supported"] is False
    assert info["unsupported_steps"] == ["BaselineSubtract(method='gaussian')"]
    assert "not yet implemented" in info["reason"]


def test_forced_ramanspy_raises_for_unsupported_translation(monkeypatch):
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=True,
            resolved_backend="ramanspy",
        )

    def fake_translation_support(pipeline):
        return False, ["BaselineSubtract(method='gaussian')"]

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 10})],
        backend="ramanspy",
    )

    with pytest.raises(NotImplementedError) as exc:
        pre._resolve_pipeline_backend_for_input(
            pipe,
            modality="Raman",
            axis_kind="raman_shift_cm-1",
        )

    msg = str(exc.value)
    assert "RamanSPy translation is currently implemented for" in msg
    assert "Unsupported steps" in msg


def test_pipeline_apply_records_backend_provenance_for_native_path(monkeypatch):
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=False,
            supported_for_input=True,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    x = np.linspace(100, 200, 11)
    y = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)

    ds = SpectralDataset(
        x=x,
        y=y,
        modality="Raman",
        axis_kind="raman_shift_cm-1",
        meta={"x_trimmed_on_load": False},
    )

    pipe = Pipeline(
        steps=[
            CropByRange((110, 190)),
            SmoothSavGol(window_length=5, polyorder=2),
        ],
        backend="native",
    )

    out = pipe.apply(ds)

    assert out.meta["preprocessing_backend"] == "native"
    assert out.meta["preprocessing_backend_info"]["requested_backend"] == "native"
    assert out.meta["preprocessing_backend_info"]["resolved_backend"] == "native"
    assert out.meta["pipeline"]["backend"] == "native"

    # crop metadata should survive on the native path
    assert "crop" in out.meta
    assert out.meta["crop"]["applied"] is True

    # shape should shrink after crop
    assert out.x.size < ds.x.size
    assert out.y.size == out.x.size