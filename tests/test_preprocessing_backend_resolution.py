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
    assert info["fallback_used"] is True
    assert "not yet implemented" in info["fallback_reason"]


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


# ---------------------------------------------------------------------------
# New v0.4.8 dispatch tests — single-spectrum RamanSPy resolution
# ---------------------------------------------------------------------------

def _make_raman_ds():
    x = np.linspace(100, 200, 11)
    y = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
    return SpectralDataset(
        x=x, y=y,
        modality="Raman", axis_kind="raman_shift_cm-1",
        meta={"x_trimmed_on_load": False},
    )


def test_pipeline_apply_single_raman_supported_auto_resolves_to_ramanspy(monkeypatch):
    """auto on a fully-supported Raman single spectrum dispatches to RamanSPy."""
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
        return True, []

    call_log = []

    def fake_ramanspy_single(ds, pipeline):
        call_log.append("called")
        return ds.x.copy(), ds.y.copy(), {}

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)
    monkeypatch.setattr(pre, "apply_pipeline_ramanspy_single", fake_ramanspy_single)

    ds = _make_raman_ds()
    pipe = Pipeline(steps=[CropByRange((110, 190))], backend="auto")
    out = pipe.apply(ds)

    assert len(call_log) == 1, "apply_pipeline_ramanspy_single should be called exactly once"
    assert out.meta["preprocessing_backend"] == "ramanspy"


def test_pipeline_apply_single_pl_auto_resolves_to_native(monkeypatch):
    """auto on a PL spectrum stays native because PL is not supported_for_input."""
    def fake_bridge(**kwargs):
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=False,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    x = np.linspace(600, 800, 11)
    y = np.ones(11)
    ds = SpectralDataset(
        x=x, y=y,
        modality="PL", axis_kind="wavelength_nm",
        meta={"x_trimmed_on_load": False},
    )
    pipe = Pipeline(steps=[SmoothSavGol(window_length=5, polyorder=2)], backend="auto")
    out = pipe.apply(ds)

    assert out.meta["preprocessing_backend"] == "native"
    assert out.meta["preprocessing_backend_info"]["resolved_backend"] == "native"


def test_pipeline_apply_single_raman_gaussian_auto_falls_back_to_native(monkeypatch):
    """auto falls back to native when the pipeline contains a non-translatable step."""
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

    ds = _make_raman_ds()
    pipe = Pipeline(
        steps=[BaselineSubtract({"method": "gaussian", "gaussian_sigma": 5})],
        backend="auto",
    )
    out = pipe.apply(ds)

    info = out.meta["preprocessing_backend_info"]
    assert info["resolved_backend"] == "native"
    assert info["fallback_used"] is True
    assert info["fallback_reason"] is not None


def test_pipeline_apply_single_forced_ramanspy_calls_ramanspy_dispatch(monkeypatch):
    """Forced ramanspy dispatches to apply_pipeline_ramanspy_single exactly once."""
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
        return True, []

    call_log = []

    def fake_ramanspy_single(ds, pipeline):
        call_log.append("called")
        return ds.x.copy(), ds.y.copy(), {}

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)
    monkeypatch.setattr(pre, "apply_pipeline_ramanspy_single", fake_ramanspy_single)

    ds = _make_raman_ds()
    pipe = Pipeline(steps=[SmoothSavGol(window_length=5, polyorder=2)], backend="ramanspy")
    pipe.apply(ds)

    assert len(call_log) == 1, "apply_pipeline_ramanspy_single should be called exactly once"