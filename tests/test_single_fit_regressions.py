import numpy as np
import pytest

import ramanpl.preprocessing as pre
import ramanpl.integration.ramanspy_bridge as bridge_mod
import ramanpl.single_fit._single_fit_core as core
from ramanpl.single_fit._single_fit_core import (
    generate_p0_trials,
)
from ramanpl.single_fit.RamanFit import RamanFit
from ramanpl.single_fit.PLfit import PLfit


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


def test_generate_p0_trials_midpoint_is_centred_on_bound_midpoint():
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    base_p0 = np.array([2.0, 2.0])

    trials = generate_p0_trials(
        lb,
        ub,
        base_p0=base_p0,
        n_starts=2000,
        strategy="midpoint",
        random_state=123,
    )

    assert np.allclose(trials[0], base_p0)

    extra = np.vstack(trials[1:])
    mean_vec = extra.mean(axis=0)

    # midpoint anchor should be near (lb + ub) / 2 = [5, 5]
    assert np.allclose(mean_vec, np.array([5.0, 5.0]), atol=0.15)

    # all generated trials must remain within bounds
    assert np.all(extra >= lb)
    assert np.all(extra <= ub)


def test_generate_p0_trials_jitter_is_centred_on_user_base_p0():
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    base_p0 = np.array([2.0, 2.0])

    trials = generate_p0_trials(
        lb,
        ub,
        base_p0=base_p0,
        n_starts=2000,
        strategy="jitter",
        random_state=123,
    )

    assert np.allclose(trials[0], base_p0)

    extra = np.vstack(trials[1:])
    mean_vec = extra.mean(axis=0)

    # jitter anchor should remain near the user-supplied base_p0 = [2, 2]
    assert np.allclose(mean_vec, np.array([2.0, 2.0]), atol=0.15)

    # all generated trials must remain within bounds
    assert np.all(extra >= lb)
    assert np.all(extra <= ub)



def test_ramanfit_auto_promotes_to_ramanspy_when_supported(monkeypatch):
    pytest.importorskip("ramanspy")
    def fake_bridge(**kwargs):
        supported = (
            kwargs["modality"] == "Raman"
            and kwargs["axis_kind"] == "raman_shift_cm-1"
        )
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=supported,
            resolved_backend="native",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(bridge_mod, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    x = np.linspace(350.0, 430.0, 201)
    y = 10.0 + 50.0 * np.exp(-0.5 * ((x - 385.0) / 4.0) ** 2)

    custom_peaks = {
        "test_peak": ([370.0, 1.0, 0.01], [400.0, 10.0, 100.0]),
    }

    obj = RamanFit(
        spectra=y,
        wavenumber=x,
        custom_peaks=custom_peaks,
        normalize=False,
        smoothing=False,
        background_remove=False,
        preprocessing_backend="auto",
    )

    assert obj.preprocessing_backend_resolved == "ramanspy"
    assert obj.preprocessing_backend_info["requested_backend"] == "auto"
    assert obj.preprocessing_backend_info["resolved_backend"] == "ramanspy"


def test_plfit_auto_falls_back_to_native_when_input_is_not_supported(monkeypatch):
    def fake_bridge(**kwargs):
        supported = (
            kwargs["modality"] == "Raman"
            and kwargs["axis_kind"] == "raman_shift_cm-1"
        )
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=supported,
            resolved_backend="native",
            reason=None if supported else "RamanSPy backend currently supports Raman/c m^-1 only.",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    x = np.linspace(1.75, 2.05, 201)
    y = 5.0 + 30.0 * np.exp(-0.5 * ((x - 1.92) / 0.02) ** 2)

    obj = PLfit(
        spectra=y,
        energy=x,
        normalize=True,
        smoothing=False,
        background_remove=False,
        preprocessing_backend="auto",
    )

    assert obj.preprocessing_backend_resolved == "native"
    assert obj.preprocessing_backend_info["requested_backend"] == "auto"
    assert obj.preprocessing_backend_info["resolved_backend"] == "native"
    assert obj.preprocessing_backend_info["supported_for_input"] is False


def test_plfit_forced_ramanspy_raises_when_input_is_not_supported(monkeypatch):
    def fake_bridge(**kwargs):
        supported = (
            kwargs["modality"] == "Raman"
            and kwargs["axis_kind"] == "raman_shift_cm-1"
        )
        return _mock_bridge_result(
            kwargs["requested_backend"],
            kwargs["modality"],
            kwargs["axis_kind"],
            ramanspy_available=True,
            supported_for_input=supported,
            resolved_backend="ramanspy" if supported else kwargs["requested_backend"],
            reason=None if supported else "RamanSPy backend currently supports Raman/c m^-1 only.",
        )

    def fake_translation_support(pipeline):
        return True, []

    monkeypatch.setattr(bridge_mod, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    x = np.linspace(1.75, 2.05, 201)
    y = 5.0 + 30.0 * np.exp(-0.5 * ((x - 1.92) / 0.02) ** 2)

    with pytest.raises(NotImplementedError, match="RamanSPy backend currently supports Raman"):
        PLfit(
            spectra=y,
            energy=x,
            normalize=True,
            smoothing=False,
            background_remove=False,
            preprocessing_backend="ramanspy",
        )