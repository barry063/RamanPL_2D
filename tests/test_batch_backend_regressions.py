"""
Regression tests for RamanBatch backend propagation and behaviour preservation (v0.4.4).

Backend resolution uses the same monkeypatching approach as test_single_fit_regressions.py
— the bridge and translation-support functions are patched at the preprocessing module level
so the batch path exercises exactly the same policy logic as single-spectrum fitting.
"""

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ramanpl.preprocessing as pre
from ramanpl.batch import RamanBatch


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

_X = np.linspace(350.0, 430.0, 81)
_PEAK_Y = 10.0 + 80.0 * np.exp(-0.5 * ((_X - 385.0) / 4.0) ** 2)

_CUSTOM_PEAKS = {
    "test_peak": ([370.0, 1.0, 0.01], [400.0, 15.0, 300.0]),
}


def _write_raman_txt(path, x, y):
    """Write a minimal 2-column tab-delimited txt file (1 header row)."""
    header = "wavenumber\tintensity"
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter="\t", header=header, comments="")


def _make_batch(tmp_path, n=2, **kwargs):
    """
    Create n synthetic Raman txt files and return an unfitted RamanBatch.
    Adds small per-spectrum noise so spectra are distinct.
    """
    rng = np.random.default_rng(42)
    files = []
    for i in range(n):
        y = _PEAK_Y + rng.normal(0, 0.5, _PEAK_Y.size)
        p = tmp_path / f"spec_{i}.txt"
        _write_raman_txt(str(p), _X, y)
        files.append(str(p))

    return RamanBatch(
        files,
        custom_peaks=_CUSTOM_PEAKS,
        normalize=False,
        smoothing=False,
        background_remove=False,
        **kwargs,
    )


def _mock_bridge_result(requested_backend, modality, axis_kind, *, ramanspy_available=True,
                        supported_for_input=True, resolved_backend=None, reason=None):
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


# ---------------------------------------------------------------------------
# 1. Backend resolution: auto + supported pipeline → ramanspy
# ---------------------------------------------------------------------------

def test_raman_batch_auto_supported_pipeline_resolves_ramanspy(tmp_path, monkeypatch):
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

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    batch = _make_batch(tmp_path, n=2, preprocessing_backend="auto")
    batch.fit()

    fitters = [f for _, _, f in batch.fits]
    for fitter in fitters:
        assert fitter.preprocessing_backend_resolved == "ramanspy", (
            f"Expected 'ramanspy', got '{fitter.preprocessing_backend_resolved}'"
        )
        assert fitter.preprocessing_backend_info["requested_backend"] == "auto"
        assert fitter.preprocessing_backend_info["resolved_backend"] == "ramanspy"


# ---------------------------------------------------------------------------
# 2. Backend resolution: auto + unsupported translation → stays native
# ---------------------------------------------------------------------------

def test_raman_batch_auto_gaussian_baseline_stays_native(tmp_path, monkeypatch):
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
        # Simulate: Gaussian baseline step is not translatable
        return False, ["gaussian_baseline"]

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    batch = _make_batch(tmp_path, n=2, preprocessing_backend="auto")
    batch.fit()

    fitters = [f for _, _, f in batch.fits]
    for fitter in fitters:
        assert fitter.preprocessing_backend_resolved == "native", (
            f"Expected 'native' fallback, got '{fitter.preprocessing_backend_resolved}'"
        )


# ---------------------------------------------------------------------------
# 3. Backend resolution: forced ramanspy + untranslatable pipeline → raises
# ---------------------------------------------------------------------------

def test_raman_batch_forced_ramanspy_untranslatable_pipeline_raises(tmp_path, monkeypatch):
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
        return False, ["gaussian_baseline"]

    monkeypatch.setattr(pre, "resolve_preprocessing_backend", fake_bridge)
    monkeypatch.setattr(pre, "pipeline_supports_ramanspy_translation", fake_translation_support)

    batch = _make_batch(tmp_path, n=1, preprocessing_backend="ramanspy")

    with pytest.raises(NotImplementedError):
        batch.fit()


# ---------------------------------------------------------------------------
# 4. Behaviour preservation: plot_overlay / plot_waterfall return (fig, ax)
#    and source ordering matches file order
# ---------------------------------------------------------------------------

def test_raman_batch_plot_overlay_and_waterfall_behaviour_preserved(tmp_path):
    matplotlib.use("Agg")

    batch = _make_batch(tmp_path, n=3)
    batch.fit()

    # overlay — raw only
    result_overlay = batch.plot_overlay()
    assert isinstance(result_overlay, tuple) and len(result_overlay) == 2
    fig_ov, ax_ov = result_overlay
    assert isinstance(fig_ov, plt.Figure)
    assert isinstance(ax_ov, plt.Axes)
    plt.close(fig_ov)

    # overlay — raw + fit
    result_rawfit = batch.plot_overlay(raw_fit=True)
    fig_rf, ax_rf = result_rawfit
    assert isinstance(fig_rf, plt.Figure)
    plt.close(fig_rf)

    # waterfall
    result_wf = batch.plot_waterfall()
    fig_wf, ax_wf = result_wf
    assert isinstance(fig_wf, plt.Figure)
    plt.close(fig_wf)

    # source ordering: loaded specs must match files order
    sources = [s.source for s in batch.specs]
    assert sources == list(batch.files), "Source order does not match file order"


# ---------------------------------------------------------------------------
# 5. Behaviour preservation: long-format DataFrame has exact expected columns
# ---------------------------------------------------------------------------

def test_raman_batch_long_table_schema_preserved(tmp_path):
    batch = _make_batch(tmp_path, n=2)
    batch.fit()

    df = batch.to_dataframe()
    expected_cols = {"source", "peak", "position", "fwhm", "peak_height"}
    assert set(df.columns) == expected_cols, (
        f"Long table columns changed. Got: {set(df.columns)}, expected: {expected_cols}"
    )
    assert len(df) > 0


# ---------------------------------------------------------------------------
# 6. Behaviour preservation: summary_by_peak() returns DataFrame with
#    per-peak count / mean / std columns for all default metrics
# ---------------------------------------------------------------------------

def test_raman_batch_summary_by_peak_schema_preserved(tmp_path):
    batch = _make_batch(tmp_path, n=3)
    batch.fit()

    stats_df = batch.summary_by_peak(print_summary=False)

    assert "peak" in stats_df.columns, "summary_by_peak must have a 'peak' column"
    assert "test_peak" in stats_df["peak"].values

    for metric in ("position", "fwhm", "peak_height"):
        for stat in ("count", "mean", "std"):
            col = f"{metric}_{stat}"
            assert col in stats_df.columns, (
                f"summary_by_peak missing column '{col}'"
            )
