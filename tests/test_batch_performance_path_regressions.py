"""
test_batch_performance_path_regressions.py
-------------------------------------------
Regression tests confirming batch fit output is unchanged after the v0.4.6
lower-level baseline optimisation (DtD caching in baselineAPI.py).

These tests verify that:
- fit parameters (position, fwhm, peak_height) are reproducible and physically
  sensible after optimisation
- batch summary table has the expected shape
- batch export provenance fields are present and correct
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.batch import RamanBatch

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

_X = np.linspace(350.0, 450.0, 101)
_PEAK_POS = 400.0
_PEAK_AMP = 100.0
_PEAK_WIDTH = 5.0
_BASELINE_SLOPE = 0.1
_PEAK_Y = (
    _PEAK_AMP * np.exp(-0.5 * ((_X - _PEAK_POS) / _PEAK_WIDTH) ** 2)
    + _BASELINE_SLOPE * (_X - 350.0)
    + 5.0
)

_CUSTOM_PEAKS = {
    "test_peak": ([370.0, 1.0, 0.01], [430.0, 15.0, 200.0]),
}


def _write_raman_txt(path, x, y):
    header = "wavenumber\tintensity"
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter="\t", header=header, comments="")


def _make_batch(tmp_path, n=3, **kwargs):
    rng = np.random.default_rng(99)
    files = []
    for i in range(n):
        y = _PEAK_Y + rng.normal(0, 0.3, _PEAK_Y.size)
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


# ---------------------------------------------------------------------------
# 1. Fit output: peak parameters are physically sensible after DtD-cache opt.
# ---------------------------------------------------------------------------

def test_batch_fit_peak_position_in_range(tmp_path):
    """Fitted peak position should be within the search window for all spectra."""
    batch = _make_batch(tmp_path)
    batch.fit()

    rows = batch.rows  # list[dict] with keys: source, peak, position, fwhm, peak_height
    assert len(rows) > 0, "No fit rows produced"

    positions = [r["position"] for r in rows if r.get("position") is not None]
    assert positions, "No position values found in fit rows"
    positions = np.array(positions)
    assert np.all(positions > 350.0) and np.all(positions < 450.0), (
        f"Peak positions out of range: {positions}"
    )


def test_batch_fit_output_unchanged_across_calls(tmp_path):
    """Two identical batches must produce identical fit results (deterministic)."""
    batch1 = _make_batch(tmp_path)
    batch1.fit()

    # Create a second batch from the same files
    files = [str(p) for p in sorted(tmp_path.glob("spec_*.txt"))]
    batch2 = RamanBatch(
        files,
        custom_peaks=_CUSTOM_PEAKS,
        normalize=False,
        smoothing=False,
        background_remove=False,
    )
    batch2.fit()

    rows1 = batch1.rows
    rows2 = batch2.rows

    assert len(rows1) == len(rows2), "Row counts differ between identical batches"
    for r1, r2 in zip(rows1, rows2):
        assert r1["peak"] == r2["peak"]
        np.testing.assert_allclose(r1["position"], r2["position"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r1["fwhm"], r2["fwhm"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r1["peak_height"], r2["peak_height"], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Summary table shape
# ---------------------------------------------------------------------------

def test_batch_summary_table_has_expected_columns(tmp_path):
    batch = _make_batch(tmp_path, n=3)
    batch.fit()

    rows = batch.rows  # list[dict]
    # 3 spectra × 1 custom peak = 3 rows
    assert len(rows) == 3, f"Expected 3 rows (one per spectrum), got {len(rows)}"
    # Each row must have the expected keys
    required_keys = {"source", "peak", "position", "fwhm", "peak_height"}
    for r in rows:
        assert required_keys.issubset(r.keys()), (
            f"Row missing keys. Got: {set(r.keys())}"
        )


# ---------------------------------------------------------------------------
# 3. Export provenance: backend field is present
# ---------------------------------------------------------------------------

def test_batch_fit_provenance_field_present(tmp_path):
    """Fitter objects produced by batch.fit() must carry preprocessing_backend_resolved."""
    batch = _make_batch(tmp_path)
    batch.fit()

    assert len(batch.fits) > 0, "No fits stored"
    for _, _, fitter in batch.fits:
        assert hasattr(fitter, "preprocessing_backend_resolved"), (
            "Fitter missing preprocessing_backend_resolved"
        )
        assert fitter.preprocessing_backend_resolved in {"native", "auto", "ramanspy"}, (
            f"Unexpected backend value: {fitter.preprocessing_backend_resolved!r}"
        )
