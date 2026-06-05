"""
Generate golden feature-table CSVs for v0.6.6 (pre-v0.6.7 snapshot).
Run once before any v0.6.7 edits. Output files are byte-parity references.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.mapping import RamanMapping
from ramanpl.mapping._pl_mapping import PLMapping

# ---------------------------------------------------------------------------
# Shared fixture parameters (must match test_api_stability.py fixtures)
# ---------------------------------------------------------------------------
N_PTS = 80
X_RAMAN = np.linspace(300.0, 700.0, N_PTS)
X_PL = np.linspace(1.85, 2.10, N_PTS)
Y_MAP, X_MAP = 3, 4

CUSTOM_PEAKS_RAMAN = {
    "E2g": ([380.0, 1.0, 0.001], [392.0, 20.0, 5.0]),
    "A1g": ([402.0, 1.0, 0.001], [412.0, 20.0, 5.0]),
}
CUSTOM_PEAKS_PL = {
    "Trion":   ([1.88, 0.005, 0.001], [1.96, 0.10, 5.0]),
    "Exciton": ([1.96, 0.005, 0.001], [2.05, 0.10, 5.0]),
}
GOOD_PARAMS_RAMAN = np.array([386.0, 3.0, 2.0, 407.0, 4.0, 3.0], dtype=float)
GOOD_PARAMS_PL    = np.array([1.91, 0.02, 1.5, 2.00, 0.03, 2.5], dtype=float)


def _inject_fit(m, good_params):
    n_params = len(good_params)
    m.fitted_params = np.full((m.Y, m.X, n_params), float("nan"), dtype=float)
    m.residual_map = np.full((m.Y, m.X), float("nan"), dtype=float)
    m.norm_scale_map = np.full((m.Y, m.X), 1.0, dtype=float)
    m.fit_diagnostics_map = np.empty((m.Y, m.X), dtype=object)
    m.fit_diagnostics_map[:, :] = None
    n_peaks = len(list(m.peak_params))
    m.peak_intensities = np.full((m.Y, m.X, n_peaks), float("nan"), dtype=float)
    for j in range(m.Y):
        for i in range(m.X):
            m.fitted_params[j, i, :] = good_params
            m.residual_map[j, i] = 0.04
            m.norm_scale_map[j, i] = 5.0
            m.fit_diagnostics_map[j, i] = {
                "ok": True, "n_starts": 1,
                "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
            }
            for k in range(n_peaks):
                hwhm = float(good_params[m.params_per_peak * k + 1])
                amp  = float(good_params[m.params_per_peak * k + 2])
                h_norm = amp / (np.pi * hwhm) if hwhm != 0 else float("nan")
                m.peak_intensities[j, i, k] = h_norm * m.norm_scale_map[j, i]


# ---------------------------------------------------------------------------
# 1. Raman mapping
# ---------------------------------------------------------------------------
rng = np.random.default_rng(10)
peak1 = np.exp(-0.5 * ((X_RAMAN - 386.0) / 3.0) ** 2)
peak2 = np.exp(-0.5 * ((X_RAMAN - 407.0) / 4.0) ** 2)
signal = peak1 + peak2
cube = (signal[None, None, :] + rng.uniform(0.0, 0.02, (Y_MAP, X_MAP, N_PTS))).astype(float)
rm = RamanMapping.from_arrays(
    cube, X_RAMAN, X_MAP, Y_MAP,
    custom_peaks=CUSTOM_PEAKS_RAMAN,
    data_range=(300.0, 700.0),
    background_remove=False,
    smoothing=False,
    normalize=False,
)
_inject_fit(rm, GOOD_PARAMS_RAMAN)
df_raman = rm.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])
out = Path(__file__).parent / "featuretable_raman_map_v0.6.6.csv"
df_raman.to_csv(out, index=False)
print(f"Raman mapping golden: {out}  ({len(df_raman)} rows, {len(df_raman.columns)} cols)")

# ---------------------------------------------------------------------------
# 2. PL mapping
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(20)
peak1 = np.exp(-0.5 * ((X_PL - 1.91) / 0.02) ** 2)
peak2 = np.exp(-0.5 * ((X_PL - 2.00) / 0.03) ** 2)
signal = peak1 + peak2
cube2 = (signal[None, None, :] + rng2.uniform(0.0, 0.02, (Y_MAP, X_MAP, N_PTS))).astype(float)
pm = PLMapping.from_arrays(
    cube2, X_PL, X_MAP, Y_MAP,
    custom_peaks=CUSTOM_PEAKS_PL,
    data_range=(1.85, 2.10),
    background_remove=False,
    smoothing=False,
    normalize=False,
)
_inject_fit(pm, GOOD_PARAMS_PL)
df_pl = pm.feature_table(ratios=[("Exciton", "Trion")], separations=[("Exciton", "Trion")])
out2 = Path(__file__).parent / "featuretable_pl_map_v0.6.6.csv"
df_pl.to_csv(out2, index=False)
print(f"PL mapping golden:    {out2}  ({len(df_pl)} rows, {len(df_pl.columns)} cols)")

# ---------------------------------------------------------------------------
# 3. Single-fit (RamanFit)
# ---------------------------------------------------------------------------
from ramanpl.single_fit.RamanFit import RamanFit

rf = RamanFit(
    X_RAMAN, peak1 + peak2,
    custom_peaks=CUSTOM_PEAKS_RAMAN,
    normalize=False,
    background_remove=False,
    smoothing=False,
)
rf.params_fit = GOOD_PARAMS_RAMAN.copy()
rf.peak_intensity = 1.0
rf.fit_diagnostics = {"rmse": 0.04, "n_starts": 1, "n_params_at_bounds": 0}
df_single = rf.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])
out3 = Path(__file__).parent / "featuretable_single_v0.6.6.csv"
df_single.to_csv(out3, index=False)
print(f"Single golden:        {out3}  ({len(df_single)} rows, {len(df_single.columns)} cols)")

# ---------------------------------------------------------------------------
# 4. Batch (via RamanBatch)
# ---------------------------------------------------------------------------
from ramanpl.batch import RamanBatch

rf1 = RamanFit(
    X_RAMAN, peak1 + peak2,
    custom_peaks=CUSTOM_PEAKS_RAMAN,
    normalize=False,
    background_remove=False,
    smoothing=False,
)
rf1.params_fit = GOOD_PARAMS_RAMAN.copy()
rf1.peak_intensity = 1.0
rf1.fit_diagnostics = {"rmse": 0.04, "n_starts": 1, "n_params_at_bounds": 0}

from ramanpl import dataImporter
import types

# Minimal stub to satisfy _BaseBatch.feature_table
class _FakeSpectrum:
    source = "synthetic_0"

batch = RamanBatch.__new__(RamanBatch)
batch.fits = [(_FakeSpectrum(), None, rf1)]

df_batch = batch.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])
out4 = Path(__file__).parent / "featuretable_batch_v0.6.6.csv"
df_batch.to_csv(out4, index=False)
print(f"Batch golden:         {out4}  ({len(df_batch)} rows, {len(df_batch.columns)} cols)")

print("\nDone. All four golden CSVs written.")
