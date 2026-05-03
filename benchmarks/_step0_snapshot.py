"""
_step0_snapshot.py — v0.5.0 baseline snapshot generator
---------------------------------------------------------
Run BEFORE any v0.5.1 code changes to record:
  1. fitted_params.npz   — deterministic fit results
  2. v0.5.0_export.txt   — export_fit_map output
  3. checksums.txt        — SHA256 of the export file
  4. v0.5.0_baseline.json — wall-clock timing and curve_fit call count

Usage
-----
    python benchmarks/_step0_snapshot.py

Output
------
    benchmarks/results/v0.5.0_baseline/   (gitignored)
"""

import sys
import time
import json
import hashlib
import statistics
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.mapping import RamanMapping

# ---------------------------------------------------------------------------
# Synthetic cube — mirrors "small_3x4" from benchmark_mapping_preprocessing.py
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_N_PTS = 150
_X = np.linspace(200.0, 1800.0, _N_PTS)
_SHAPE = (3, 4, _N_PTS)  # Y=3, X=4
_PEAK = np.exp(-0.5 * ((_X - 520.0) / 15.0) ** 2)
_CUBE = (_PEAK[None, None, :] + _RNG.uniform(0.0, 0.05, _SHAPE)).astype(float)

_CUSTOM_PEAKS = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_DATA_RANGE = (200.0, 1800.0)
_FIT_KWARGS = {"diagnostics": "light", "random_state": 42, "n_starts": 1}

_OUT_DIR = Path(__file__).parent / "results" / "v0.5.0_baseline"


def _build_mapping():
    return RamanMapping.from_arrays(
        _CUBE, _X, 4, 3,
        custom_peaks=_CUSTOM_PEAKS,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )


def run():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fitted-parameter snapshot
    # ------------------------------------------------------------------
    m = _build_mapping()
    m.fit_spectra(fit_spectrum_kwargs=_FIT_KWARGS)

    # Save diagnostics as a plain object array via np.save
    diag_path = _OUT_DIR / "fitted_params.npz"
    np.savez(
        diag_path,
        fitted_params=m.fitted_params,
        residual_map=m.residual_map,
    )
    # fit_diagnostics_map is an object array — save separately with allow_pickle
    np.save(str(_OUT_DIR / "fit_diagnostics_map.npy"), m.fit_diagnostics_map, allow_pickle=True)
    print(f"[Step 0] Saved fitted_params.npz  →  {diag_path}")

    # ------------------------------------------------------------------
    # 2. Export-file snapshot + SHA256
    # ------------------------------------------------------------------
    export_path = _OUT_DIR / "v0.5.0_export.txt"
    m.export_fit_map(str(export_path))

    with open(export_path, "rb") as fh:
        sha256 = hashlib.sha256(fh.read()).hexdigest()

    checksums_path = _OUT_DIR / "checksums.txt"
    checksums_path.write_text(f"sha256:{sha256}  v0.5.0_export.txt\n")
    print(f"[Step 0] Saved export  →  {export_path}  (SHA256={sha256[:16]}...)")

    # ------------------------------------------------------------------
    # 3. Wall-clock timing + n_curve_fit_calls
    # ------------------------------------------------------------------
    import scipy.optimize as _scipy_opt

    _real_curve_fit = _scipy_opt.curve_fit

    def _run_timed():
        counter = {"n": 0}

        def _wrapped(*a, **kw):
            counter["n"] += 1
            return _real_curve_fit(*a, **kw)

        with patch.object(_scipy_opt, "curve_fit", _wrapped):
            t0 = time.perf_counter()
            m2 = _build_mapping()
            m2.fit_spectra(fit_spectrum_kwargs=_FIT_KWARGS)
            dt = time.perf_counter() - t0

        return dt, counter["n"]

    times = []
    n_calls_list = []
    for _ in range(3):
        dt, n = _run_timed()
        times.append(dt)
        n_calls_list.append(n)

    baseline = {
        "runtime_s_min":    round(min(times), 6),
        "runtime_s_median": round(statistics.median(times), 6),
        "runtime_s_max":    round(max(times), 6),
        "n_curve_fit_calls": int(statistics.median(n_calls_list)),
    }
    json_path = _OUT_DIR / "v0.5.0_baseline.json"
    json_path.write_text(json.dumps(baseline, indent=2))
    print(f"[Step 0] Saved baseline JSON  →  {json_path}")
    print(f"         {baseline}")
    print("[Step 0] Done.")


if __name__ == "__main__":
    run()
