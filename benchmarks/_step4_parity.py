"""
_step4_parity.py — v0.5.1 scientific-behaviour parity check
-------------------------------------------------------------
Run AFTER v0.5.1 code changes to confirm fitted_params and residual_map
are identical to the v0.5.0 baseline snapshot, and that the per-peak
columns in the wide-format export are byte-identical (new QA columns
excluded).

Usage
-----
    python benchmarks/_step4_parity.py
"""

import sys
import csv
import hashlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.mapping import RamanMapping

_BASELINE_DIR = Path(__file__).parent / "results" / "v0.5.0_baseline"
_OUT_DIR = Path(__file__).parent / "results" / "v0.5.1_parity"

_RNG = np.random.default_rng(42)
_N_PTS = 150
_X = np.linspace(200.0, 1800.0, _N_PTS)
_SHAPE = (3, 4, _N_PTS)
_PEAK = np.exp(-0.5 * ((_X - 520.0) / 15.0) ** 2)
_CUBE = (_PEAK[None, None, :] + _RNG.uniform(0.0, 0.05, _SHAPE)).astype(float)

_CUSTOM_PEAKS = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_DATA_RANGE = (200.0, 1800.0)
_FIT_KWARGS = {"diagnostics": "light", "random_state": 42, "n_starts": 1}


def run():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Re-run the same fit on modified code ---
    m = RamanMapping.from_arrays(
        _CUBE, _X, 4, 3,
        custom_peaks=_CUSTOM_PEAKS,
        data_range=_DATA_RANGE,
        background_remove=False,
        smoothing=False,
        normalize=False,
    )
    m.fit_spectra(fit_spectrum_kwargs=_FIT_KWARGS)

    # --- Load baseline ---
    snap = np.load(str(_BASELINE_DIR / "fitted_params.npz"))
    snap_fp = snap["fitted_params"]
    snap_rm = snap["residual_map"]

    ok = True

    # 1. fitted_params identical
    if np.array_equal(m.fitted_params, snap_fp):
        print("[Parity] fitted_params: IDENTICAL")
    else:
        diff = np.abs(m.fitted_params - snap_fp)
        print(f"[Parity] FAIL: fitted_params differ  max_abs={np.nanmax(diff):.3e}")
        ok = False

    # 2. residual_map identical (NaN-aware)
    if np.array_equal(
        np.where(np.isnan(m.residual_map), 0, m.residual_map),
        np.where(np.isnan(snap_rm), 0, snap_rm),
    ) and (np.isnan(m.residual_map) == np.isnan(snap_rm)).all():
        print("[Parity] residual_map: IDENTICAL")
    else:
        print("[Parity] FAIL: residual_map differs")
        ok = False

    # 3. Export per-peak columns byte-identical (exclude the 4 new QA columns)
    new_export = str(_OUT_DIR / "v0.5.1_export.txt")
    m.export_fit_map(new_export)

    def _read_data_rows(path):
        rows = []
        with open(path, newline="") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                rows.append(line.rstrip("\r\n"))
        return rows

    new_rows = _read_data_rows(new_export)
    old_rows = _read_data_rows(str(_BASELINE_DIR / "v0.5.0_export.txt"))

    # Strip the 4 QA columns from the new export rows (they are appended last)
    _N_NEW_COLS = 4

    def _strip_last_n(csv_line, n):
        # detect delimiter from first data row
        delim = "\t" if "\t" in csv_line else ","
        parts = csv_line.split(delim)
        return delim.join(parts[:-n]) if len(parts) > n else csv_line

    new_stripped = [_strip_last_n(r, _N_NEW_COLS) for r in new_rows]

    if new_stripped == old_rows:
        print("[Parity] wide-format per-peak columns: IDENTICAL")
    else:
        mismatch = [(i, o, n) for i, (o, n) in enumerate(zip(old_rows, new_stripped)) if o != n]
        for i, o, n in mismatch[:5]:
            print(f"  row {i}: OLD={o!r}")
            print(f"          NEW={n!r}")
        if len(new_stripped) != len(old_rows):
            print(f"  row count differs: old={len(old_rows)} new={len(new_stripped)}")
        print("[Parity] FAIL: per-peak column values differ")
        ok = False

    # Save diff report
    diff_path = _OUT_DIR / "fitted_params_diff.txt"
    diff_path.write_text("PASS\n" if ok else "FAIL — see stdout for details\n")

    if ok:
        print("[Parity] ALL CHECKS PASSED — safe to proceed.")
    else:
        print("[Parity] FAILURE — stop and diagnose before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    run()
