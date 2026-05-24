# v0.6.4 Closeout

**Date:** 2026-05-24
**Branch:** `v0.6.4-dev`

---

## Feature: Parallel mapping fit with row-band warm-start

Two independent workstreams delivered in this build:

1. **Deprecation shim removal + `tol` parameter** — `methods`/`lam_grid` kwargs
   (shimmed in v0.6.3 with `DeprecationWarning`) are now removed; passing them
   raises `TypeError`. `tol` added to `_ALLOWED_PARAMS["arpls"]` and
   `_ALLOWED_PARAMS["airpls"]`, enabling convergence-tolerance sweeps via
   `method_grids`.

2. **`n_jobs` parallel row-band dispatch** — `RamanMapping.fit_spectra` and
   `PLMapping.fit_spectra` now accept `n_jobs` (default 1 = serial,
   byte-parity with v0.6.3). When `n_jobs > 1`, the Y-row loop is split into
   contiguous bands and dispatched via `joblib.Parallel(backend="loky")`.
   Module-level workers in `_parallel.py` avoid loky pickling restrictions on
   bound methods.  Measured speedup: 2.42× at `n_jobs=4` on `extended_15x15`.

---

## `n_jobs` API summary

```python
# Default (serial)
mapping.fit_spectra()

# Parallel — no warm-start (byte-identical to n_jobs=1)
mapping.fit_spectra(n_jobs=4)

# Parallel — warm-start (row_reset=True required)
mapping.fit_spectra(warm_start=True, seed_coord=(10, 10), row_reset=True, n_jobs=4)
```

### Hard constraints

| Condition | Behaviour |
|---|---|
| `n_jobs=1` | Serial; byte-parity with v0.6.3 |
| `n_jobs > self.Y` | Clamped to `self.Y` + `UserWarning` |
| `n_jobs > 1` + `warm_start=True` + `row_reset=False` | `ValueError` |
| Non-int or `< 1` | `ValueError` |

---

## Files changed

### New

| File | Step | Purpose |
|------|------|---------|
| `src/ramanpl/mapping/_parallel.py` | 5 | Band workers, splitter, validator, merger |
| `tests/test_parallel_fit_mapping.py` | 5 | 11 parallel-fit tests |
| `benchmarks/results/mapping_fit_benchmark_v0.6.4.csv` | 6 | Versioned benchmark (24 rows) |
| `benchmarks/results/parallel_speedup_v0.6.4.txt` | 6 | Speedup table |
| `docs/source/user-guide/parallel-fitting.md` | 7 | User guide for `n_jobs` |
| `closeout_v0.6.4.md` | 7 | This file |

### Modified

| File | Step | Change |
|------|------|--------|
| `src/ramanpl/_autotune.py` | 1 | `_shim_methods_lam_grid` removed; `methods`/`lam_grid` removed from signature |
| `src/ramanpl/mapping/_preprocess.py` | 1 | `methods`/`lam_grid` removed from `autotune_baseline` |
| `src/ramanpl/single_fit/RamanFit.py` | 1 | Same |
| `src/ramanpl/single_fit/PLfit.py` | 1 | Same |
| `src/ramanpl/_autotune.py` | 2 | `"tol"` added to `_ALLOWED_PARAMS["arpls"]` and `["airpls"]` |
| `src/ramanpl/mapping/_raman_mapping.py` | 3, 5 | `_fit_rows` extracted; `n_jobs` dispatch added |
| `src/ramanpl/mapping/_pl_mapping.py` | 3, 5 | Same |
| `pyproject.toml` | 4, 7 | `joblib>=1.3` dependency; `version = "0.6.4"` |
| `tests/test_autotune_baseline_mapping.py` | 1, 2 | `test_methods_kwarg_removed_raises_typeerror`; `test_tol_param_accepted_for_arpls` |
| `tests/test_autotune_baseline_single_fit.py` | 1 | `test_methods_kwarg_removed_raises_typeerror` |
| `tests/test_packaging_smoke.py` | 4, 7 | `test_joblib_importable`; version `"0.6.4"` |
| `benchmarks/benchmark_mapping_fit.py` | 6 | `n_jobs` axis (8→24 variants) |
| `tests/test_release_benchmark_smoke.py` | 6 | `"n_jobs"` in `_FIT_REQUIRED_FIELDS` |
| `example-usage/Mapping/Mapping Raman Example.ipynb` | 6b | `n_jobs` demo cells |
| `example-usage/Mapping/Mapping PL Example.ipynb` | 6b | `n_jobs` demo cells |
| `src/ramanpl/__init__.py` | 7 | `__version__ = "0.6.4"` |
| `CITATION.cff` | 7 | `version: 0.6.4`, `date-released: "2026-05-24"` |
| `docs/source/conf.py` | 7 | `release = "0.6.4"` |
| `docs/source/changelog.md` | 7 | v0.6.4 in Recent releases |
| `CHANGELOG` | 7 | `[v0.6.4] — 2026-05-24` section |
| `README.md` | 7 | v0.6.4 roadmap row → ✓ 2026-05-24 |

---

## Verification results

| Check | Result |
|-------|--------|
| `from ramanpl import __version__` | `"0.6.4"` ✓ |
| Byte-parity: `n_jobs=1` vs v0.6.3 benchmark | **identical n_curve_fit_calls** ✓ |
| `pytest tests/test_parallel_fit_mapping.py -v` | **11 passed** ✓ |
| `pytest -q` (full suite) | **315 passed, 3 skipped, 1 deselected** in 1416.86s ✓ |
| `grep _shim_methods_lam_grid src/` | **0 hits** ✓ |
| `python -c "import joblib"` | exits 0 ✓ |
| `benchmarks/results/mapping_fit_benchmark_v0.6.4.csv` | **24 rows**, no NaN in finite fields ✓ |
| Speedup at `n_jobs=4` (`extended_15x15`, `n_starts=1`) | **2.42×** ✓ |

---

## Hard constraints satisfied

- [x] `n_jobs=1` output byte-identical to v0.6.3 (no algorithmic changes)
- [x] Unsafe mode (`n_jobs>1 + warm_start=True + row_reset=False`) raises `ValueError`
- [x] `n_jobs > self.Y` clamped with `UserWarning` — never raises
- [x] Workers are module-level functions — loky pickling safe
- [x] No changes to per-pixel fitting algorithms (`_fit_utils.py`, `baselineAPI.py`, `peak_models.py`)
- [x] `joblib>=1.3` declared as hard dependency in `pyproject.toml`
