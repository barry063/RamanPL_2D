# v0.6.2 Closeout

**Date:** 2026-05-20
**Branch:** `v0.6.2-dev`
**Worktree:** `../RamanPL_2D-v0.6.2`

---

## Feature: Seed-pixel baseline auto-tuning

Opt-in `autotune_baseline` / `apply_choice` diagnostic added to all four fit classes:
`RamanMapping`, `PLMapping`, `RamanFit`, `PLfit`.

### Workflow

1. Call `obj.autotune_baseline(seed_coord=..., plot=True)` — scores a configurable grid of
   baseline candidates on a representative spectrum (read-only, never mutates the object).
2. Inspect the RMSE ranking: `result.ranking[:5]`.
3. Call `obj.apply_choice(result.winner)` — commits the winner, invalidates preprocessing
   caches, re-applies the full pipeline from the pristine raw spectrum.
4. Run `fit_spectra()` / `fit_spectrum()` as normal.

### Default grid (24 candidates)

| Method   | Sweep                                   | Count |
|----------|-----------------------------------------|-------|
| `asls`   | lam ∈ {1e3, 1e4, 1e5, 1e6, 1e7}        | 5     |
| `arpls`  | lam ∈ {1e3, 1e4, 1e5, 1e6, 1e7}        | 5     |
| `airpls` | lam ∈ {1e3, 1e4, 1e5, 1e6}             | 4     |
| `poly`   | poly_order ∈ {1, 2, 3, 4, 5}           | 5     |
| `gaussian` | gaussian_sigma ∈ {5, 10, 20, 50, 100} | 5     |
| **Total** |                                        | **24** |

---

## Files changed

### New

| File | Step | Purpose |
|------|------|---------|
| `src/ramanpl/_autotune.py` | 2 | Core scoring logic, `BaselineAutotuneResult`, default grid |
| `tests/test_autotune_baseline_mapping.py` | 6 | 10 mapping autotune tests |
| `tests/test_autotune_baseline_single_fit.py` | 6 | 12 single-fit autotune tests |
| `docs/source/user-guide/baseline-autotune.md` | 10 | Full API reference and workflow docs |
| `example-usage/Validation/Baseline_Autotune_Demo.ipynb` | 9 | Demo notebook (synthetic data) |
| `benchmarks/results/bench_snapshot_v0.6.1.txt` | 1 | Pre-feature test suite snapshot |

### Modified

| File | Step | Change |
|------|------|--------|
| `src/ramanpl/mapping/_preprocess.py` | 3, 5 | `autotune_baseline`, `apply_choice`, provenance |
| `src/ramanpl/single_fit/RamanFit.py` | 4 | `_raw_spectra_pristine`, `_x_axis_pristine`, façade methods |
| `src/ramanpl/single_fit/PLfit.py` | 4 | Same as RamanFit (PL axis) |
| `src/ramanpl/single_fit/_single_fit_core.py` | 5 | `baseline_autotune` provenance block |
| `pyproject.toml` | 7 | `version = "0.6.2"` |
| `src/ramanpl/__init__.py` | 7 | `__version__ = "0.6.2"` |
| `CITATION.cff` | 7 | `version: 0.6.2`, `date-released: "2026-05-20"` |
| `docs/source/conf.py` | 7 | `release = "0.6.2"` |
| `docs/source/index.md` | 10 | `baseline-autotune` toctree entry |
| `docs/source/user-guide/mapping.md` | 10 | Cross-link to baseline-autotune |
| `docs/source/changelog.md` | 7 | v0.6.2 entry in Recent releases |
| `CHANGELOG` | 7 | Full `[v0.6.2] — 2026-05-20` section |
| `README.md` | 7 | v0.6.2 roadmap row → ✓ |
| `tests/test_notebook_smoke.py` | 9 | `Baseline_Autotune_Demo.ipynb` in CANONICAL_NOTEBOOKS |
| `tests/test_packaging_smoke.py` | 7 | Expected version `"0.6.2"` |

---

## Verification results

| Check | Result |
|-------|--------|
| `pytest -q` (full suite) | **293 passed, 3 skipped** |
| `pytest tests/test_autotune_baseline_*.py -v` | **22 passed** |
| `pytest tests/test_notebook_smoke.py::test_notebook_executes_without_error[Baseline_Autotune_Demo.ipynb]` | **PASSED** |
| `from ramanpl import __version__` (worktree src) | `"0.6.2"` |
| Benchmark parity (`n_curve_fit_calls` row-by-row) | **PASS** — autotune is read-only; `fit_spectra` unchanged |

---

## Hard constraints satisfied

- [x] No edits to `src/ramanpl/baselineAPI.py`
- [x] No new keyword on `fit_spectra` or `fit_spectrum`
- [x] No new runtime dependency in `pyproject.toml`
- [x] `autotune_baseline()` never mutates the object — only `apply_choice()` does
- [x] `apply_choice()` raises `ValueError` if pipeline has 0 or >1 `BaselineSubtract` steps
