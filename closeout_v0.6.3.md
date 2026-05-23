# v0.6.3 Closeout

**Date:** 2026-05-22
**Branch:** `v0.6.3-dev`

---

## Feature: Autotune API refinement, real-data notebooks, API docs

Three pain points from the v0.6.2 closeout resolved:

1. **`method_grids` parameter sweeps** — `autotune_baseline()` now accepts a
   `{method: {param: [values]}}` dict; candidates are the Cartesian product over
   each method's parameter lists. The old `methods` / `lam_grid` pair is kept as
   a `DeprecationWarning` shim that reproduces the v0.6.2 24-candidate grid
   byte-for-byte when the defaults are used.

2. **Real-data notebooks** — `Baseline_Autotune_Demo.ipynb` extended with
   section "5. Real-data example: non-linear background" using a bilayer-graphene
   spectrum (`Raman Sample 532nm 2L-Graphene.txt`). Autotune blocks added to
   `Raman_background-remove.ipynb`; both notebooks added to / already in the smoke
   test suite.

3. **Sphinx autodoc** — `docs/source/api/autotune.rst` created with
   `automethod` directives for `autotune_baseline` and `apply_choice` on all four
   façade classes, plus `autoclass` for `BaselineAutotuneResult`.

---

## `method_grids` API summary

```python
# Focused two-axis sweep (v0.6.3+)
result = mapping.autotune_baseline(
    seed_coord=(1, 1),
    method_grids={
        "arpls":  {"lam": [1e4, 1e5, 1e6], "niter": [50, 100]},
        "poly":   {"poly_order": [1, 2, 3, 4, 5]},
    },
)

# Default (omit method_grids) — reproduces 24-candidate v0.6.2 grid
result = mapping.autotune_baseline(seed_coord=(1, 1))
```

### Default grid (unchanged from v0.6.2 — byte-parity preserved)

| Method     | Sweep                                    | Count |
|------------|------------------------------------------|-------|
| `asls`     | lam ∈ {1e3…1e7}, p=0.001, niter=20      | 5     |
| `arpls`    | lam ∈ {1e3…1e7}, niter=50               | 5     |
| `airpls`   | lam ∈ {1e3…1e6}, niter=50               | 4     |
| `poly`     | poly_order ∈ {1, 2, 3, 4, 5}            | 5     |
| `gaussian` | gaussian_sigma ∈ {5, 10, 20, 50, 100}   | 5     |
| **Total**  |                                          | **24** |

---

## Files changed

### New

| File | Step | Purpose |
|------|------|---------|
| `benchmarks/results/bench_snapshot_v0.6.2.txt` | 0b | Pre-v0.6.3 test baseline snapshot |
| `docs/source/api/autotune.rst` | 5 | Sphinx autodoc for autotune public surface |
| `closeout_v0.6.3.md` | 9 | This file |

### Modified

| File | Step | Change |
|------|------|--------|
| `src/ramanpl/_autotune.py` | 1 | `method_grids` param, `_shim_methods_lam_grid`, `_validate_method_grids`, `_default_baseline_grid` refactor, constants |
| `src/ramanpl/mapping/_preprocess.py` | 2 | `method_grids=None` façade kwarg pass-through |
| `src/ramanpl/single_fit/RamanFit.py` | 2 | Same |
| `src/ramanpl/single_fit/PLfit.py` | 2 | Same |
| `tests/test_autotune_baseline_mapping.py` | 3 | 10 call sites → `method_grids=`; 7 new tests |
| `tests/test_autotune_baseline_single_fit.py` | 3 | 9 call sites → `method_grids=`; 2 new tests |
| `docs/source/user-guide/baseline-autotune.md` | 4 | Rewritten with `method_grids` examples and "Deprecated arguments" subsection |
| `docs/source/api/index.rst` | 5 | `autotune` toctree entry |
| `example-usage/Validation/Baseline_Autotune_Demo.ipynb` | 6, post | Real-data section (bilayer graphene); observation narrative filled in |
| `example-usage/Ramanfit/Raman_background-remove.ipynb` | 7, post | Autotune blocks (7-candidate focused grid); section 6 Gaussian baseline example added |
| `tests/test_notebook_smoke.py` | 7 | `Raman_background-remove.ipynb` in `CANONICAL_NOTEBOOKS` |
| `pyproject.toml` | 8 | `version = "0.6.3"` |
| `src/ramanpl/__init__.py` | 8 | `__version__ = "0.6.3"` |
| `CITATION.cff` | 8 | `version: 0.6.3`, `date-released: "2026-05-22"` |
| `docs/source/conf.py` | 8 | `release = "0.6.3"` |
| `docs/source/changelog.md` | 8 | v0.6.3 in Recent releases |
| `CHANGELOG` | 8 | `[v0.6.3] — 2026-05-22` section |
| `README.md` | 8 | v0.6.3 roadmap row → ✓ 2026-05-22 |
| `tests/test_packaging_smoke.py` | 8 | Expected version `"0.6.3"` |

---

## Verification results

| Check | Result |
|-------|--------|
| `from ramanpl import __version__` | `"0.6.3"` ✓ |
| Default grid byte-parity vs `bench_snapshot_v0.6.2.txt` | **24 candidates — exact match** ✓ |
| `pytest tests/test_autotune_baseline_*.py -v` | **33 passed** (18 mapping + 15 single-fit) in 7.64s ✓ |
| `pytest -q` (full suite) | **305 passed, 3 skipped, 1 deselected** in 1133.10s (0:18:53) ✓ |
| Notebook smoke (8 notebooks, from full suite) | **8 passed** — including new `Raman_background-remove.ipynb` ✓ |
| DeprecationWarning from converted test call sites | **0** — no warnings from `method_grids=` sites ✓ |

---

## Hard constraints satisfied

- [x] No edits to `src/ramanpl/baselineAPI.py`
- [x] No new keyword on `fit_spectra` or `fit_spectrum`
- [x] No new runtime dependency in `pyproject.toml`
- [x] `autotune_baseline()` never mutates the object — only `apply_choice()` does
- [x] `apply_choice()` raises `ValueError` if pipeline has 0 or >1 `BaselineSubtract` steps
- [x] Default 24-candidate grid byte-parity with v0.6.2 preserved

---

## Post-checklist fixes

### Bug: double-plot in Jupyter when `plot=True`

`autotune_baseline(plot=True)` rendered the comparison figure twice in notebooks.
After `IPython.display.display(fig)` the figure remained in matplotlib's figure
manager; the `%matplotlib inline` cell-end hook then re-rendered it.
**Fix:** `plt.close(fig)` called immediately after `display()` in
`autotune_baseline_for_object` (`src/ramanpl/_autotune.py`). The figure object is
still accessible via `result.figure`. All 33 autotune tests pass after the fix.

### Bug: Python 3.9 import failure — `dict | None` annotation (CI P1)

The `method_grids` parameter in all three façade methods was annotated `dict | None`,
which is Python 3.10+ syntax and raises `TypeError` at class-definition time on 3.9,
breaking every mapping and single-fit import. **Fix:** annotation dropped from
`_preprocess.py`, `RamanFit.py`, and `PLfit.py`; type remains documented in the
docstring. Caught by CI Python 3.9 matrix job.

### Bug: one-shot iterator consumed by validator (CI P2)

`_validate_method_grids` called `len(list(v))` to check for empty value sequences,
exhausting generators before `_default_baseline_grid` could build candidates — silently
producing zero candidates and an `IndexError` at `ranking[0]`. **Fix:** `_default_baseline_grid`
now materialises all value sequences to plain lists before calling validation; the
per-value guard in `_validate_method_grids` tightened to `isinstance(v, list)` +
`len(v) == 0`. All 33 autotune tests pass.

### Notebooks completed

- `Baseline_Autotune_Demo.ipynb` section 5 — observation narrative filled in:
  Gaussian σ=50 won (RMSE=0.0712); iterative methods excluded from top-5 due to
  peak-to-window ratio; amplitude trade-off of Gaussian baseline documented.
- `Raman_background-remove.ipynb` section 6 added — "Using `gaussian` baseline":
  `gaussian_sigma=50`, same fit/plot structure as sections 1–4, tuning rules and
  amplitude trade-off note included.

---

## Known limitations — deferred to v0.6.4

### 1. `tol` absent from `_ALLOWED_PARAMS` for iterative methods

`_ALLOWED_PARAMS` for `airpls` and `arpls` does not include `tol`.  Passing
`tol` via `method_grids` raises `ValueError("Unknown parameter")`.  Pending
downstream verification that `baselineAPI.py` forwards `tol` to the solver
before exposing it in the validated parameter set.

### 2. Deprecated `methods` / `lam_grid` removal

`methods` and `lam_grid` emit `DeprecationWarning` in v0.6.3 and are
scheduled for removal in v0.6.4.  The shim must be deleted and the removal
noted in the v0.6.4 CHANGELOG.
