# v0.6.5 Closeout — Similarity-Based Seed Selection for Warm-Start

**Date:** 2026-05-28  
**Branch:** `v0.6.5-dev`  
**Released from:** `main` (after branch merge)

---

## Feature Summary

v0.6.5 adds an opt-in `cluster_seeds` keyword to `RamanMapping.fit_spectra` and `PLMapping.fit_spectra`. When enabled, preprocessed spectra are grouped into clusters using PCA + k-means (scikit-learn); one representative pixel per cluster is fitted first; each representative's fitted parameters are then used as the initial guess (`p0`) for the remaining pixels in that cluster.

**Algorithm contract:**
- The final reported parameters still come from normal Lorentzian / pseudo-Voigt least-squares fitting on each pixel. Cluster seeding changes only the initial guess; it does not alter the fitting model.
- `cluster_seeds=True` is serial-only (`n_jobs=1` required). Two-phase parallel dispatch is deferred.
- `cluster_seeds=False` (default) is byte-identical to v0.6.4.

---

## API Summary

```python
# Default: disabled, byte-parity with v0.6.4
mapping.fit_spectra()
mapping.fit_spectra(cluster_seeds=False)

# Enabled with defaults: n_clusters=min(8, sqrt(X*Y)), n_components=3
mapping.fit_spectra(cluster_seeds=True, n_jobs=1)

# Custom config
mapping.fit_spectra(
    cluster_seeds={"n_clusters": 6, "n_components": 3, "random_state": 42},
    n_jobs=1,
)
```

**Hard constraints:**

| Mode | Behaviour |
|---|---|
| `cluster_seeds=False, n_jobs=1` | v0.6.4 serial — byte-identical |
| `cluster_seeds=False, n_jobs>1` | v0.6.4 row-band parallel — unchanged |
| `cluster_seeds=True, n_jobs=1` | Full cluster-seeded warm-start |
| `cluster_seeds=True, n_jobs>1` | `ValueError` naming both resolutions |
| `cluster_seeds=True, seed_coord=...` | `ValueError` naming both resolutions |

**`initial_p0` with `cluster_seeds=True`:** `initial_p0` is used as the base `p0` for representative-pixel fits; cluster members then receive each representative's fitted result as their `p0_start`.

**`warm_start` interaction:** `cluster_seeds=True` does not silently mutate `warm_start`. Within a cluster, `warm_start=True` propagates successful fits between members (intra-cluster); `warm_start=False` means all members start from the representative's result only, without intra-cluster chaining.

**Default `n_clusters` formula:** `min(8, max(1, int(sqrt(X*Y))))`. This is a pragmatic heuristic, not a principled optimum; documented as such.

**scikit-learn dependency:** required for `cluster_seeds=True`; base install unaffected when `cluster_seeds=False`.

---

## Files Changed

### New

| File | Purpose |
|---|---|
| `src/ramanpl/mapping/_cluster_seeds.py` | Helper module: 6 functions (lazy sklearn imports) |
| `tests/test_cluster_seed_helpers.py` | 15 tests: 10 helper tests + 5 schedule invariant tests |
| `tests/test_cluster_seed_fit_mapping.py` | 15 integration tests |
| `benchmarks/results/mapping_fit_benchmark_v0.6.5.csv` | Versioned benchmark results |
| `benchmarks/results/cluster_seed_speedup_v0.6.5.txt` | Call-count comparison |
| `closeout_v0.6.5.md` | This document |

### Modified

| File | Change |
|---|---|
| `src/ramanpl/mapping/_raman_mapping.py` | `cluster_seeds` keyword, validation, `_fit_single_pixel`, cluster dispatch |
| `src/ramanpl/mapping/_pl_mapping.py` | Same as Raman |
| `benchmarks/benchmark_mapping_fit.py` | `cluster_seeds` axis; `noisy_5x8` case; speedup writer |
| `benchmarks/validation_v0.6.0_vs_v0.5.0.py` | `"cluster_seeds"` and `"n_jobs"` added to `_CSV_FIELDS`; same fields added to `_run_hard_case` return dict (post-merge CI fix) |
| `tests/test_api_stability.py` | New test for `cluster_seeds` keyword presence |
| `tests/test_release_benchmark_smoke.py` | Expects `cluster_seeds` field in records |
| `tests/test_cluster_seed_helpers.py` | Six `_build_cluster_schedule` fixtures updated to `(cluster_id, (x, y))` format (post-merge fix) |
| `src/ramanpl/mapping/_raman_mapping.py` | Cluster seed broadcast gated on `warm_start_rmse_gate` (post-merge fix) |
| `src/ramanpl/mapping/_pl_mapping.py` | Same as Raman (post-merge fix) |
| `tests/test_packaging_smoke.py` | Version bump to `0.6.5` |
| `pyproject.toml` | Version: `0.6.5` |
| `src/ramanpl/__init__.py` | `__version__ = "0.6.5"` |
| `CITATION.cff` | `version: 0.6.5`, `date-released: "2026-05-28"` |
| `docs/source/conf.py` | `release = "0.6.5"` |
| `docs/source/user-guide/mapping.md` | `cluster_seeds` usage section |
| `docs/source/user-guide/parallel-fitting.md` | Serial-only constraint |
| `docs/source/api-stability.md` | §9 additive changes |
| `CHANGELOG` | v0.6.5 section |
| `docs/source/changelog.md` | v0.6.5 entry |
| `example-usage/Mapping/Mapping Raman Example.ipynb` | `cluster_seeds` markdown + code demo cells (Option B) |
| `example-usage/Mapping/Mapping PL Example.ipynb` | `cluster_seeds` markdown + code demo cells (Option B) |

---

## Verification Results

### Test commands run

```
pytest tests/test_cluster_seed_helpers.py -q        → 15/15 passed
pytest tests/test_cluster_seed_fit_mapping.py -v    → 15/15 passed
pytest tests/test_parallel_fit_mapping.py -q        → 26/26 passed
pytest tests/test_api_stability.py -q               → 9/9 passed
pytest tests/test_release_benchmark_smoke.py -q     → 9/9 passed
pytest tests/test_ml_clustering.py -q               → 26/26 passed
python -c "import ramanpl; import ramanpl.mapping"  → OK (no sklearn)
python benchmarks/benchmark_mapping_fit.py --v065   → completed
```

### Benchmark acceptance gates

1. **Success rate (primary gate):** `cluster_seeds=True` success rate = 1.0 on all 4 benchmark cubes. Same as v0.6.4 baseline. **PASS.**

2. **`n_curve_fit_calls` reduction (algorithmic gate):** 0% reduction on all benchmark cubes. **NOT VERIFIED on synthetic data.** See "Known Limitations" below.

3. **Parameter tolerance (secondary sanity check):** `test_cluster_seeds_params_within_tolerance_raman` passes with `rtol=1e-3, atol=1e-5`. **PASS.**

---

## Known Limitations

### Call-count reduction not measurable on synthetic benchmark cubes

The `n_curve_fit_calls` acceptance gate requires that cluster-seeded fitting uses fewer `curve_fit` invocations than the baseline on at least one benchmark cube. On the current synthetic benchmark cubes (all pixels drawn from the same spectral shape with additive Gaussian noise), `n_curve_fit_calls` is identical for `cluster_seeds=True` and `cluster_seeds=False`.

**Reason:** On homogeneous single-domain cubes, the optimizer converges to the global minimum from any reasonable starting point (`p0_base` midpoint or cluster representative's result). The RMSE after the first curve_fit call is already the best achievable, so the retry trigger is independent of the starting point.

**Expected benefit on real data:** The call-count benefit manifests when different spatial regions have different spectral signatures (multi-domain maps). In that case, the representative pixel's fitted parameters are a better starting guess for same-domain members than the global midpoint, which can reduce the number of adaptive-multistart retries.

**Recorded tolerance change:** None. The secondary tolerance (`rtol=1e-3, atol=1e-5`) is unchanged from the plan.

### Serial-only

`cluster_seeds=True` raises `ValueError` when `n_jobs > 1`. Two-phase parallel cluster dispatch is a candidate for a later release.

### Notebook decision

**Option B chosen** (demo cells inside existing mapping notebooks). Two cells (markdown explanation + code demo) were added to each of:

- `example-usage/Mapping/Mapping Raman Example.ipynb`
- `example-usage/Mapping/Mapping PL Example.ipynb`

Both pairs are inserted immediately after the existing `n_jobs` demo section, following the same style. Each code cell demonstrates `cluster_seeds=True` (auto defaults) and `cluster_seeds={...}` (custom config). No standalone demo notebook was created.

---

## Refactor-drift record

No helpers were extracted from `_fit_rows`. The new `_fit_single_pixel` method is an addition to each mapping class (new code, not extracted from existing code). `_fit_rows` is unchanged.

---

## Hard constraints verified

| Constraint | Status |
|---|---|
| `cluster_seeds=False` byte-identical to v0.6.4 | VERIFIED (parity tests) |
| Final parameters from normal curve_fit on each pixel | VERIFIED (model unchanged) |
| No changes to `baselineAPI.py`, `peak_models.py`, curve-fit objective | VERIFIED |
| Base install clean without scikit-learn | VERIFIED (smoke test + no-import test) |
| `cluster_seeds=True + n_jobs>1` raises | VERIFIED |
| `cluster_seeds=True + seed_coord` raises | VERIFIED |
| No export schema changed | VERIFIED (API stability tests) |
