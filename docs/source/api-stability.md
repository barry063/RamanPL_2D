# API stability — v0.5.5–v0.6.6 freeze contract

This document is the written, citable stability contract for the public surface
introduced in v0.5.1–v0.5.4 and extended additively through v0.6.6. The same
surfaces are enforced as regression tests in `tests/test_api_stability.py`.

---

## 1. Scope of the v0.5.5–v0.6.6 freeze

The v0.6.6 consolidation build extends the v0.5.5–v0.6.0 freeze contract to cover
the additive public surface introduced in v0.6.1–v0.6.5. No existing frozen name is
renamed, reordered, or removed. No new fitting algorithms, preprocessing algorithms,
peak models, or export schemas are introduced in v0.6.6.

The following four surfaces were frozen as of v0.5.5 and remain frozen through v0.6.6:

- **Feature-table column schema** — the set of column names emitted by
  `feature_table()`, and the relative order in which they appear.
- **Descriptor naming** — the suffix vocabulary used for per-peak columns.
- **`[ml]` extra boundary** — what is in the optional scikit-learn extra and
  what is not.
- **QA column names** — the names of the quality-assurance columns appended to
  every feature table.

Additive-only changes (new columns at the end, new keyword arguments with
backward-compatible defaults) are permitted within the v0.5.x series. No
frozen name will be renamed or reordered without a major version bump.

---

## 2. Frozen public functions and classes

| Symbol | Module |
|--------|--------|
| `build_feature_row` | `ramanpl.descriptors` |
| `validate_peak_pairs` | `ramanpl.descriptors` |
| `pca_reduce` | `ramanpl.ml` / `ramanpl.ml.clustering` |
| `kmeans_cluster` | `ramanpl.ml` / `ramanpl.ml.clustering` |
| `RamanMapping.feature_table` | `ramanpl.Mapping` |
| `PLMapping.feature_table` | `ramanpl.Mapping` |
| `RamanBatch.feature_table` | `ramanpl.batch` |
| `PLBatch.feature_table` | `ramanpl.batch` |
| `RamanFit.feature_table` | `ramanpl.RamanFit` |
| `PLfit.feature_table` | `ramanpl.PLfit` |

---

## 3. Frozen suffix vocabulary

Every per-peak column name is `{peak_label}{suffix}`, where `suffix` is one of:

| Suffix | Meaning |
|--------|---------|
| `_position` | Fitted peak centre (cm⁻¹ for Raman; nm or eV for PL) |
| `_fwhm` | Full width at half maximum |
| `_peak_height` | Fitted peak height (absolute intensity) |
| `_peak_height_norm` | Peak height normalised to the tallest peak in the fit |
| `_separation` | Position difference for a peak pair (see §5) |
| `_ratio` | Peak-height ratio for a peak pair (see §5) |

No other suffix will be introduced for per-peak columns without a major version
bump.

---

## 4. Frozen QA column names

Every feature table produced by `feature_table()` includes the following
quality-assurance columns as a contiguous block at the end of the DataFrame:

| Column | Type | Meaning |
|--------|------|---------|
| `rmse` | `float` | Root-mean-square residual of the fit |
| `ok` | `bool` | `True` if the fit converged within quality thresholds |
| `n_starts` | `int` | Number of multistart trials attempted |
| `n_params_at_bounds` | `int` | Number of fitted parameters that hit a bound |

Mapping outputs also include `x` and `y` (spatial coordinates) as the first
two columns.

---

## 5. Order convention for ratios and separations

For `ratios=[(P1, P2)]`:

- Emitted column name: `{P1}_{P2}_ratio`
- Value: `peak_height[P1] / peak_height[P2]`
- If `peak_height[P2] == 0`, the value is `NaN` (not `inf`).

For `separations=[(P1, P2)]`:

- Emitted column name: `{P1}_{P2}_separation`
- Value: `position[P1] − position[P2]`

Swapping the order produces the reciprocal ratio (with a different column name)
or the negated separation (with a different column name). Both are valid; the
convention must be applied consistently within a single analysis.

---

## 6. `[ml]` extra boundary

**In scope (v0.5.4 onwards):**

- PCA reduction on feature tables (`pca_reduce`)
- K-means clustering on feature tables (`kmeans_cluster`)
- Chaining PCA → k-means on the PC subspace

**Out of scope (deferred to v0.7.x, conditional on labelled datasets):**

- Supervised classification (e.g. layer-number labels)
- Any ML operating directly on raw spectra
- Any ML operating on preprocessing outputs (baseline, smoothed spectra)

The base install (`pip install -e .`) does not require scikit-learn. Importing
`ramanpl.ml` and `ramanpl.ml.clustering` without scikit-learn is safe; a clean
`ImportError` is raised only when `pca_reduce` or `kmeans_cluster` is called.

---

## 7. Modality parity

The frozen contract applies uniformly to Raman and PL data:

- The same six suffix primitives apply to both `RamanMapping` and `PLMapping`.
- The same QA column names appear in both.
- The same `pca_reduce` and `kmeans_cluster` functions operate on feature tables
  from both modalities.
- Peak labels are user-defined strings in both cases (e.g. `A1g`, `E2g` for
  Raman; `Exciton`, `Trion` for PL); no modality-specific column name
  convention exists.

---

## 8. Compatibility policy

Within the v0.5.x–v0.6.x series:

- **Additive-only** changes are permitted: new columns appended after the QA
  block, new keyword arguments with backward-compatible defaults, new public
  functions.
- **No breaking changes**: frozen column names will not be renamed, reordered,
  or removed.
- **Major version bump required** for any rename, reorder, or removal of a
  frozen name.

The regression test `tests/test_api_stability.py` enforces this contract
automatically. Any change that breaks those tests is a breaking change.

---

## 9. v0.6.1 — `show_progress` contract

v0.6.1 adds the `show_progress` keyword to `RamanMapping.fit_spectra`,
`PLMapping.fit_spectra`, and `fit_spectra_batch` with default **`True`**.

Frozen contract:

- `show_progress=True` is the default on all three entry points.
- Enabling or disabling progress display must not change fitted parameter values
  or any export column.
- The `tqdm` package is a hard dependency from v0.6.1 onwards.

---

## 10. v0.6.2 — autotune contract

v0.6.2 adds `autotune_baseline()` and `apply_choice()` to `RamanMapping`,
`PLMapping`, `RamanFit`, and `PLfit`.

Frozen contract:

- `autotune_baseline()` is **diagnostic and non-mutating**: it scores candidate
  baseline configurations but does not modify `self.preprocessing`.
- `apply_choice(result.winner)` is the **explicit mutation point**: it commits
  the chosen baseline spec to `self.preprocessing` and invalidates any cached
  preprocessed cube.
- The provenance block written to exports by autotune workflows is additive
  and does not alter the frozen feature-table column schema.

---

## 11. v0.6.3 — `method_grids` contract

v0.6.3 replaces the v0.6.2 `methods` / `lam_grid` kwargs with the unified
`method_grids` API.

Frozen contract:

- `method_grids` is the supported configuration API for `autotune_baseline()`.
- The removed `methods` and `lam_grid` kwargs from v0.6.2 are not restored and
  must not be reintroduced.

---

## 12. v0.6.4 — parallel-fit contract

v0.6.4 adds the `n_jobs` keyword to `RamanMapping.fit_spectra` and
`PLMapping.fit_spectra`.

Frozen contract:

- `n_jobs=1` (serial) is the default.
- `n_jobs > 1` uses row-band parallel fitting via `loky`; no output schema or
  feature-table column changes result from parallel execution.
- Unsafe warm-start state propagation across parallel workers raises explicitly
  rather than silently producing incorrect results.

---

## 13. v0.6.5 — cluster-seed contract

v0.6.5 adds the `cluster_seeds` keyword to `RamanMapping.fit_spectra` and
`PLMapping.fit_spectra`.

Frozen contract:

- `cluster_seeds=False` is the default; existing call sites are unaffected.
- `cluster_seeds=True` requires scikit-learn (`pip install ramanpl[ml]`) and
  requires `n_jobs=1`; using `cluster_seeds=True` with `n_jobs > 1` raises.
- `cluster_seeds=True` and `seed_coord` are mutually exclusive.
- The implementation is in `ramanpl.mapping._cluster_seeds` (package-private).
  It does not extend the public `ramanpl.ml` surface.
- Frozen column vocabulary, QA columns, and feature-table schema are unchanged.

Known limitation: on homogeneous synthetic cubes, `cluster_seeds=True` does not
reduce `n_curve_fit_calls` compared to the default initialisation. The measured
benefit depends on multi-domain data with distinct spectral regions.
