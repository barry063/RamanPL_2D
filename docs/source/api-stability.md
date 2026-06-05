# API stability — v0.5.5–v0.6.7 freeze contract

This document is the written, citable stability contract for the public surface
introduced in v0.5.1–v0.5.4 and extended additively through v0.6.6. The same
surfaces are enforced as regression tests in `tests/test_api_stability.py`.

---

## 1. Scope of the v0.5.5–v0.6.7 freeze

The v0.6.7 build extends the v0.5.5–v0.6.6 freeze contract additively with four new
feature-table column suffixes (`_component_area`, `_component_area_norm`,
`_component_area_fraction`, `_area_ratio`) and a new `area_ratios=` keyword on all
five `feature_table()` entry points. No existing frozen name is renamed, reordered,
or removed. No new fitting algorithms, preprocessing algorithms, or peak models are
introduced in v0.6.7.

The following four surfaces were frozen as of v0.5.5 and remain frozen through v0.6.7:

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

| Suffix | Meaning | Added |
|--------|---------|-------|
| `_position` | Fitted peak centre (cm⁻¹ for Raman; nm or eV for PL) | v0.5.2 |
| `_fwhm` | Full width at half maximum | v0.5.2 |
| `_peak_height` | Fitted peak height (absolute intensity) | v0.5.2 |
| `_peak_height_norm` | Peak height normalised (fit-space units) | v0.5.2 |
| `_component_area` | Analytic peak area in absolute intensity units (`amp × intensity_scale`) | v0.6.7 |
| `_component_area_norm` | Analytic peak area in normalised units (= `amp`) | v0.6.7 |
| `_component_area_fraction` | `area[peak] / Σ area[all peaks]`; 0-sum → NaN | v0.6.7 |
| `_separation` | Position difference for a peak pair (see §5) | v0.5.2 |
| `_ratio` | Peak-height ratio for a peak pair (see §5) | v0.5.2 |
| `_area_ratio` | Component-area ratio for a peak pair via `area_ratios=` (see §14) | v0.6.7 |

The column emission order within a single feature row is:
per-peak block (all suffixes for peak 1, then peak 2, …) → separation pairs →
height-ratio pairs → area-ratio pairs → QA block.

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

---

## 14. v0.6.7 — component-area contract

v0.6.7 adds component area columns to the feature table and a new `area_ratios=`
keyword to all five `feature_table()` entry points.

### Scientific basis

All three lineshapes in `peak_models.py` (Lorentzian, Gaussian, pseudo-Voigt) are
area-normalised: the analytic integral over (−∞, ∞) equals the `amp` fit parameter.
Therefore:

- `component_area_norm = amp` (normalised fit units, identical to `amp` in the
  fit parameter vector)
- `component_area = amp × intensity_scale` (absolute intensity units, same scale
  as `peak_height`)

The analytic identity holds exactly for the fitted model. The only approximation is
window truncation, which is negligible unless a peak is very broad relative to the
spectral window (i.e. FWHM is a significant fraction of the window width).

### New column definitions

| Column | Definition | Units |
|--------|------------|-------|
| `{peak}_component_area` | `amp × intensity_scale` | absolute intensity |
| `{peak}_component_area_norm` | `amp` | normalised (fit-space) |
| `{peak}_component_area_fraction` | `area[peak] / Σ area[all peaks]`; 0-sum → NaN | dimensionless |
| `{P1}_{P2}_area_ratio` | `area_norm[P1] / area_norm[P2]`; 0-denom → NaN | dimensionless |

Fraction and `area_ratio` are computed from `component_area_norm` (`amp`); because
`intensity_scale` is constant across peaks within a row, the scale-invariant result
is identical to the scaled form.

### `area_ratios=` keyword

All five `feature_table()` entry points accept a new `area_ratios=` keyword-only
argument (default `None`) with the same `list[tuple[str, str]]` signature as
`ratios=`. Each pair `(P1, P2)` adds a `{P1}_{P2}_area_ratio` column.

### Frozen contract

- Column order within a row: all per-peak suffixes for peak 1 (position, fwhm,
  peak_height, peak_height_norm, component_area, component_area_norm,
  component_area_fraction), then peak 2, … → separation pairs → ratio pairs →
  area_ratio pairs → QA block.
- `_component_area`, `_component_area_norm`, `_component_area_fraction`, and
  `_area_ratio` are added to `_FROZEN_SUFFIXES` in `test_api_stability.py`.
- `export()` and long-format exporters are **unchanged** in v0.6.7 (`amp` was
  already present in long-format output).
- No new `curve_fit` calls; no benchmark impact.
- Plotting of component-area columns is deferred to v0.6.8.
