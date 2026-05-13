# API stability — v0.5.5–v0.6.0 freeze contract

This document is the written, citable stability contract for the public surface
introduced in v0.5.1–v0.5.4. The same surfaces are enforced as a regression
test in `tests/test_api_stability.py`.

---

## 1. Scope of the v0.5.5–v0.6.0 freeze

v0.6.0 inherits the v0.5.5 freeze contract unchanged; the four frozen surfaces below
remain frozen through v0.6.0 without modification.

The following four surfaces are frozen as of v0.5.5:

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

Within the v0.5.x series:

- **Additive-only** changes are permitted: new columns appended after the QA
  block, new keyword arguments with backward-compatible defaults, new public
  functions.
- **No breaking changes**: frozen column names will not be renamed, reordered,
  or removed.
- **Major version bump required** for any rename, reorder, or removal of a
  frozen name.

The regression test `tests/test_api_stability.py` enforces this contract
automatically. Any change that breaks those tests is a breaking change.
