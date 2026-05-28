# Mapping workflows

`RamanMapping` and `PLMapping` fit spectra over a 2D spatial grid, producing heatmaps of fitted parameters.

## Baseline auto-tuning (v0.6.2+)

Before running a full mapping fit you can score a grid of baseline configurations
on a single seed pixel to find the most appropriate baseline for your dataset.
See {doc}`baseline-autotune` for the full workflow, API reference, and default
grid specification.

## Data structure

A mapping dataset is a 3D spectral cube: `(rows, cols, spectral_axis)`. Each pixel contains one spectrum. The mapping classes handle:

- spectral axis (wavenumber in cm⁻¹ for Raman, wavelength or energy for PL)
- spatial dimensions (rows × columns)
- per-pixel fitting results

## Loading mapping data

```python
from ramanpl import DataImporter

data = DataImporter("mapping.wdf")  # Renishaw WiRE mapping file
```

Or construct from arrays:

```python
from ramanpl.mapping import RamanMapping

raman_map = RamanMapping(
    spectra=cube,          # shape (rows, cols, n_wavenumbers)
    wavenumber=wn,
    materials=["MoS2"],
    pipeline=pipe,
)
```

## Preprocessing

The same `Pipeline` used in single-spectrum workflows applies to mapping cubes. The mapping preprocessor applies vectorised operations over all pixels.

For Raman mapping with a translatable pipeline, `preprocessing_backend="auto"` will use RamanSPy if available. See {doc}`backend-behaviour` for details.

## Fitting

```python
raman_map.fit_spectra(
    warm_start=True,
    fit_spectrum_kwargs=dict(
        adaptive_multistart=True,
        fast_n_starts=1,
        n_starts=4,
        p0_strategy="jitter",
        retry_on_fail=True,
        retry_on_high_rmse=True,
        retry_rmse_gate=0.10,
        diagnostics="light",
    )
)
```

**Warm start:** reuses the previous pixel's fit result as the initial guess for the next pixel — useful for spatially correlated maps.

## Cluster-seeds warm-start (v0.6.5+)

`cluster_seeds=True` groups spectra into clusters before fitting, fits one representative pixel per cluster first, and uses each representative's fitted parameters as the initial guess (`p0`) for the remaining pixels in that cluster.

```python
raman_map.fit_spectra(
    warm_start=True,
    cluster_seeds=True,           # opt-in; default False
    fit_spectrum_kwargs={"random_state": 42},
)
```

The default clustering uses `n_clusters = min(8, max(1, √(X×Y)))` PCA components and k-means. This is a pragmatic heuristic — it is not a principled optimal choice. For fine-grained control, pass a config dict:

```python
raman_map.fit_spectra(
    cluster_seeds={"n_clusters": 6, "n_components": 3, "random_state": 42},
)
```

**Constraints (v0.6.5):**

- `cluster_seeds=True` requires `n_jobs=1`. Passing `n_jobs > 1` raises `ValueError`.
- `cluster_seeds=True` and `seed_coord` are mutually exclusive.
- The final reported parameters still come from normal Lorentzian / pseudo-Voigt least-squares fitting on each pixel. Cluster seeding only changes the initial guess; it does not alter the fitting model.
- `scikit-learn` must be installed (`pip install ramanpl[ml]`). The base install is unaffected when `cluster_seeds=False` (default).

## Diagnostics modes

| Mode | Behaviour |
|------|-----------|
| `"full"` | Stores per-pixel diagnostics including bound masks |
| `"light"` | Stores compact QA summaries only |
| `"none"` | Disables per-pixel diagnostics storage |

`fit_summary()` still works with `diagnostics="none"` using the residual map.

## Fit quality summary

```python
rep = raman_map.fit_summary()
```

Reports fit success rate, RMSE statistics, failure reasons, and bound-sticking summaries.

## Parameter maps and visualisation

```python
raman_map.plot_map("A1g_position")    # heatmap of peak position
raman_map.plot_map("A1g_fwhm")        # heatmap of FWHM
raman_map.plot_map("A1g_intensity")   # heatmap of peak intensity
```

Raman-specific derived maps:

```python
raman_map.plot_map("A1g_E2g_separation")  # A1g − E2g peak separation
raman_map.plot_map("E2g_A1g_ratio")       # E2g / A1g intensity ratio
```

## Feature tables

`feature_table()` returns a wide-format `pandas.DataFrame` — one row per pixel — with per-peak descriptors and QA columns. Optional `ratios` and `separations` arguments add derived columns.

```python
df = raman_map.feature_table(
    ratios=[("A1g", "E2g")],
    separations=[("A1g", "E2g")],
)
# Columns: x, y, A1g_position, A1g_fwhm, A1g_peak_height, A1g_peak_height_norm,
#          E2g_*, A1g_E2g_separation, A1g_E2g_ratio,
#          rmse, ok, n_starts, n_params_at_bounds
```

The same method is available on `PLMapping`. Failed pixels emit a full row with `ok=False` and NaN per-peak fields.

**Order convention.** For `ratios=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_ratio = peak_height[P1] / peak_height[P2]`. For
`separations=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_separation = position[P1] − position[P2]`. Swapping the
order produces the reciprocal ratio (with a different column name) or
the negated separation. Zero denominator → NaN.

The feature-table column schema and the `[ml]` clustering tools apply
uniformly to `RamanMapping` and `PLMapping`. The example above uses
Raman peak labels; replace them with PL peak labels (e.g. `Trion`,
`Exciton`) without changing any column-naming convention.

For unsupervised analysis on the resulting DataFrame, see {doc}`clustering`.

## Residual maps

```python
raman_map.plot_residual_map()
raman_map.inspect_residuals()
```

## Export

```python
raman_map.export("map_results.txt")
```

Export files include backend provenance metadata. See {doc}`export-provenance`.

## PLMapping

`PLMapping` follows the same interface. PL mapping is always native-only.

## See also

- {doc}`backend-behaviour` — backend contract for mapping preprocessing
- {doc}`export-provenance` — export metadata fields
