# Batch workflows

`RamanBatch` and `PLBatch` automate fitting over a collection of spectra loaded from multiple files.

## Use case

Batch workflows are for sets of independent spectra (e.g. multiple samples or measurement positions) where you want a unified result table. Use {doc}`mapping` instead when spectra come from a spatially ordered 2D grid.

## Loading files

```python
from ramanpl import RamanBatch

files = ["sample_01.txt", "sample_02.txt", "sample_03.txt"]

batch = RamanBatch(
    files,
    materials=["MoS2"],
    pipeline=pipe,
    preprocessing_backend="auto",
)
```

## Fitting

```python
batch.fit()
```

## Result table

```python
df = batch.table()
```

Returns a DataFrame with one row per spectrum and columns for each fitted peak parameter (position, FWHM, intensity/peak height).

## Feature tables

`feature_table()` returns a wide-format `pandas.DataFrame` — one row per source file — with per-peak descriptors and QA columns.

```python
df = batch.feature_table(
    separations=[("A1g", "E2g")],
    ratios=[("A1g", "E2g")],
)
# Columns: source, A1g_position, A1g_fwhm, …, A1g_E2g_separation, A1g_E2g_ratio,
#          rmse, ok, n_starts, n_params_at_bounds
```

**Order convention.** For `ratios=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_ratio = peak_height[P1] / peak_height[P2]`. For
`separations=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_separation = position[P1] − position[P2]`. Swapping the
order produces the reciprocal ratio (with a different column name) or
the negated separation. Zero denominator → NaN.

For unsupervised analysis on the resulting DataFrame, see {doc}`clustering`.

## Export

**Wide format** (one row per spectrum, all peaks as columns):
```python
batch.export("raman_batch.txt", wide=True)
```

**Long format** (one row per peak per spectrum):
```python
batch.export("raman_batch.txt", wide=False)
```

Export files include `export_kind: "batch_fit"` and backend provenance fields. See {doc}`export-provenance`.

## Visualisation

```python
batch.plot_waterfall()   # waterfall plot of all spectra
batch.plot_overlay()     # overlay of raw vs fitted spectra
```

## Backend provenance in batch

The `preprocessing_backend_requested` and `preprocessing_backend_resolved` fields appear in the export header for each batch run. If any spectrum falls back from RamanSPy to native, `preprocessing_backend_fallback_reason` is included.

## PLBatch

`PLBatch` follows the same interface. PL batch workflows are always native-only.

## See also

- {doc}`preprocessing` — pipeline construction
- {doc}`backend-behaviour` — backend selection contract
- {doc}`export-provenance` — export metadata fields
