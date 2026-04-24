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
