# Mapping workflows

`RamanMapping` and `PLMapping` fit spectra over a 2D spatial grid, producing heatmaps of fitted parameters.

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
