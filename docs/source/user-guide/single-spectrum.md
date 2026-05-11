# Single-spectrum fitting

RamanPL_2D provides two single-spectrum fitters: `RamanFit` for Raman spectra and `PLfit` for photoluminescence spectra.

## RamanFit

### Data loading

`RamanFit` expects a 1D intensity array and a 1D wavenumber array (cm⁻¹):

```python
from ramanpl import RamanFit
import numpy as np

wavenumber, intensity = np.loadtxt("spectrum.txt", unpack=True)
```

For `.wdf` files from Renishaw WiRE, use `DataImporter`:

```python
from ramanpl import DataImporter
data = DataImporter("spectrum.wdf")
wavenumber, intensity = data.wavenumber, data.spectra
```

### Peak initialisation

**From a materials library:**
```python
fit = RamanFit(spectra=intensity, wavenumber=wavenumber, materials=["MoS2"])
```

**Custom peaks** — each entry is `([lower_bounds], [upper_bounds])` for `[position, width, amplitude]`:
```python
fit = RamanFit(
    spectra=intensity,
    wavenumber=wavenumber,
    custom_peaks={
        "P1": ([380, 2, 0], [385, 20, 5000]),
        "P2": ([405, 2, 0], [410, 20, 5000]),
    },
)
```

**Remove unwanted library peaks:**
```python
fit = RamanFit(spectra=intensity, wavenumber=wavenumber,
               materials=["MoS2"], remove_peaks=["substrate_Si"])
```

### Peak profiles

| Profile | Argument |
|---------|----------|
| Lorentzian (default) | `peak_model="lorentzian"` |
| Pseudo-Voigt | `peak_model="pvoigt"` |

### Preprocessing pipeline

```python
from ramanpl.preprocessing import Pipeline, CropByRange, SmoothSavGol, BaselineSubtract

pipe = Pipeline(steps=[
    CropByRange((100, 500)),
    SmoothSavGol(window_length=9, polyorder=3),
    BaselineSubtract({"method": "poly", "poly_order": 3}),
])

fit = RamanFit(spectra=intensity, wavenumber=wavenumber,
               materials=["MoS2"], pipeline=pipe)
```

### Fitting, plotting, exporting

```python
fit.fit()
fit.plot()
fit.export("result.txt")
```

### Feature tables

`feature_table()` returns a single-row `pandas.DataFrame` of peak descriptors and QA columns after fitting.

```python
df = fit.feature_table(
    separations=[("A1g", "E2g")],
    ratios=[("A1g", "E2g")],
)
# Columns: A1g_position, A1g_fwhm, A1g_peak_height, A1g_peak_height_norm,
#          E2g_*, A1g_E2g_separation, A1g_E2g_ratio,
#          rmse, ok, n_starts, n_params_at_bounds
```

The same method is available on `PLfit`.

**Order convention.** For `ratios=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_ratio = peak_height[P1] / peak_height[P2]`. For
`separations=[(P1, P2)]`, the emitted column is
`{P1}_{P2}_separation = position[P1] − position[P2]`. Swapping the
order produces the reciprocal ratio (with a different column name) or
the negated separation. Zero denominator → NaN.

For unsupervised analysis on the resulting DataFrame, see {doc}`clustering`.

## PLfit

`PLfit` follows the same interface as `RamanFit`. PL workflows always use the native backend regardless of `preprocessing_backend`.

```python
from ramanpl import PLfit

fit = PLfit(
    spectra=intensity,
    wavenumber=wavelength,
    custom_peaks={
        "A": ([680, 5, 0], [700, 50, 5000]),
    },
    pipeline=pipe,
)
fit.fit()
fit.plot()
fit.export("pl_result.txt")
```

## See also

- {doc}`preprocessing` — pipeline construction
- {doc}`backend-behaviour` — backend selection contract
- {doc}`../api/index` — API reference
