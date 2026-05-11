# Quickstart

This page shows the shortest working path for Raman and PL single-spectrum fitting.

## Raman single-spectrum

```python
import numpy as np
from ramanpl import RamanFit
from ramanpl.preprocessing import Pipeline, CropByRange, SmoothSavGol, BaselineSubtract

# Load your spectrum (wavenumber array and intensity array)
wavenumber = np.loadtxt("spectrum.txt", usecols=0)
intensity = np.loadtxt("spectrum.txt", usecols=1)

# Define a preprocessing pipeline
pipe = Pipeline(steps=[
    CropByRange((100, 500)),
    SmoothSavGol(window_length=9, polyorder=3),
    BaselineSubtract({"method": "poly", "poly_order": 3}),
])

# Fit with a materials library
fit = RamanFit(
    spectra=intensity,
    wavenumber=wavenumber,
    materials=["MoS2"],
    pipeline=pipe,
)
fit.fit()
fit.plot()
fit.export("result.txt")
```

## PL single-spectrum

```python
import numpy as np
from ramanpl import PLfit
from ramanpl.preprocessing import Pipeline, CropByRange, BaselineSubtract

wavelength = np.loadtxt("pl_spectrum.txt", usecols=0)
intensity = np.loadtxt("pl_spectrum.txt", usecols=1)

pipe = Pipeline(steps=[
    CropByRange((600, 800)),
    BaselineSubtract({"method": "poly", "poly_order": 3}),
])

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

## Feature tables

Every fitter class (`RamanFit`, `PLfit`, `RamanMapping`, `PLMapping`,
`RamanBatch`, `PLBatch`) exposes a `feature_table()` method that returns a
wide-format `pandas.DataFrame` — one row per spectrum — with per-peak
descriptors and QA columns:

```python
df = fit.feature_table(ratios=[("A1g", "E2g")], separations=[("A1g", "E2g")])
```

## Where to go next

- {doc}`user-guide/preprocessing` — pipeline construction and supported steps
- {doc}`user-guide/backend-behaviour` — native vs RamanSPy backend contract
- {doc}`user-guide/single-spectrum` — full single-spectrum workflow reference
- {doc}`examples/canonical-notebooks` — executable notebook examples
