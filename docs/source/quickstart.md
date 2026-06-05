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

### Component area columns (v0.6.7+)

`feature_table()` emits three component-area columns for each fitted peak,
plus an optional `area_ratios=` keyword for peak-to-peak area comparisons:

| Column | Meaning | Units |
|--------|---------|-------|
| `{peak}_component_area` | Analytic peak area (`amp × intensity_scale`) | same as `peak_height` |
| `{peak}_component_area_norm` | Analytic peak area in fit-space units (`amp`) | normalised |
| `{peak}_component_area_fraction` | `area[peak] / Σ area[all peaks]`; 0-sum → NaN | dimensionless |

```python
# Area ratio between two peaks
df = fit.feature_table(area_ratios=[("A1g", "E2g")])
# → adds column A1g_E2g_area_ratio = amp[A1g] / amp[E2g]
```

**Scientific basis:** all three lineshapes (Lorentzian, Gaussian, pseudo-Voigt)
in `peak_models.py` are area-normalised — the analytic integral over (−∞, ∞)
equals the `amp` fit parameter exactly. No extra fitting calls are needed.
`component_area_fraction` and `area_ratio` are scale-invariant (computed from
`amp` directly).

**Truncation caveat:** the analytic identity assumes integration over the full
real line. Window truncation is negligible for well-resolved peaks but may be
material for peaks very broad relative to the spectral window.

**Scope:** `export()` and long-format exporters are unchanged (`amp` was already
present in long-format output). Plotting of component-area columns is planned
for v0.6.8.

## Where to go next

- {doc}`user-guide/preprocessing` — pipeline construction and supported steps
- {doc}`user-guide/backend-behaviour` — native vs RamanSPy backend contract
- {doc}`user-guide/single-spectrum` — full single-spectrum workflow reference
- {doc}`examples/canonical-notebooks` — executable notebook examples
