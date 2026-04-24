# Preprocessing

RamanPL_2D uses a modular `Pipeline` abstraction to define preprocessing steps applied before spectral fitting.

## Pipeline

A `Pipeline` is an ordered sequence of preprocessing steps. The same pipeline object can be passed to single-spectrum fitters, batch workflows, and mapping workflows.

```python
from ramanpl.preprocessing import Pipeline, CropByRange, SmoothSavGol, BaselineSubtract

pipe = Pipeline(
    steps=[
        CropByRange((100, 500)),
        SmoothSavGol(window_length=9, polyorder=3),
        BaselineSubtract({"method": "poly", "poly_order": 3}),
    ],
    backend="auto",
)
```

## Supported steps

| Step | Description |
|------|-------------|
| `CropByRange(range)` | Crop spectrum to the given wavenumber/wavelength range |
| `SmoothSavGol(window_length, polyorder)` | Savitzky–Golay smoothing |
| `BaselineSubtract(baseline_spec)` | Baseline subtraction; see below for specification |

## Baseline specification

Baselines are configured as a dictionary passed to `BaselineSubtract`:

**Polynomial baseline:**
```python
BaselineSubtract({"method": "poly", "poly_order": 3})
```

**airPLS baseline:**
```python
BaselineSubtract({"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6})
```

**arPLS baseline:**
```python
BaselineSubtract({"method": "arpls", "lam": 1e5})
```

**asLS baseline:**
```python
BaselineSubtract({"method": "asls", "lam": 1e5, "p": 0.001})
```

**Gaussian baseline** (native-only — not translatable to RamanSPy):
```python
BaselineSubtract({"method": "gaussian"})
```

## Backend

The `backend` argument on `Pipeline` controls which preprocessing implementation is used. See {doc}`backend-behaviour` for the full contract.

## Legacy arguments

For backwards compatibility, `smoothing=True`, `background_remove=True`, and `baseline_method` are still accepted by fitter constructors. Pipeline-based preprocessing is preferred for new workflows.
