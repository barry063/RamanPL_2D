# RamanPL_2D

**RamanPL_2D** is a Python toolkit for the analysis and visualisation of **Raman** and **photoluminescence (PL)** spectra in two-dimensional materials.  
It provides tools for extracting **peak positions, peak intensities / heights, and FWHM**, and supports **single-spectrum fitting, batch analysis, and spectral mapping**.

The package is designed to support reproducible spectroscopy analysis workflows for 2D materials research.


## Features

### Spectral analysis

- Import and process Raman and PL spectra from **`.txt` and `.wdf`**
- Compatible with **Renishaw WiRE** exported data
- Single-spectrum fitting using two peak models:
  - **Lorentzian** (default; compatible with materials libraries)
  - **Pseudo-Voigt (pVoigt)** — linear combination of Lorentzian and Gaussian

### Flexible peak definitions

- Library-based peak definitions via `materials` and `substrate`
- Fully user-defined peaks using `custom_peaks`
- Remove unwanted peaks via `remove_peaks`
- Consistent behaviour across:
  - single-spectrum fitting
  - batch fitting
  - mapping

### Batch processing

Batch workflows allow automated analysis of many spectra:

- automated fitting across multiple spectra
- extraction of peak parameters (position, FWHM, intensity / peak height)
- summary statistics **per peak**
- export to `.csv` / `.txt`

### Mapping analysis

- Heatmaps of fitted parameters:
  - peak intensity / peak height
  - peak position
  - FWHM
- Raman-specific derived maps:
  - **A1g − E2g peak separation**
  - **E2g / A1g intensity ratio**
- Heatmaps of **integrated spectral intensity**
- Mapping fit diagnostics:
  - residual maps
  - residual distribution inspection
  - bound-sticking summaries
  - optional compact or disabled per-pixel diagnostics for production runs

### Preprocessing

- Modular preprocessing with `Pipeline`
- Shared preprocessing support across:
  - single-spectrum fitting
  - batch workflows
  - mapping workflows
- Supported preprocessing operations include:
  - crop by range
  - Savitzky–Golay smoothing
  - baseline subtraction (`poly`, `gaussian`, `asLS`, `arPLS`, `airPLS`)

### Performance and robustness

- Adaptive mapping multistart fitting:
  - cheap first-pass fit
  - retry only when needed
- Faster mapping preprocessing for common workflows:
  - vectorised Savitzky–Golay smoothing
  - vectorised Gaussian baseline subtraction
  - batched polynomial baseline subtraction
- Vectorised peak summation for cheaper repeated model evaluations during optimisation

### Visualisation

- Raw vs fitted spectra overlay
- Waterfall plots for spectral collections
- Dynamic inspection of spectral fitting results

For features such as `pipeline`, adaptive mapping fits, and diagnostics control, please check the [demonstration](#demonstration) section and example notebooks in [`example-usage/`](example-usage/).

For development roadmap and future plans, see the [development roadmap](#development-roadmap) section below.

---

## Source Code Structure

```text
RamanPL_2D/
├── src/
│   ├── setup.py
│   ├── install.ipynb
│   └── ramanpl/
│       ├── __init__.py
│       ├── baselineAPI.py                  # Baseline subtraction kernels (asLS/arPLS/airPLS/poly/gaussian); shared DtD cache
│       ├── batch.py                        # Batch workflows for Raman / PL spectra
│       ├── dataImporter.py                 # WDF / TXT import for single spectra and mapping data
│       ├── exporter.py                     # CSV / TXT export helpers and metadata provenance
│       ├── operation.py                    # Spectrum / map arithmetic operations
│       ├── peak_models.py                  # Peak model definitions (Lorentzian, Gaussian, pVoigt)
│       ├── preprocessing.py                # Preprocessing pipeline framework and benchmark pipeline builders
│       ├── raman_materials.json            # Built-in Raman materials library
│       ├── schema.py                       # Canonical schema validators (modality, axis kind, baseline spec)
│       │
│       ├── single_fit/                     # Single-spectrum fitting (Raman and PL)
│       │   ├── __init__.py
│       │   ├── _single_fit_core.py         # Shared fitting utilities (multistart, export helpers, RMSE)
│       │   ├── RamanFit.py                 # Raman single-spectrum fitter
│       │   └── PLfit.py                    # PL single-spectrum fitter
│       │
│       ├── mapping/                        # Mapping workflows (Raman and PL)
│       │   ├── __init__.py
│       │   ├── _diagnostics.py             # Per-pixel fit diagnostics helpers
│       │   ├── _fit_utils.py               # Shared mapping fit utilities
│       │   ├── _image.py                   # Heatmap and image plotting helpers
│       │   ├── _io.py                      # Mapping I/O, coordinates, and layout helpers
│       │   ├── _pl_mapping.py              # PL mapping workflow
│       │   ├── _preprocess.py              # Mapping preprocessing mixin (cache, dispatch, provenance)
│       │   └── _raman_mapping.py           # Raman mapping workflow
│       │
│       ├── integration/                    # Optional RamanSPy backend
│       │   ├── __init__.py
│       │   ├── ramanspy_adapter.py         # RamanPL ↔ RamanSPy array conversion helpers
│       │   ├── ramanspy_bridge.py          # Backend resolution and fallback decision logic
│       │   └── ramanspy_translate.py       # Translation of preprocessing pipelines to RamanSPy
│       │
│       # Compatibility façades (preserve legacy import paths)
│       ├── _single_fit_core.py             # re-exports ramanpl.single_fit._single_fit_core
│       ├── RamanFit.py                     # re-exports ramanpl.single_fit.RamanFit
│       ├── PLfit.py                        # re-exports ramanpl.single_fit.PLfit
│       └── Mapping.py                      # re-exports ramanpl.mapping classes
```

## Change log

See [CHANGELOG](CHANGELOG) for details on recent updates and new features.


## Getting Started

For users new to Python or Visual Studio Code (VS Code), the following steps will help you get started:

### 1. Install Python

Download and install the latest version of Python from: <https://www.python.org/downloads/>

### 2. Install Visual Studio Code

Download and install VS Code from: <https://code.visualstudio.com/>

### 3. Set Up Python in VS Code

- Install the official Python extension by Microsoft.

- Follow the official VS Code tutorial: Getting Started with Python in [VS Code](https://code.visualstudio.com/docs/python/python-tutorial)

### 4. Clone the Repository

Press the `code` button on the webpage to find your best way to clone th repository, enter the bash command in the terminal. An example of clone it via http can be like:

```bash
git clone https://github.com/barry063/RamanPL_2D.git
```

This will clone the open version of `RamanPL_2D` codes.

### 5. Install Dependencies

After you clone it, first go to the directory of your local `RamanPL_2D`, enter the bash command:

 ```bash
cd RamanPL_2D
```

Then in the same directory, enter the bash command:

 ```bash
pip install -r requirements.txt
```

This will automatically check all the required python packages required and install them in your local environment. 

### 6. **(Optional)** Installing the Library Locally (for VSCode & Jupyter Notebook)

To use the `RamanPL_2D` toolkit in your own scripts or Jupyter notebooks, you can install the package locally using either of the following methods:

#### Option 1: Install as Editable Package (Recommended)

This method allows you to import the package from anywhere, and source-code changes will be reflected without reinstalling.

1. Navigate to the source folder:

```bash
cd "path\to\RamanPL_2D\src"
```

2. Install in editable mode:

```bash
pip install -e .
```

3. You can then import it in Python as usual:

```python
from ramanpl import RamanFit
from ramanpl import PLfit
```

#### Option 2: Run `install.ipynb` for installation

Go to the `src` folder, click open the `install.ipynb` jupyter-notebook file. **DON'T** move the `install.ipynb` out of the `src` folder!
Simply run all the codes in the `install.ipynb` to run the installation in the jupyter-notebook

**Important**: all the `.ipynb` in the example usage folders assume you have installed the package. So if you couldn't run the example nicely, maybe install the packages first.

#### Option 3: Add Folder to PYTHONPATH or sys.path (Manual)

If you prefer not to install the package, you can manually add the source folder to your Python path.

In your script or jupyter-notebook:

```python
import sys
sys.path.append(r"path\RamanPL_2D\src")

from ramanpl import RamanFit
```

**This approach is transient – it must be repeated each time the Python kernel restarts unless automated via environment variables or startup scripts.**



### 7. Run Example Notebook

- Open `example_analysis.ipynb` in the `example-usage/` folder using VS Code or Jupyter.
- Run the cells to see the toolkit in action.


### 8. Optional RamanSPy backend

RamanPL_2D supports **RamanSPy as an optional preprocessing backend** for supported **Raman** workflows.

For a local source install from the `src/` directory, use:

Install with:

```bash
pip install -e .[ramanspy]
```

If RamanSPy is not installed, preprocessing falls back to the native implementation when `preprocessing_backend="auto"` is used.


---


# Demonstration

## Preprocessing pipelines

Version **v0.3.4** introduced a modular preprocessing pipeline framework for Raman and PL spectra.

Typical preprocessing steps include:

- spectral cropping
- smoothing
- baseline subtraction

A pipeline consists of ordered preprocessing steps:

```python
from ramanpl.preprocessing import Pipeline
```

Each step modifies a `SpectralDataset` object and passes the result to the next step.

Example steps currently included:

| Step               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `CropByRange`      | Crop spectra to selected spectral window                         |
| `SmoothSavGol`     | Savitzky–Golay smoothing                                         |
| `BaselineSubtract` | Background subtraction (`poly` / `airPLS` / `arPLS` / `AsLS` / `Gaussian`) |

---

### Legacy preprocessing arguments

For backward compatibility, the following arguments still work:

```python
smoothing=True
background_remove=True
baseline_method="poly"
```

However, pipeline-based preprocessing is recommended for new workflows.

## Preprocessing & Backend Contract

### Backend selection

Pass `preprocessing_backend` to any fitter or mapping class:

| Value | Behaviour |
| ------------ | -------------------------------------------------------------------- |
| `"native"` | Always uses the built-in Python/NumPy implementation |
| `"auto"` | Uses RamanSPy when available and supported; otherwise falls back to native |
| `"ramanspy"` | Forces RamanSPy; raises an error if unavailable or unsupported |

**RamanSPy-capable workflows:** Raman (cm⁻¹) single-spectrum and mapping preprocessing, when using a fully translatable pipeline.  
**Native-only workflows:** PL spectra (any axis), and any pipeline step not yet translatable to a RamanSPy operation (e.g. Gaussian baseline).

#### Backend support matrix

| Workflow | Backend requested | Backend resolved |
| -------- | ----------------- | ---------------- |
| Raman + supported pipeline | `"native"` | `"native"` |
| Raman + supported pipeline | `"auto"` | `"ramanspy"` (if installed) |
| Raman + supported pipeline | `"ramanspy"` | `"ramanspy"` |
| Raman + unsupported step | `"auto"` | `"native"` (fallback) |
| Raman + unsupported step | `"ramanspy"` | raises `NotImplementedError` |
| PL + any backend | `"auto"` | `"native"` |

### Provenance fields in exports

All export files include canonical provenance keys:

| Key | Meaning |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| `preprocessing_backend_requested` | The backend the user requested (`"native"`, `"auto"`, or `"ramanspy"`) |
| `preprocessing_backend_resolved` | The backend that was actually used |
| `preprocessing_backend_fallback_reason` | Reason for fallback (only present when a fallback occurred) |

`preprocessing_backend_fallback_reason` is **absent** when no fallback occurred — do not assert on its presence unless you expect a fallback.

### Performance note

Native preprocessing uses vectorised NumPy operations and is typically faster for large mapping cubes. RamanSPy may offer advantages for iterative baselines (airPLS/arPLS) where its internal convergence logic differs. Parity between the two backends is tolerance-verified in `tests/test_mapping_backend_parity.py`.

## Baseline specification

Baseline algorithms are now configured using a dictionary specification:

Example of `airPLS` baseline specification:

```python
baseline_spec = {
    "method": "airpls",
    "lam": 1e6,
    "niter": 50,
    "tol": 1e-6,
}
```

Example polynomial baseline:

```python
baseline_spec = {
    "method": "poly",
    "poly_order": 3,
}
```

The legacy argument `poly_degree` was removed in v0.4.7. Use `baseline_spec={"method": "poly", "poly_order": N}` instead.

## Adaptive mapping fitting and diagnostics (v0.3.9)

Version v0.3.9 improves mapping efficiency in three main ways:

- adaptive multistart fitting
- faster cube-level preprocessing for common workflows
- configurable diagnostics storage

## Adaptive multistart fitting

Mapping fits can now use a cheap first pass and only retry with more expensive initialisations when needed.

Example:

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
        retry_on_bound_hit=False,
        retry_rmse_gate=0.10,
        diagnostics="light",
    )
)
```

### How it works

- Fitting is performed in **peak-normalised space** for stability.
- The fitter generates `n_starts` initial guesses using one of:
  - `p0_strategy="midpoint"`: midpoint of bounds (baseline behaviour)
  - `p0_strategy="random"`: uniform random within bounds
  - `p0_strategy="jitter"`: Gaussian perturbations around the current `p0`, clipped to bounds
- The best candidate is selected using RMSE (and optionally a penalty term for width “inflation” toward its upper bound).

### Available strategies:

| Strategy | Description                                |
| -------- | ------------------------------------------ |
| midpoint | midpoint of parameter bounds               |
| random   | random uniform sampling within bounds      |
| jitter   | Gaussian perturbation around initial guess |


#### Example

```python
raman_map.fit_spectra(
    warm_start=True,
    fit_spectrum_kwargs=dict(
        n_starts=10,
        p0_strategy="jitter",
        random_state=0
    )
)
```

## Diagnostics levels

Three diagnostics modes are available for mapping fits:

| Mode    | Behaviour                                               |
| ------- | ------------------------------------------------------- |
| `full`  | stores full per-pixel diagnostics including bound masks |
| `light` | stores compact QA summaries only                        |
| `none`  | disables per-pixel diagnostics storage                  |

Example:

```python 
pl_map.fit_spectra(
    fit_spectrum_kwargs=dict(
        diagnostics="none"
    )
)
```

`fit_summary()` still works in diagnostics="none" mode using the residual map, but detailed bound-sticking diagnostics are not available.

## Verifying bound-sticking (QA check)

After fitting, you can summarise mapping fit quality using:

```python
rep = raman_map.fit_summary()
```

This reports:

- fit success rate
- RMSE statistics
- failure reasons, when diagnostics are available
- bound-sticking summaries, when diagnostics are available


---

## Backend behaviour examples

The following notebooks are the canonical references for backend behaviour in v0.4.8:

| Notebook | What it proves |
| -------- | -------------- |
| [`example-usage/Ramanfit/Raman_backend_demo.ipynb`](example-usage/Ramanfit/Raman_backend_demo.ipynb) | Raman single-spectrum: `native`, `auto`, and `ramanspy` backends on a supported pipeline |
| [`example-usage/Mapping/Raman_mapping_backend_demo.ipynb`](example-usage/Mapping/Raman_mapping_backend_demo.ipynb) | Raman mapping: backend propagation and export provenance fields |
| [`example-usage/backend/Backend_fallback_cases.ipynb`](example-usage/backend/Backend_fallback_cases.ipynb) | Unsupported/native-fallback: Gaussian baseline and PL workflows remain native-only |

---

## New Features in v0.4.x: RamanSPy integration

### Preprocessing backend selection

Preprocessing supports three backend modes:

- `native` — always use the built-in preprocessing implementation
- `auto` — use RamanSPy when available, supported for the input, and the full preprocessing pipeline is currently translatable; otherwise fall back to native
- `ramanspy` — force RamanSPy preprocessing and raise an error if unavailable or unsupported

### Current support

RamanSPy preprocessing support in the current build is limited to:

- **Raman workflows**
- **Raman shift axis (`cm^-1`)**
- the following translated preprocessing steps:
  - `CropByRange`
  - `SmoothSavGol`
  - `BaselineSubtract` with:
    - `poly`
    - `asls`
    - `airpls`
    - `arpls`

The following remain native-only for now:

- PL preprocessing workflows
- `BaselineSubtract(method="gaussian")`

#### Single-spectrum example

```python
from ramanpl import RamanFit

raman_fit = RamanFit.RamanFit(
    spectra=spectra,
    wavenumber=wavenumber,
    custom_peaks={
        "P1": ([210, 2, 0], [235, 30, 10]),
        "P2": ([325, 2, 0], [360, 40, 10]),
    },
    smoothing=True,
    background_remove=True,
    baseline_method={"method": "poly", "poly_order": 3},
    preprocessing_backend="auto",
)
```

#### Custom preprocessing pipeline example

```python
from ramanpl.preprocessing import Pipeline, CropByRange, SmoothSavGol, BaselineSubtract

pipe = Pipeline(
    steps=[
        CropByRange((120, 480)),
        SmoothSavGol(window_length=9, polyorder=3),
        BaselineSubtract({"method": "poly", "poly_order": 3}),
    ],
    backend="auto",
)
```

In `auto` mode, RamanPL_2D records the resolved backend in preprocessing metadata. In mapping exports, the requested and resolved backend are both written into export metadata for provenance.

#### Forced RamanSPy example

```python
from ramanpl.preprocessing import Pipeline, SmoothSavGol

pipe = Pipeline(
    steps=[
        SmoothSavGol(window_length=9, polyorder=3),
    ],
    backend="ramanspy",
)
```

### Mapping backend benchmarking (v0.4.3)

Version v0.4.3 introduces a reproducible benchmark harness
(`benchmarks/benchmark_mapping_preprocessing.py`) for Raman mapping cube preprocessing. The
harness compares native and RamanSPy backend runtime and memory usage across six pipeline
configurations (crop, Savitzky–Golay, polynomial baseline, AsLS, airPLS, arPLS) on three
synthetic datasets (3×4, 10×12, and 20×24 pixels). Cube consistency and parity tests validate
axis ordering, shape invariants, and adapter round-trip correctness for both backends;
v0.4.4 completes `RamanBatch` backend integration: `preprocessing_backend="native" | "auto" | "ramanspy"` now propagates cleanly through batch fitting, export metadata records both requested and resolved backend (with `export_kind: "batch_fit"`), and all existing plotting/table behaviour is preserved. Example:

```python
b = RamanBatch(files, materials=["MoS2"], preprocessing_backend="auto")
b.fit()
b.export("raman_fit.txt", wide=True)  # header includes preprocessing_backend_requested/resolved
```

---

# TO-DO

# TO-DO

## Development Roadmap

> Roadmap update (late v0.4.x → v0.5.0)  
> Development remains focused on RamanSPy integration as an optional preprocessing backend for supported Raman workflows.  
> The immediate priorities are to complete performance characterisation, finalise the preprocessing/backend API, add executable example notebooks, optimise the native fallback path, and harden release validation for packaging and CI.  
> Advanced analysis features and machine-learning-assisted workflows remain deferred until preprocessing behaviour, backend provenance, and release stability are mature.

### v0.4.x — RamanSPy integration

| Version | Scope | Details |
|--------|------|--------|
| **v0.4.0** ✅ | Backend infrastructure | Add optional RamanSPy dependency.<br>Introduce internal adapter layer (`integration/ramanspy_adapter`).<br>Implement Spectrum / mapping-cube conversion.<br>Add preprocessing backend selector (`native / ramanspy / auto`).<br>Record backend in metadata. |
| **v0.4.1** ✅ | Pipeline translation and stabilisation | Translate `preprocessing.Pipeline` → RamanSPy for supported Raman preprocessing steps.<br>Support: crop, Savitzky–Golay, selected baselines (`poly`, `asls`, `airpls`, `arpls`).<br>Preserve native fallback for unsupported steps and workflows.<br>Stabilise backend propagation through single-spectrum and mapping preprocessing. |
| **v0.4.2** ✅ | Validation and documentation | Add regression tests for `native / auto / ramanspy` backend behaviour.<br>Add Raman vs PL backend-compatibility checks.<br>Update notebooks and README examples.<br>Verify export metadata and preprocessing provenance. |
| **v0.4.3** ✅ | Mapping benchmarking and performance review | Benchmark Raman mapping preprocessing: native vs RamanSPy.<br>Measure conversion overhead and memory behaviour.<br>Confirm axis ordering, cube consistency, and representative-dataset correctness.<br>Use benchmark results to guide the next integration and optimisation steps. |
| **v0.4.4** ✅ | Batch integration | Propagate backend into `RamanBatch`.<br>Ensure consistent export metadata.<br>Maintain existing plotting and table behaviour.<br>Keep unsupported workflows on the native path. |
| **v0.4.5** ✅ | API cleanup and hardening | Improve backend error messages and resolution consistency across single-spectrum, mapping, and batch workflows.<br>Reduce remaining duplication in mapping/batch internals.<br>Consolidate preprocessing/export helper paths.<br>Unify backend provenance serialisation and fallback reporting. |

### v0.4.x+ — completion and release preparation

| Version | Scope | Details |
|--------|------|--------|
| **v0.4.6** ✅ | Performance profiling and baseline optimisation | Pre-compute `D^T D` once per `(n, diff_order)` key and cache across calls in `asLS`, `arPLS`, `airPLS` — eliminates `O(niter)` sparse matrix products per spectrum.<br>Added `benchmarks/benchmark_baseline_kernels.py` microbenchmark for native baseline methods; extended mapping benchmark with 5 cube sizes (up from 3).<br>Added parity, cache-correctness, and batch regression tests.<br>Scientific behaviour and public API unchanged. |
| **v0.4.7** ✅ | API cleanup & deprecation removal | Finalise the preprocessing/backend contract before broader documentation and release hardening.<br>Remove deprecated parameters and legacy pathways.<br>Finalise preprocessing schema (`baseline_spec` / pipeline spec).<br>Freeze backend provenance/export field names and resolution behaviour.<br>Ensure example notebooks and release tests target the stabilised API. |
| **v0.4.8** ✅ | Executable examples and backend demonstrations | Add user-facing notebooks that document supported backend behaviour clearly.<br>Add a supported Raman single-spectrum notebook comparing `native / auto / ramanspy`.<br>Add a supported Raman mapping notebook showing backend propagation and export provenance.<br>Add an unsupported/native-fallback notebook demonstrating clear fallback behaviour for unsupported workflows.<br>Link these notebooks from the README as the canonical backend-behaviour examples. |
| **v0.4.9** ✅ | Native baseline optimisation | Eliminated per-iteration sparse matrix allocation in asLS / arPLS / airPLS kernels: W and Z diagonal updated in-place; RHS and convergence buffers pre-allocated. Native iterative baselines promoted to the mapping cube fast path — SpectralDataset construction per pixel eliminated. Benchmark gains: 1D kernels 22–71% faster; mapping (10×12 cube) asLS ~50% faster. All parity tests pass at atol=1e-10; public API unchanged. |
| **v0.4.10** | Release validation and packaging gates | Prepare the project for reliable pip/GitLab/GitHub-based release workflows.<br>Add CI jobs for unit tests, integration tests, and optional RamanSPy extras install.<br>Add package build/install smoke tests for clean environments.<br>Add notebook smoke execution for selected `example-usage/` workflows.<br>Add benchmark smoke runs for release validation, while keeping full performance comparisons advisory rather than strict CI gates. |
| **v0.5.0** | Raman preprocessing milestone release | Mark supported Raman preprocessing integration as stable and release-ready.<br>Declare supported Raman preprocessing workflows complete for `native / auto / ramanspy`.<br>Freeze the supported backend contract and documentation for the milestone release.<br>Ensure provenance/export behaviour, examples, and packaging validation are all aligned.<br>Defer broader analysis extensions until after the preprocessing/backend milestone is complete. |

---

### Notes

- RamanSPy is currently used as an **optional preprocessing backend** only.
- Integration is limited to **Raman workflows (cm⁻¹ axis)**; PL workflows remain on the native backend.
- Existing APIs (`Pipeline`, `RamanFit`, `Mapping`, `Batch`) remain **backward compatible** during v0.4.x.
- `BaselineSubtract(method="gaussian")` remains native-only at this stage.
- Current development priority is **validation, documentation, backend hardening, and behaviour stabilisation** before broader RamanSPy feature expansion.

---

# License

This project is licensed for **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for details.

# Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>
