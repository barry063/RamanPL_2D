# RamanPL_2D

**RamanPL_2D** is a Python toolkit for the analysis and visualisation of **Raman and photoluminescence (PL) spectra** in two-dimensional materials.  
It provides tools for extracting **peak positions, intensities, and FWHM**, performing **single-spectrum fitting, batch analysis, and spectral mapping**.

The package is designed to support reproducible analysis workflows for 2D material spectroscopy experiments.


## Features

### Spectral analysis

- Import and process Raman and PL spectra from **`.txt` and `.wdf`**
- Compatible with **Renishaw WiRE exported data**
- Single-spectrum fitting using two peak models:

  - **Lorentzian** (default; compatible with materials library)
  - **Pseudo-Voigt (pVoigt)** — linear combination of Lorentzian and Gaussian

### Flexible peak definitions

- Library-based peak definitions via `materials` and `substrate`
- Fully user-defined peaks using `custom_peaks`
- Remove unwanted peaks via `remove_peaks`
- Consistent behaviour across:
  - single spectrum fitting
  - batch fitting
  - mapping

### Batch processing

Batch workflows allow automated analysis of many spectra:

- automated fitting across multiple spectra
- extraction of peak parameters (position, FWHM, intensity)
- summary statistics **per peak**
- export to `.csv` / `.txt`

### Mapping analysis

- Heatmaps of fitted parameters:
  - peak intensity
  - peak position
  - FWHM
- Raman specific derived maps:
  - **A1g − E2g peak separation**
  - **E2g / A1g intensity ratio**
- Heatmaps of **integrated spectral intensity**

### Visualisation

- Raw vs fitted spectra overlay
- Waterfall plots for spectral collections
- Dynamic inspection of spectral fitting results

### Quality diagnostics

- residual analysis
- residual distribution inspection
- dynamic spectrum fitting view

For features like `pipeline` and `multistart`, please check [code examples](#demostration).

For more details on the features and example of spefici use, please refer to the [example usage notebooks](example-usage/).

---

## Repository Structure

```bash
RamanPL_2D/
    ├── example-usage/
    │ ├── Mapping/
    │ ├── multi-plot/
    │ ├── PLfit/
    │ └── Ramanfit/
    │
    ├── src/
    │ ├── ramanpl/
    │ │ ├── __init__.py
    │ │ ├── RamanFit.py
    │ │ ├── PLfit.py
    │ │ ├── single_fit/
    │ │ │ ├── __init__.py
    │ │ │ ├── RamanFit.py
    │ │ │ ├── PLfit.py
    │ │ │ └── _single_fit_core.py
    │ │ ├── preprocessing.py
    │ │ ├── raman_materials.json
    │ │ ├── baselineAPI.py
    │ │ ├── dataImporter.py
    │ │ ├── peak_models.py
    │ │ ├── operation.py
    │ │ ├── batch.py
    │ │ ├── exporter.py
    │ │ └── Mapping.py
    │
    │ ├── install.ipynb
    │ └── setup.py
    │
    ├── requirements.txt
    └── README.md
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

This method allows you to import your package from anywhere, and changes in your source code will be automatically reflected without needing to reinstall.

1. Navigate to the Source Folder

```bash
    cd "path to code\RamanPL_2D\src"
```

2. Install Using `pip`

```bash
pip install .
```

3. This will install the library in **editable** mode. You can now import it in Python like:

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

---

# Demostration

## Preprocessing Pipelines (after v0.3.4)

Version **0.3.4** introduces a modular **spectral preprocessing pipeline framework**.

This enables standardised preprocessing workflows for Raman and PL spectra before fitting.

The pipeline design is inspired by modern data-processing frameworks and allows users to construct reproducible analysis chains.

Typical preprocessing steps include:

- spectral cropping
- smoothing
- baseline subtraction
- future extensions (normalisation, filtering, etc.)

---

### Pipeline architecture

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
| `BaselineSubtract` | Background subtraction (poly / airPLS / arPLS / AsLS / Gaussian) |

### Legacy preprocessing arguments

For backward compatibility, the following arguments still work:

```python
smoothing=True
background_remove=True
baseline_method="poly"
baseline_kwargs={"poly_order": 3}
```

In v0.3.5, pipeline-based preprocessing is supported across single, batch, and mapping workflows.
For backwards compatibility, some legacy preprocessing metadata fields may still appear in exports.
A later cleanup release will simplify metadata headers when `preprocessing=Pipeline(...)` is used explicitly.

However, **pipeline-based preprocessing is recommended for new workflows.** The legacy arguments will be deprecated in a future release (By version v0.4.0).

## Baseline specification

Baseline algorithms are now configured using a dictionary specification:

Example of `airPLS` baseline specification:
```python
baseline_spec = {   "method": "airpls",
                    "lam": 1e6,
                    "niter": 50,
                    "tol": 1e-6,
                }
```

Example polynomial baseline:

```python
baseline_spec = {   "method": "poly",
                    "poly_order": 3,
                }
```
The legacy argument poly_degree is deprecated and will be removed in a future release (By release v0.4.0).

## Multi-start fitting (v0.3.0)

Multi-start fitting helps reduce bound-sticking artefacts in multi-peak fits.

The fitter performs several fits from different starting parameters and selects the best solution.

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

## Verifying bound-sticking (QA check)

After fitting, you can quantify how often fitted parameters sit on their bounds. This is a practical sanity check for over-restrictive bounds or model misspecification.

### Generic bound-hit check from fitted parameters

```python
rep = bound_hit_report(raman_map)
```

---

## TO-DO

| Version | Planned feature                                   |
| ------- | ------------------------------------------------- |
| v0.3.7  | Optimise mapping modularisation                   |
| v0.3.8  | Optimise schema and polish                        |
| v0.3.9  | Optimise mapping and fitting efficiency           |
| v0.4.0+ | integrate standardised analysis with RamanSPy     |
| v0.5.0+ | integrate with machine learning abilities         |


## License

This project is licensed for **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for details.

## Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>
