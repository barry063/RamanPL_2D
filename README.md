# RamanPL_2D

**RamanPL_2D** is a Python-based toolkit designed for the analysis and visualisation of Raman and photoluminescence (PL) spectra in two-dimensional materials. It facilitates the extraction of peak positions, intensities, and full width at half maximum (FWHM) from spectral data, offering an intuitive interface for researchers working with 2D materials.

## Features

- Compatible analysis tools with most Renishaw Raman spectroscopy systems
- Import and process Raman and PL spectra from `.txt` and `.wdf`
- Single-spectrum fitting of 2 models for Raman and PL：
  - **Lorentzian** (default; compatible with materials library)
  - **Pseudo-Voigt (pVoigt)** — linear combination of Lorentzian and Gaussian
- **Batch processing** of multiple spectra (RamanBatch / PLBatch):
  - automated fitting across many spectra
  - consistent peak parameter extraction (position, FWHM, intensity)
  - summary statistics of fitted parameters **by peak**
  - export of batch results to `.csv` / `.txt`
- **Flexible peak definition system**:
  - library-based default peaks using `materials` and `substrate`
  - fully user-defined `custom_peaks` (replaces defaults)
  - selective suppression of unwanted peaks via `remove_peaks`
  - consistent behaviour across single fit, batch, and mapping
- Visualisation of raw and fitted spectra (overlay & waterfall plots)
- **Sanity checks for fitting quality**:
  - normalised residual calculation
  - residual distribution diagnostics
  - dynamic spectrum fitting view
- Raman and PL **mapping analysis**:
  - heatmaps of integrated intensity within selected ranges
  - heatmaps of fitted peak intensity, position, and FWHM
  - heatmaps of Raman **A1g − E2g** peak separation
  - heatmaps of **E2g / A1g** peak ratio
- Arithmetic processing of spectra and mapping data for flexible operations

Batch processing and mapping workflows share the same peak-definition logic as single-spectrum fitting. Users may rely on built-in material libraries for
rapid analysis, or fully override peak definitions using `custom_peaks` and `remove_peaks` for advanced or non-standard systems.

### Change log

See [CHANGELOG](CHANGELOG)

## Repository Structure

```bash
RamanPL_2D/
    ├── example-usage/ # Sample spectral data files and demonstrated usage of python codes by jupyter-notebook (`.ipynb`files)
    │ ├── Mapping/      # PL, Raman data mapping using `Mapping.py`
    │ ├── multi-plot/   # Demonstrate batch fitting, plotting, and parameter analysis using `batch.py`
    │ ├── PLfit/        # PL component curve fitting using `PLfit.py`
    │ └── Ramanfit/     # Raman spectrum and component peak fitting using `RamanFit.py` and `raman_materials.json`
    ├── src/                # Source code for data processing and analysis DON'T CHANGE THE FOLDER STRUCTURE!
    │ ├── ramanpl/          # header of the pacakage name, so you should use "from ramanpl import RamanFit" forspecific module
    │ │ ├── __init__.py               # For package installation only, header to indicate this is a folder of python packages
    │ │ ├── RamanFit.py               # Class modules for Raman spectra fitting and plotting, to be used with raman_materials.json
    │ │ ├── raman_materials.json      # Class modules for Raman spectra fitting and plotting, to be used with raman_materials.json
    │ │ ├── PLfit.py                  # Class modules for Raman spectra fitting and plotting
    │ │ ├── baselinAPI.py             # Helper codes for Raman/PL background subtration
    │ │ ├── dataImporter.py           # Helper codes for importing .wdf/.txt data files
    │ │ ├── peak_model.py             # Helper codes for calling lorentzian/pesudo-voigt model when fitting
    │ │ ├── operation.py              # Classs modules for making math operations of multiple data files.
    │ │ ├── batch.py                  # Batch fitting, statistical analysis, plotting, and export of multiple spectra
    │ │ ├── exporter.py               # Classs modules for exporting the fitted data and parameters into .csv files
    │ │ └── Mapping.py                # Mapping of Raman, PL and integration of spectra
    │ ├── install.ipynb     # A jupyter-notebook run to install  our package
    │ └── setup.py          # For package installation only, include some required python packages for using
    ├── requirements.txt    # List of required Python packages
    └── README.md           # Project documentation
```

---

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


------

## Multi-start fitting (v0.3.0)

Multi-start fitting reduces “bound-sticking” artefacts in multi-peak Lorentzian models by running several fits from different initial guesses (`p0` trials) and selecting the best result (lowest score / RMSE with optional penalties).

### How it works
- Fitting is performed in **peak-normalised space** for stability.
- The fitter generates `n_starts` initial guesses using one of:
  - `p0_strategy="midpoint"`: midpoint of bounds (baseline behaviour)
  - `p0_strategy="random"`: uniform random within bounds
  - `p0_strategy="jitter"`: Gaussian perturbations around the current `p0`, clipped to bounds
- The best candidate is selected using RMSE (and optionally a penalty term for width “inflation” toward its upper bound).

### Example (Raman mapping)

```python
from ramanpl import Mapping

custom_peaks = {
    "E2g":   ([348, 0, 0], [360, 10, 10]),
    "A1g":   ([418, 0, 0], [424, 10, 10]),
    "Si":    ([518, 0, 0], [525, 20, 10]),
    "2LA(M)":([340, 0, 0], [350,  5,  5]),
}

raman_map = Mapping.RamanMapping(
    "Mapping Raman Sample.wdf",
    custom_peaks=custom_peaks,
    data_range=(320, 560),
    normalize=True,            # display scaling only
    background_remove=True,
    step_size=0.5,
)

_ = raman_map.fit_spectra(
    warm_start=True,
    fit_spectrum_kwargs=dict(
        n_starts=10,
        p0_strategy="jitter",
        random_state=0,
        # optional robustness knobs if exposed in your build:
        # width_penalty=0.25,
        # prefer_nonbound=True,
        # rmse_tie_tol=1e-3,
    )
)
```

## Verifying bound-sticking (QA check)

After fitting, you can quantify how often fitted parameters sit on their bounds. This is a practical sanity check for over-restrictive bounds or model misspecification.

### Generic bound-hit check from fitted parameters
```python
import numpy as np

def bounds_from_custom_peaks(custom_peaks):
    lb, ub = [], []
    for v in custom_peaks.values():
        lb.extend(v[0]); ub.extend(v[1])
    return np.asarray(lb, float), np.asarray(ub, float)

def bound_hit_report(mapping_obj, *, rtol=1e-6, atol=1e-12):
    lb, ub = bounds_from_custom_peaks(mapping_obj.custom_peaks)
    P = np.asarray(mapping_obj.fitted_params, float)  # shape [Y, X, 3*n_peaks]
    P2 = P.reshape(-1, P.shape[-1])

    valid = np.isfinite(P2).all(axis=1)
    P2 = P2[valid]

    hit_upper = np.isclose(P2, ub, rtol=rtol, atol=atol)
    hit_lower = np.isclose(P2, lb, rtol=rtol, atol=atol)

    centres_u = hit_upper[:, 0::3].mean(axis=0)
    widths_u  = hit_upper[:, 1::3].mean(axis=0)
    amps_u    = hit_upper[:, 2::3].mean(axis=0)

    names = list(mapping_obj.custom_peaks.keys())
    print("Peaks:", names)
    print("Upper-bound hit fraction per peak (centre):", centres_u)
    print("Upper-bound hit fraction per peak (width) :", widths_u)
    print("Upper-bound hit fraction per peak (amp)   :", amps_u)
    print("Overall upper-bound hit fraction centre:", float(hit_upper[:,0::3].mean()))
    print("Overall upper-bound hit fraction width :", float(hit_upper[:,1::3].mean()))
    print("Overall upper-bound hit fraction amp   :", float(hit_upper[:,2::3].mean()))

    return dict(
        peaks=names,
        upper_centre=centres_u, upper_width=widths_u, upper_amp=amps_u,
        lower_centre=hit_lower[:,0::3].mean(axis=0),
        lower_width =hit_lower[:,1::3].mean(axis=0),
        lower_amp   =hit_lower[:,2::3].mean(axis=0),
    )

# usage:
# rep = bound_hit_report(raman_map)
```

-------

## To-do

- v0.3.5: Add `pipeline` like modules for data pre-processing, fitting, and post-processing to streamline common workflows (e.g. mapping with multi-start fitting and bound-hit reporting).
- v0.4.0: Integrate methods with `RamanSPy` for better standardisation.

## License

This project is licensed for **non-commercial academic use only**.  
Commercial use is prohibited without prior written permission.  
See the [LICENSE](LICENSE) file for details.

## Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>