# RamanPL_2D (version 0.2.5)

**RamanPL_2D** is a Python-based toolkit designed for the analysis and visualisation of Raman and photoluminescence (PL) spectra in two-dimensional materials. It facilitates the extraction of peak positions, intensities, and full width at half maximum (FWHM) from spectral data, offering an intuitive interface for researchers working with 2D materials.

## Features

- Compatible analysis tools with most ReinShaw Raman Spectroscopy equipment
- Import and process Raman and PL spectra from `.txt`and `.wdf`
- Peak fitting using Lorentzian models
- Visualisation of raw and fitted spectra
- Heatmaps of integrated spectra data in a selected range
- Heatmaps of the data processed with a `filter_range` and `data_range` selection.
- Heatmaps of intensity at specific wavenumber/energy by fitted spectra
- Auto-calculation of important data for 2D materials: **A1g - E2g peak difference** and **FWHM** of peaks
- Heatmaps of Raman spectrum **A1g - E2g Peak difference** and **E2g/A1g peak ratio**
- Sanity check: normalised residual calculation and distribution, dynamical spectrum fitting view

### Change log

**Version 0.2.5 (2025-12-27):**

    1. Added a few new examples in PLfit and Ramanfit example foldders to demonstrate background subtraction using various fitting methods.
    2. Updated `PLfit.py` and `RamanFit.py` to include a new `baseline_method` parameter in the constructor, allowing users to choose between 'poly' (polynomial fitting) and 'gaussian' (Gaussian fitting) for background removal.
    3. Created a new module `baselineAPI.py` to handle baseline correction methods, which is now imported and used in both `PLfit.py`,`RamanFit.py` and `Mapping.py`.
    4. Deprecating the previous `poly_degree` and `gaussian_sigma` parameters in favour of the new `baseline_method` parameter. Warnings are issued if the old parameters are used.
    5. Added new background removal options in `baselineAPI.py`: `airpls`, `asls`, `arpls`, which are now utilised in the fitting classes.

*All changes are backward compatible with previous version (0.2.4).*

**Version 0.2.4 (2025-12-26):**

    1. Updated `RamanFit.py` to fix a bug where the fitting was incorrectly performed on unnormalised data when `normalize` was set to False. Now, fitting is always done on normalised data, and the `normalize` option only affects display and output scaling. 
    2. Updated `Mapping.py` to ensure that Raman mapping fitting is performed on normalised data regardless of the `normalize` setting. The `normalize` option now only influences the display and output scaling of heatmaps and spectra. And updated specific intensity heatmap plotting, ratio heatmap plotting, and distance heatmap plotting methods to reflect this change.
    3. Updated example notebooks `Mapping Raman Example.ipynb` and `Mapping Raman txt Example.ipynb` to reflect the changes in fitting behaviour and clarify the purpose of the `normalize` option.
    4. Optimise the fitting speed in `Mapping.py` by improving the warm-start logic and reducing redundant computations.

*All changes are backward compatible with previous version (0.2.3).*

**Version 0.2.3 (2025-12-25):**

    1. Changed `normalize` option in `Mapping.py` to control whether to display normalised data for heatmap display. This option does not affect the fitting process.
    2. Added `warm_start` and `reset_on_fail` options in `Mapping.py`'s `fit_spectra()` method to improve fitting stability.
    3. Updated `plot_residual_distribution()`, `plot_heatmap()`, `plot_spectrum()_fit` method in `Mapping.py` to visualise the distribution of fitting residuals for quality check.
    4. Added `export_p0()` method in `PLfit.py` to export the fitted parameters after fitting for further analysis.
    5. Updated Mapping examples in `Mapping PL Example.ipynb`, `PL_component.ipynb`, and `Mapping PL txt Example.ipynb` to reflect the changes in `normalize` option, added residual distribution plotting, `p0_pkg` for better `p0` guess in the example.

*All changes are backward compatible with previous version (0.2.0).*

**Version 0.2.0 (2025-6-6):**

    1. Move source code to `ramanpl` folder, create package installation manuals for local library installation and usage
    2. Modified the jupyter-notebook codes in example for local library usage.

**Version 0.1.5 (2025-6-5):**

    1. Added *Gr* and *2L-Gr* for graphene/bilayer-graphene library file `raman_materials.json`
    2. Added new sample bilayer graphene data (`Raman Sample 532nm 2L-Graphene.txt`) into example folders
    3. Modify the doc-strings in `Mapping.py`

## Repository Structure

```bash
RamanPL_2D/
    ├── example-usage/ # Sample spectral data files and demonstrated usage of python codes by jupyter-notebook (`.ipynb`files)
    │ ├── Mapping/     # PL, Raman data mapping using `Mapping.py`
    │ ├── PLfit/       # PL component curve fitting using `PLfit.py`
    │ └── Ramanfit/    # Raman spectrum and component peak fitting using `RamanFit.py` and `raman_materials.json`
    ├── src/                # Source code for data processing and analysis DON'T CHANGE THE FOLDER STRUCTURE!
    │ ├── ramanpl/          # header of the pacakage name, so you should use "from ramanpl import RamanFit" forspecific module
    │ │ ├── __init__.py               # For package installation only, header to indicate this is a folder of python packages
    │ │ ├── RamanFit.py               # Class modules for Raman spectra fitting and plotting, to be used with raman_materials.json
    │ │ ├── raman_materials.json      # Class modules for Raman spectra fitting and plotting, to be used with raman_materials.json
    │ │ ├── PLfit.py                  # Class modules for Raman spectra fitting and plotting
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

```bash
git clone https://github.com/barry063/RamanPL_2D.git
cd RamanPL_2D
```

### 5. Install Dependencies

 ```bash
pip install -r requirements.txt
```

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

## To-do

- (v0.2.6) Add peak arithmatic processing (subtraction / addition) and change `read_lines` to `data_range` that reflect the actuall wavenumber/energy range
- (v0.2.7) Add a batch processing and batch visualisation tools or functionalities
... ...
- (v0.3.0+) Add Monte-Carlo peak-fitting functionalities so that best-fit is easier to get.

## License

This project is licensed for **non-commercial academic use only**.  
Commercial use is prohibited without prior written permission.  
See the [LICENSE](LICENSE) file for details.

## Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>
