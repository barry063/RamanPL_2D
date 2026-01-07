# RamanPL_2D (version 0.2.7.1)

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
- Arithmetic processing of the mapping data or single point data for flexible operation.
- Sanity check: normalised residual calculation and distribution, dynamical spectrum fitting view

### Change log

See [CHANGELOG](CHANGELOG)

## Repository Structure

```bash
RamanPL_2D/
    ├── example-usage/ # Sample spectral data files and demonstrated usage of python codes by jupyter-notebook (`.ipynb`files)
    │ ├── Mapping/      # PL, Raman data mapping using `Mapping.py`
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
    │ │ ├── operation.py              # Classs modules for making math operations of multiple data files.
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


## To-do

- (v0.2.8) Add output functionalities to export fitted parameters and heatmap data to CSV or Excel files for further analysis.
- (v0.2.9) Add a batch processing and batch visualisation tools or functionalities
- (v0.3.0+) Add Monte-Carlo peak-fitting functionalities so that best-fit is easier to get.

## License

This project is licensed for **non-commercial academic use only**.  
Commercial use is prohibited without prior written permission.  
See the [LICENSE](LICENSE) file for details.

## Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>
