# RamanPL_2D (version 0.2.0)

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

## Usage Examples

### 1. Mapping.py - PL Mapping Analysis

```python
    import numpy as np
    from Mapping import PLMapping

    # Initialize PL mapping analysis
    pl_map = PLMapping(
        filename='sample_pl.wdf',
        custom_peaks={
            'Exciton': ([1.95, 0.01, 0.1], [2.05, 0.1, 10]),  # [min_loc, min_scale, min_amp], [max...]
            'Trion': ([1.75, 0.01, 0.1], [1.9, 0.1, 10])
        },
        data_range=(1.6, 2.2),  # eV
        step_size=0.3,  # μm
        poly_degree=3,
        baseline_method='poly'
    )

    # Show optical image
    pl_map.show_optical_image()

    # Perform fitting across all points
    pl_map.fit_spectra()

    # Plot exciton position heatmap
    pl_map.plot_heatmap(data_type='exciton_position', 
                    filter_range=(1.95, 2.05),
                    cmap='viridis')

    # Plot spectrum fit at specific coordinates
    pl_map.plot_spectrum_fit(x=10, y=15)
```

### 2. PLfit.py - Single Spectrum PL Analysis

```python
    import numpy as np
    from PLfit import PLfit

    # Generate sample data
    energy = np.linspace(1.7, 2.2, 500)
    spectra = np.random.normal(size=500) + 10*np.exp(-(energy-2.0)**2/(0.02))

    # Initialize and fit
    pl = PLfit(spectra, energy,
            background_remove=True,
            baseline_method='poly',
            smoothing=True,
            smooth_window=15)

    # Update bounds for better fitting
    pl.update_bounds(Exciton=([1.95, 0.01, 1], [2.05, 0.1, 5]))

    # Perform fitting
    params, cov = pl.fit_spectrum()

    # Visualize results
    pl.plot_fit(params, x_lim=[1.8, 2.1])
```

### 3. RamanFit.py - Raman Analysis with Material Library

```python
    from RamanFit import RamanFit
    import numpy as np

    # Sample data (would typically load from file)
    wavenumber = np.linspace(250, 750, 1000)
    spectra = np.random.normal(size=1000) + 100*np.exp(-(wavenumber-350)**2/(50))

    # Initialize with material parameters
    raman = RamanFit(
        spectra=spectra,
        wavenumber=wavenumber,
        materials=['WS2'],  # Load from raman_materials.json
        substrate='SiO2',
        background_remove=True,
        baseline_method='poly',
        smoothing=True
    )

    # Remove unwanted peaks (optional)
    raman.remove_peaks('A1g(Γ)')

    # Perform fitting
    params, cov = raman.fit_spectrum()

    # Visualize results
    raman.plot_fit(params, x_lim=[300, 450], x_ticks=[300, 350, 400, 450])
```

Material Library Format (`raman_materials.json`):

```json
    {
        "WS2": {
            "substrate": false,
            "peaks": {
                "lower_bound": [350, 0, 0, 418, 0, 0],
                "upper_bound": [358, 5, 10, 424, 5, 10],
                "peak_labels": ["E12g(Γ)", "A1g(Γ)"]
            }
        },
        "SiO2": {
            "substrate": true,
            "peaks": {
                "lower_bound": [300, 0, 0],
                "upper_bound": [310, 5, 5],
                "peak_labels": ["SiO2_peak"]
            }
        }
    }
```

- Tips: if you need to add new materials to the current materials peak library, please email me at <hy377@cam.ac.uk>

## 4. Mapping.py - PL Integration Mapping (Without Peak Fitting)

```python
    from Mapping import PL_Integration

    # Initialize integration analysis
    pl_int = PL_Integration(
        filename='sample_pl_map.wdf',
        integration_range=(1.85, 2.05),  # eV range for integration
        step_size=0.3,  # μm
        poly_degree=3,
        background_remove=True
    )

    # Show optical image with mapping area
    pl_int.show_optical_image()

    # Calculate integrated intensities
    pl_int.calculate_integration()

    # Plot heatmap with intensity filtering
    pl_int.plot_integration_heatmap(
        cmap='plasma',
        filter_range=(500, 5000),  # Ignore values <500 and cap at 5000
        x_range=(10, 30),  # Show subset of X coordinates
        y_range=(5, 25)    # Show subset of Y coordinates
    )

    # Plot spectrum at specific coordinate with/without background
    pl_int.plot_spectrum(x=15, y=20)

    # Expected Output:
    # - Heatmap showing intensity distribution in specified region
    # - Spectrum plot comparing raw and background-removed data
    # - Console output showing calculated integration values
```

---

## To-do

- Add more materials to the `raman_materials.json` library, such as hBN etc.
- Add a manuals basic science behind curve-fitting
- Add more background substraction options such as shirley, tougard, etc.
- Add a batch processing and batch visualisation tools or functionalities
- Fix bugs (as always)

## License

This project is licensed for **non-commercial academic use only**.  
Commercial use is prohibited without prior written permission.  
See the [LICENSE](LICENSE) file for details.

## Contact

For issues, questions, or collaboration ideas:  
Hao Yu – <hy377@cam.ac.uk>
