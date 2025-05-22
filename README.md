# PLMapping

Python tools for in-depth analysis of photoluminescence (PL) and Raman mapping data acquired from `.wdf` files (Renishaw WiRE format). Supports automated background removal, peak fitting (Lorentzian), residual analysis, and visualisation of peak position/intensity heatmaps.

---

## 📦 Features

- Polynomial or Gaussian baseline correction
- Support for multiple peak fitting via custom Lorentzian models
- Residual-based quality check
- Heatmap plotting for peak position and intensity
- Flexible control over fitting ranges and filters
- Integration-ready for Jupyter notebooks

---

## 📁 Project Structure

```python
Project/
├── Improved_PLMapping.py # Core class for PL mapping and analysis
├── example_notebook.ipynb # Example Jupyter workflow
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore # Optional Git exclusions
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Use the provided Jupyter notebook as a guide for using the `Mapping` class:

```python
from Mapping import PLMapping

custom_peaks = {
    'Exciton': ([1.8, 0, 0], [1.88, 10, 10]),
    'Trion':   ([1.76, 0, 0], [1.82, 10, 10])
}

pl_map = PLMapping(
    filename='example.wdf',
    custom_peaks=custom_peaks,
    step_size=0.5,
    poly_degree=2,
    background_remove=True,
    smoothing=True,
    baseline_method='poly',
    data_range=(0, 1012)
)

pl_map.fit_spectra()
pl_map.plot_heatmap(data_type='exciton_position')
```

## Requirements

- Python 3.7+
- Jupyter Notebook (recommended)
- Renishaw .wdf files (requires renishawWiRE module)

## 📌 To Do

- Add support for additional baseline methods (e.g. asymmetric least squares)
- Integrate Raman and PL mapping in a unified framework
- Export heatmap and fit data to CSV or HDF5
- Improve interactive diagnostics and batch visualisation

## 📜 License

This project is licensed for **non-commercial academic use only**.  
Commercial use is prohibited without prior written permission.  
See the [LICENSE](LICENSE) file for details.

## 🙋 Contact

For issues, questions, or collaboration ideas:  
Hao Yu – [hy377@cam.ac.uk]
