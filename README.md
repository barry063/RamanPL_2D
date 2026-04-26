# RamanPL_2D

**RamanPL_2D** is a Python toolkit for the analysis and visualisation of **Raman** and **photoluminescence (PL)** spectra in two-dimensional materials. It extracts peak positions, intensities, and FWHM values and supports single-spectrum fitting, batch processing, and spectral mapping. The package is designed for reproducible spectroscopy workflows and exports provenance metadata alongside fitted results. An optional [RamanSPy](https://github.com/baettigph/ramanspy) backend is supported for Raman preprocessing pipelines.

---

## Key features

- Single-spectrum and batch fitting (Lorentzian, pseudo-Voigt)
- Spectral mapping with heatmaps and derived maps (peak separation, intensity ratios)
- Modular preprocessing pipelines with `native`, `auto`, and `ramanspy` backend modes
- Import from `.wdf` (Renishaw WiRE) and `.txt`
- Export to `.csv` / `.txt` with full provenance metadata
- CI-validated packaging, documented public API, and canonical example notebooks

---

## Installation

Base install (no optional dependencies):

```bash
pip install -e .
```

With optional RamanSPy preprocessing backend:

```bash
pip install -e ".[ramanspy]"
```

Requires Python ≥ 3.9.

---

## Quickstart

```python
from ramanpl import RamanFit

raman_fit = RamanFit.RamanFit(
    spectra=spectra,
    wavenumber=wavenumber,
    materials=["MoS2"],
    smoothing=True,
    background_remove=True,
    baseline_method={"method": "poly", "poly_order": 3},
    preprocessing_backend="auto",
)
raman_fit.fit()
raman_fit.export("results.txt")
```

See [`example-usage/`](example-usage/) for full notebooks.

---

## Documentation

For local documentation build:

```bash
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.

Key pages:

- [Installation](docs/source/installation.md)
- [Quickstart](docs/source/quickstart.md)
- [Backend behaviour](docs/source/user-guide/backend-behaviour.md)
- [API reference](docs/source/api/index.rst)
- [Canonical notebook examples](docs/source/examples/canonical-notebooks.md)

---

## Backend support summary

| Workflow | `native` | `auto` | `ramanspy` |
| -------- | -------- | ------ | ---------- |
| Raman + supported pipeline | native | ramanspy (if installed) | ramanspy |
| Raman + unsupported step | native | native (fallback) | raises error |
| PL (any pipeline) | native | native | native |

Supported pipeline steps for RamanSPy: `CropByRange`, `SmoothSavGol`, `BaselineSubtract` with `poly`, `asls`, `airpls`, `arpls`. Gaussian baseline and all PL workflows remain native-only.

See [backend behaviour docs](docs/source/user-guide/backend-behaviour.md) for full details.

---

## Development and validation

Release validation commands and pre-tag checklist are in [`RELEASE.md`](RELEASE.md).

CI runs on GitHub Actions (`.github/workflows/ci.yml`) and GitLab (`.gitlab-ci.yml`):

- base tests (no RamanSPy required)
- RamanSPy extras tests
- package build and clean-install smoke
- notebook smoke (with RamanSPy)
- benchmark smoke
- documentation build

---

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

---

## Licence

BSD-3-Clause. See [`LICENSE`](LICENSE) for details.

---

## Contact

Hao Yu — <yuhao19980603@gmail.com>  
Issues and pull requests: <https://github.com/barry063/RamanPL_2D/issues>
