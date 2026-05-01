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

### v0.5.x–v0.6.0 — Interpretable ML-assisted analysis roadmap

> Post-v0.5.0 development focuses on lightweight, interpretable machine-learning-assisted workflows.
> The priority is to improve fitting efficiency and mapping-scale analysis while preserving physically interpretable peak fitting, explicit preprocessing provenance, and stable export behaviour.
> ML components should assist peak proposal, feature extraction, clustering, and workflow triage; final scientific quantities should remain traceable to deterministic fitting wherever possible.

| Version | Scope | Details |
|---|---|---|
| **v0.5.1** | ML-ready spectral feature tables | - Add a lightweight `ramanpl.ml.features` module. <br> - Convert single-fit, batch, and mapping outputs into structured descriptor tables. <br> - Include interpretable descriptors: peak position, FWHM, peak height, peak ratios, peak separations, residual/RMSE, fit success, bound-sticking flags, backend/provenance fields. <br> - Support CSV export of feature tables for external analysis. <br> - Add deterministic tests using synthetic spectra and fitted mock outputs. <br> - No trained ML model yet; this build prepares the data foundation. |
| **v0.5.2** | Peak proposal and fitting initialisation | - Add lightweight peak proposal utilities for estimating candidate peak centres, widths, and presence/absence before curve fitting. <br> - Use signal-processing and classical methods first, e.g. local maxima, prominence, derivative/curvature cues, and optional `scikit-learn`-based helpers. <br> - Feed proposed peaks into existing Lorentzian/pseudo-Voigt fitting as improved initial guesses. <br> - Preserve deterministic fitting as the final source of peak parameters. <br> - Benchmark against current warm-start/adaptive fitting on single-spectrum and mapping workflows. <br> - Acceptance criterion: faster fitting or fewer failed fits without systematic drift in fitted peak position/FWHM/height. |
| **v0.5.3** | Mapping-level clustering and region discovery | - Add unsupervised analysis of fitted feature tables for Raman/PL maps. <br> - Support PCA and lightweight clustering on fitted descriptors rather than raw black-box spectra. <br> - Generate interpretable region labels, cluster maps, outlier maps, and per-cluster summary statistics. <br> - Keep clustering optional and dependency-light through an `ml` extra. <br> - Provide one notebook showing domain discovery on fitted mapping descriptors. <br> - Avoid claiming automatic material identification at this stage. |
| **v0.5.4** | ML-assisted fitting benchmark and validation | - Consolidate benchmark tests for baseline/current fitting, peak-proposal-assisted fitting, and mapping-level workflows. <br> - Measure runtime, fit success rate, residual/RMSE, peak-position stability, FWHM stability, and peak-height stability. <br> - Add regression tests to ensure ML-assisted proposals do not change final scientific behaviour unexpectedly. <br> - Add failure-mode analysis: weak peaks, overlapping peaks, noisy spectra, broad background, and low-SNR spectra. <br> - Keep benchmark thresholds advisory rather than strict CI timing gates. |
| **v0.5.5** | API cleanup and ML module stabilisation | - Review the `ramanpl.ml` public API after v0.5.1–v0.5.4. <br> - Consolidate feature-table schema, naming conventions, export fields, and optional dependency handling. <br> - Remove duplicated helper logic between fitting, mapping, and ML modules. <br> - Improve error messages for missing optional ML dependencies. <br> - Freeze the first stable ML-assisted workflow contract. <br> - No new algorithms unless required to stabilise existing behaviour. |
| **v0.5.6** | Interpretable quality metrics and fit triage | - Add quality scoring for spectra and fitted pixels using interpretable metrics. <br> - Include flags for low SNR, high residual, failed fit, suspicious FWHM, peak overlap, boundary-hitting parameters, and poor baseline correction. <br> - Use these metrics to triage mapping pixels before or after fitting. <br> - Provide quality maps and summary tables for mapping workflows. <br> - Keep the scoring transparent and rule-based first; any learned score should be optional and explainable. |
| **v0.5.7** | Lightweight model persistence and reproducibility | - Add optional support for saving/loading lightweight fitted preprocessing or ML-assist configurations. <br> - Store model/config metadata: package version, feature schema version, preprocessing provenance, training data description, and dependency versions. <br> - Support simple serialisation for classical models only, e.g. `joblib` for `scikit-learn` models. <br> - Avoid large neural-network dependencies by default. <br> - Add tests that saved configurations reproduce the same predictions or proposals. |
| **v0.5.8** | Supervised classification pilot, experimental | - Add an experimental supervised-learning interface only after feature-table and validation infrastructure is stable. <br> - Target narrow tasks first, e.g. layer-number class, material family label, or map-region class, depending on available labelled data. <br> - Require explicit labels, train/test split, and validation metrics. <br> - Report confusion matrix and class-wise performance, not only accuracy. <br> - Clearly mark the interface as experimental and dataset-dependent. <br> - Do not present this as universal material identification. |
| **v0.5.9** | Pre-v0.6 review, documentation, and examples | - Review all ML-assisted workflows for scientific validity, API consistency, and documentation clarity. <br> - Clean notebooks and examples so they demonstrate supported workflows only. <br> - Add documentation pages for feature extraction, peak proposal, clustering, quality metrics, and supervised-classification limitations. <br> - Re-check optional dependency boundaries: base install should not require ML dependencies. <br> - Update README, docs, and changelog to distinguish stable, optional, and experimental ML features. |
| **v0.6.0** | Interpretable ML-assisted analysis milestone | - Declare the first stable ML-assisted analysis milestone. <br> - Stable components: feature tables, peak proposal for fitting initialisation, mapping clustering, quality metrics, and reproducible lightweight model/config handling. <br> - Experimental components may remain clearly marked, especially supervised material classification. <br> - Maintain deterministic physical fitting as the final authority for reported peak parameters. <br> - Provide full validation examples showing speed, robustness, and scientific parity against non-ML workflows. |

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
