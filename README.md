# RamanPL_2D

**RamanPL_2D** is a Python toolkit for the analysis and visualisation of **Raman** and **photoluminescence (PL)** spectra in two-dimensional materials. It extracts peak positions, intensities, and FWHM values and supports single-spectrum fitting, batch processing, and spectral mapping. The package is designed for reproducible spectroscopy workflows and exports provenance metadata alongside fitted results. An optional [RamanSPy](https://github.com/baettigph/ramanspy) backend is supported for Raman preprocessing pipelines.

---

## Key features

* Single-spectrum and batch fitting (Lorentzian, pseudo-Voigt)
* Spectral mapping with heatmaps and derived maps (peak separation, intensity ratios)
* Modular preprocessing pipelines with `native`, `auto`, and `ramanspy` backend modes
* Import from `.wdf` (Renishaw WiRE) and `.txt`
* Export to `.csv` / `.txt` with full provenance metadata
* CI-validated packaging, documented public API, and canonical example notebooks

---

## Installation

Base install (no optional dependencies):

```
pip install -e .
```

With optional RamanSPy preprocessing backend:

```
pip install -e ".[ramanspy]"
```

Requires Python ≥ 3.9.

---

## Quickstart

```
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

See [`example-usage/`](example-usage) for full notebooks.

---

## Documentation

For local documentation build:

```
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.

Key pages:

* [Installation](docs/source/installation.md)
* [Quickstart](docs/source/quickstart.md)
* [Backend behaviour](docs/source/user-guide/backend-behaviour.md)
* [API reference](docs/source/api/index.rst)
* [Canonical notebook examples](docs/source/examples/canonical-notebooks.md)

---

## Backend support summary

| Workflow | `native` | `auto` | `ramanspy` |
| --- | --- | --- | --- |
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

### v0.5.x–v0.6.0 — Interpretable analysis and ML-assisted workflows roadmap

> Post-v0.5.0 development focuses on improving fitting efficiency and mapping-scale analysis while preserving physically interpretable peak fitting, explicit preprocessing provenance, and stable export behaviour.
> Earlier builds (v0.5.1–v0.5.3) extend existing deterministic infrastructure: per-pixel quality columns in mapping exports, generalised peak descriptors, and classical signal-processing aids for failed-fit recovery. Machine-learning components appear only from v0.5.4 onwards, as an optional `[ml]` extra, and are confined to unsupervised analysis of fitted descriptors. Final scientific quantities remain traceable to deterministic fitting wherever possible.

| Version | Scope | Details |
|---|---|---|
| **v0.5.1** | Mapping-fit benchmark and per-pixel QA columns | - Add `benchmarks/benchmark_mapping_fit.py` measuring `fit_spectra()` runtime across cube sizes, `n_starts` settings, and warm-start on/off. <br> - Establish a recorded "before" performance baseline ahead of any efficiency work. <br> - Extend the per-pixel mapping export row to include `rmse`, `ok`, `n_starts`, and `n_params_at_bounds` columns (data already collected in `residual_map` and `fit_diagnostics_map`). <br> - Add an optional `long=True` flag to `export_fit_map` producing one-row-per-pixel-per-peak format for downstream pandas/scikit-learn workflows. <br> - No changes to fitting algorithms or scientific output values. |
| **v0.5.2** | Generalised peak descriptors and feature-table accessor | - Generalise the existing Raman A1g/E2g ratio and separation logic into a peak-pair-agnostic descriptor builder usable by both `RamanMapping` and `PLMapping`. <br> - Add `Mapping.feature_table()` returning a `pandas.DataFrame` of per-pixel descriptors (peak position, FWHM, peak height, configurable ratios and separations, plus the QA columns added in v0.5.1). <br> - Mirror the same accessor for batch and single-fit outputs for API consistency. <br> - No new optional dependencies; pandas is already a base dependency. |
| **v0.5.3** | Classical peak-proposal aid for failed-fit recovery | - Add `single_fit/initialisation.py` providing peak-proposal helpers based on `scipy.signal.find_peaks` (prominence, width, derivative cues). <br> - Wire proposals into `mapping/_fit_utils.py` as an additional fallback path used **only** when the existing adaptive multistart and warm-start paths have failed. <br> - This is classical signal processing, not machine learning; the module name reflects that. <br> - Acceptance criterion: lower failed-fit count on hard pixels (overlapping peaks, weak peaks, broad backgrounds) without systematic drift in fitted peak position, FWHM, or peak height on previously successful pixels. <br> - Benchmarked against the v0.5.1 baseline. |
| **v0.5.4** | Optional `[ml]` extra: unsupervised mapping clustering | - Introduce `ramanpl.ml.clustering` providing PCA and k-means on feature tables produced by v0.5.2. <br> - Add `[ml]` optional dependency on `scikit-learn`; base install remains unaffected. <br> - Operate on fitted descriptors, not on raw spectra, to keep clustering interpretable and avoid learning instrument or substrate artefacts. <br> - Provide one notebook demonstrating domain discovery on a representative mapping dataset. <br> - No claims of automatic material identification. |
| **v0.5.5** | Consolidation pause: API freeze, docs, dependency hygiene | - Review the public surface introduced in v0.5.1–v0.5.4 and freeze the feature-table schema, descriptor naming, and `[ml]` extra boundary. <br> - Update API reference, user-guide pages, and changelog to reflect the new exports and accessor. <br> - Confirm base install still functions without `scikit-learn` and that `ramanpl.ml` raises a clear error when the extra is missing. <br> - No new algorithms in this build. |
| **v0.6.0** | Stable interpretable-analysis milestone | - Declare the first stable interpretable-analysis milestone built on v0.5.1–v0.5.5. <br> - Stable components: per-pixel QA-augmented mapping export, generalised peak descriptors, `feature_table()` accessors, classical peak-proposal aid for failed-fit recovery, and optional unsupervised clustering on fitted descriptors. <br> - Maintain deterministic physical fitting as the final authority for all reported peak parameters. <br> - Provide validation examples comparing fit quality, runtime, and failure-mode behaviour against the v0.5.0 baseline. <br> - Supervised classification (e.g. layer-number labels) remains deferred to a later v0.7.x cycle, conditional on the availability of curated, labelled, multi-instrument datasets. |

---

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

---

## Licence

BSD-3-Clause. See [`LICENSE`](LICENSE) for details.

---

## Contact

Hao Yu — <hy377@cam.ac.uk>  
Issues and pull requests: <https://github.com/barry063/RamanPL_2D/issues>
