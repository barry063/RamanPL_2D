# Canonical notebook examples

The notebooks below are the primary executable examples for RamanPL_2D. They are located in the `example-usage/` directory of the repository and should be run after installing the package (`pip install -e .`).

These notebooks are treated as executable examples, not the API reference. For API details, see {doc}`../api/index`.

## Backend behaviour notebooks

These three notebooks are the canonical references for the preprocessing backend contract:

| Notebook | What it demonstrates |
|----------|----------------------|
| [`example-usage/Ramanfit/Raman_backend_demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_backend_demo.ipynb) | Raman single-spectrum: `native`, `auto`, and `ramanspy` backends on a supported pipeline |
| [`example-usage/Mapping/Raman_mapping_backend_demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Raman_mapping_backend_demo.ipynb) | Raman mapping: backend propagation and export provenance fields |
| [`example-usage/backend/Backend_fallback_cases.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/backend/Backend_fallback_cases.ipynb) | Native fallback: Gaussian baseline and PL workflows remain native-only |

## Interpretable-analysis notebooks

These notebooks cover the stable interpretable-analysis components introduced in v0.5.1–v0.6.0. Three are included in the standard CI notebook smoke test (`tests/test_notebook_smoke.py`); `Clustering_Demo.ipynb` is in the slow suite (`pytest -m slow`) due to its runtime on real WDF data.

| Notebook | What it demonstrates |
|----------|----------------------|
| [`example-usage/Mapping/Feature_Table_Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Feature_Table_Example.ipynb) | `feature_table()` accessor on Raman mapping: per-pixel peak descriptors and QA columns |
| [`example-usage/Mapping/Peak_Proposal_Demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Peak_Proposal_Demo.ipynb) | Classical peak-proposal fallback for failed-fit recovery (v0.5.3) |
| [`example-usage/Mapping/Clustering_Demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Clustering_Demo.ipynb) | PCA + k-means on feature tables using the `[ml]` extra (v0.5.4) — slow suite only |
| [`example-usage/Validation/Validation_v0.6.0_vs_v0.5.0.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Validation/Validation_v0.6.0_vs_v0.5.0.ipynb) | v0.6.0 vs v0.5.0 validation report: fit-quality parity, runtime comparison, failure-mode comparison |

## Additional examples

> **Note:** the notebooks in this section are not included in the CI smoke test. They require real WDF datasets or are intentionally slow; run them locally after acquiring the relevant data files.

| Notebook | What it demonstrates |
|----------|----------------------|
| [`example-usage/Ramanfit/Raman_component.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_component.ipynb) | Raman multi-peak fitting with component inspection |
| [`example-usage/Ramanfit/Raman_warm-start+pipeline.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_warm-start+pipeline.ipynb) | Warm start and pipeline-based preprocessing |
| [`example-usage/PLfit/PL_component.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/PLfit/PL_component.ipynb) | PL multi-peak fitting |
| [`example-usage/PLfit/PL_warm-start.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/PLfit/PL_warm-start.ipynb) | PL fitting with warm start |
| [`example-usage/Mapping/Mapping Raman Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Mapping%20Raman%20Example.ipynb) | Raman mapping end-to-end workflow (area integration under peaks moved to `Area Integration Example.ipynb` in v0.5.5) |
| [`example-usage/Mapping/Mapping PL Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Mapping%20PL%20Example.ipynb) | PL mapping end-to-end workflow (area integration under peaks moved to `Area Integration Example.ipynb` in v0.5.5) |
| [`example-usage/Mapping/Area Integration Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Area%20Integration%20Example.ipynb) | Area integration under fitted peaks for both Raman and PL mapped spectra |
