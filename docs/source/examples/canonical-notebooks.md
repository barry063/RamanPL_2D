# Canonical notebook examples

The notebooks below are the primary executable examples for RamanPL_2D. They are located in the `example-usage/` directory of the repository and should be run after installing the package (`pip install -e ./src`).

These notebooks are treated as executable examples, not the API reference. For API details, see {doc}`../api/index`.

## Backend behaviour notebooks

These three notebooks are the canonical references for the preprocessing backend contract:

| Notebook | What it demonstrates |
|----------|----------------------|
| [`example-usage/Ramanfit/Raman_backend_demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_backend_demo.ipynb) | Raman single-spectrum: `native`, `auto`, and `ramanspy` backends on a supported pipeline |
| [`example-usage/Mapping/Raman_mapping_backend_demo.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Raman_mapping_backend_demo.ipynb) | Raman mapping: backend propagation and export provenance fields |
| [`example-usage/backend/Backend_fallback_cases.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/backend/Backend_fallback_cases.ipynb) | Native fallback: Gaussian baseline and PL workflows remain native-only |

## Additional examples

| Notebook | What it demonstrates |
|----------|----------------------|
| [`example-usage/Ramanfit/Raman_component.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_component.ipynb) | Raman multi-peak fitting with component inspection |
| [`example-usage/Ramanfit/Raman_warm-start+pipeline.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Ramanfit/Raman_warm-start+pipeline.ipynb) | Warm start and pipeline-based preprocessing |
| [`example-usage/PLfit/PL_component.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/PLfit/PL_component.ipynb) | PL multi-peak fitting |
| [`example-usage/PLfit/PL_warm-start.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/PLfit/PL_warm-start.ipynb) | PL fitting with warm start |
| [`example-usage/Mapping/Mapping Raman Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Mapping%20Raman%20Example.ipynb) | Raman mapping end-to-end workflow |
| [`example-usage/Mapping/Mapping PL Example.ipynb`](https://github.com/barry063/RamanPL_2D/blob/main/example-usage/Mapping/Mapping%20PL%20Example.ipynb) | PL mapping end-to-end workflow |
