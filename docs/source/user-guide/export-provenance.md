# Export and provenance

RamanPL_2D exports results to `.txt` or `.csv` files with a metadata header. The header records the configuration and backend used, so results are reproducible and auditable.

## Why provenance matters

When `preprocessing_backend="auto"` is used, the resolved backend may differ between environments (e.g. RamanSPy installed vs not installed). The export header always records which backend actually ran, making results interpretable independently of the environment.

## Canonical provenance fields

All export files include:

| Key | Meaning |
|-----|---------|
| `preprocessing_backend_requested` | The value passed by the user (`"native"`, `"auto"`, or `"ramanspy"`) |
| `preprocessing_backend_resolved` | The backend that actually ran preprocessing |
| `preprocessing_backend_fallback_reason` | Reason a fallback from RamanSPy to native occurred — **absent when no fallback** |

Additional fields depend on workflow type:

| Key | Meaning |
|-----|---------|
| `export_kind` | Workflow type: `"single_fit"`, `"batch_fit"`, `"mapping_fit"` |
| `spectrum_type` | `"raman"` or `"pl"` |
| `x_axis_quantity` | e.g. `"wavenumber"` or `"wavelength"` |
| `x_axis_unit` | e.g. `"cm^-1"` or `"nm"` |
| `baseline_spec` | Baseline specification dict |
| `peak_model` | Peak profile used, e.g. `"lorentzian"` or `"pvoigt"` |
| `peak_labels` | List of fitted peak names |

## TXT vs CSV

Both `.txt` and `.csv` formats are supported. The metadata header uses `#`-prefixed comment lines. The data table follows the header.

## Mapping vs batch exports

**Mapping export:** one row per pixel, columns for peak parameters and spatial coordinates.

**Batch export:** one row per spectrum file (wide) or one row per peak per spectrum (long).

The `export_kind` field distinguishes these.

## Interpreting fallback reason

`preprocessing_backend_fallback_reason` is set when `auto` resolved to native instead of RamanSPy. Common values:

- `"ramanspy_not_installed"` — RamanSPy was not found in the environment
- `"pipeline_not_translatable"` — one or more pipeline steps have no RamanSPy translation
- `"modality_not_supported"` — PL workflows are always native

## See also

- {doc}`backend-behaviour` — full backend resolution rules
- {doc}`../api/index` — exporter API reference
