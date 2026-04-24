# Backend behaviour

This page is the authoritative reference for the preprocessing backend contract.

## Overview

Every fitter and mapping class accepts a `preprocessing_backend` argument (or inherits it from a `Pipeline`). The backend controls which implementation runs the preprocessing steps.

| Value | Behaviour |
|-------|-----------|
| `"native"` | Always use the built-in NumPy/SciPy implementation |
| `"auto"` | Use RamanSPy when available and the full pipeline is translatable; otherwise fall back to native |
| `"ramanspy"` | Force RamanSPy; raise `NotImplementedError` if unavailable or unsupported |

## Support matrix

| Workflow | Backend requested | Backend resolved |
|----------|-------------------|-----------------|
| Raman + supported pipeline | `"native"` | `"native"` |
| Raman + supported pipeline | `"auto"` | `"ramanspy"` (if installed) |
| Raman + supported pipeline | `"ramanspy"` | `"ramanspy"` |
| Raman + unsupported step | `"auto"` | `"native"` (fallback) |
| Raman + unsupported step | `"ramanspy"` | raises `NotImplementedError` |
| PL + any backend | `"auto"` | `"native"` |
| PL + any backend | `"ramanspy"` | raises `NotImplementedError` |

## RamanSPy-translatable steps

The following `Pipeline` steps are currently translatable to RamanSPy for Raman (cm⁻¹) workflows:

- `CropByRange`
- `SmoothSavGol`
- `BaselineSubtract` with methods: `poly`, `asls`, `airpls`, `arpls`

## Native-only paths

The following always use the native implementation regardless of `backend`:

- PL workflows (any axis unit)
- `BaselineSubtract(method="gaussian")`
- Any pipeline step not yet in the translatable set

## Provenance fields in exports

All export files record the backend that was actually used:

| Key | Meaning |
|-----|---------|
| `preprocessing_backend_requested` | The value the user passed (`"native"`, `"auto"`, or `"ramanspy"`) |
| `preprocessing_backend_resolved` | The backend that actually ran |
| `preprocessing_backend_fallback_reason` | Reason for fallback — **only present when a fallback occurred** |

`preprocessing_backend_fallback_reason` is absent when no fallback occurred. Do not assert on its presence unless you expect a fallback.

## Performance note

Native preprocessing uses vectorised NumPy operations and is typically faster for large mapping cubes. RamanSPy may offer different convergence behaviour for iterative baselines (airPLS/arPLS). Backend parity is verified in `tests/test_mapping_backend_parity.py`.

## Canonical notebook examples

See {doc}`../examples/canonical-notebooks` for notebooks that demonstrate all three backend modes.
