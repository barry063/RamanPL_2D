"""
ramanpl: Raman / PL fitting and mapping utilities.

Public API:
- Single spectra: PLfit, RamanFit
- Mapping: PLMapping, RamanMapping, PL_Integration, Raman_Integration
- Arithmetic operations: Spectrum, ArithmeticSpectrum
- Utilities: DataImporter, BaselineAPI, peak_models, exporter
- Multiple data processing: Batch
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Single-spectrum fitters (module-style, backwards compatible)
    "PLfit",
    "RamanFit",

    # Single-spectrum subpackage
    "single_fit",

    # Mapping
    "PLMapping",
    "RamanMapping",
    "PL_Integration",
    "Raman_Integration",

    # Arithmetic operations
    "Spectrum",
    "ArithmeticSpectrum",

    # Utilities
    "DataImporter",
    "BaselineAPI",

    # Optional integration helpers
    "integration",

    # Submodules (optional convenience)
    "Mapping",
    "operation",
    "dataImporter",
    "baselineAPI",
    "exporter",
    "Batch",
    "batch",
    "peak_models",
    "preprocessing",
]

__version__ = "0.5.4"

# Attributes that should resolve to a class/object directly
_LAZY_ATTR_MAP = {
    # Mapping classes
    "PLMapping": ("ramanpl.Mapping", "PLMapping"),
    "RamanMapping": ("ramanpl.Mapping", "RamanMapping"),
    "PL_Integration": ("ramanpl.Mapping", "PL_Integration"),
    "Raman_Integration": ("ramanpl.Mapping", "Raman_Integration"),

    # Arithmetic
    "Spectrum": ("ramanpl.operation", "Spectrum"),
    "ArithmeticSpectrum": ("ramanpl.operation", "ArithmeticSpectrum"),

    # Shared utilities
    "DataImporter": ("ramanpl.dataImporter", "DataImporter"),
    "BaselineAPI": ("ramanpl.baselineAPI", "BaselineAPI"),
}

# Names that should resolve to modules for backwards compatibility
_MODULE_EXPORTS = {
    "RamanFit": "ramanpl.RamanFit",
    "PLfit": "ramanpl.PLfit",
    "single_fit": "ramanpl.single_fit",
    "Mapping": "ramanpl.Mapping",
    "operation": "ramanpl.operation",
    "dataImporter": "ramanpl.dataImporter",
    "baselineAPI": "ramanpl.baselineAPI",
    "exporter": "ramanpl.exporter",
    "batch": "ramanpl.batch",
    "Batch": "ramanpl.batch",
    "peak_models": "ramanpl.peak_models",
    "preprocessing": "ramanpl.preprocessing",
    "integration": "ramanpl.integration",
}


def __getattr__(name: str) -> Any:
    # Backwards-compatible module-style exports
    if name in _MODULE_EXPORTS:
        return import_module(_MODULE_EXPORTS[name])

    # Direct class/object exports
    if name in _LAZY_ATTR_MAP:
        mod_name, attr_name = _LAZY_ATTR_MAP[name]
        mod = import_module(mod_name)
        return getattr(mod, attr_name)

    raise AttributeError(f"module 'ramanpl' has no attribute '{name}'")