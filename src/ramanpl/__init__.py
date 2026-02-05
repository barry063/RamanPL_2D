"""
ramanpl: Raman / PL fitting and mapping utilities.

Public API:
- Single spectra: PLfit, RamanFit
- Mapping: PLMapping, RamanMapping, PL_Integration, Raman_Integration
- Arithmetic operations: Spectrum, ArithmeticSpectrum
- Utilities: DataImporter, BaselineAPI
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Single-spectrum fitters
    "PLfit",
    "RamanFit",

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

    # Submodules (optional convenience)
    "Mapping",
    "operation",
    "dataImporter",
    "baselineAPI",
    "exporter",

]

# Optional: version string (set manually)
__version__ = "0.2.9.5"


# -----------------------
# Lazy attribute loading
# -----------------------
_LAZY_MAP = {
    # Single-spectrum fitters
    "PLfit": ("ramanpl.PLfit", "PLfit"),
    "RamanFit": ("ramanpl.RamanFit", "RamanFit"),

    # Mapping
    "PLMapping": ("ramanpl.Mapping", "PLMapping"),
    "RamanMapping": ("ramanpl.Mapping", "RamanMapping"),
    "PL_Integration": ("ramanpl.Mapping", "PL_Integration"),
    "Raman_Integration": ("ramanpl.Mapping", "Raman_Integration"),

    # Arithmetic
    "Spectrum": ("ramanpl.operation", "Spectrum"),
    "ArithmeticSpectrum": ("ramanpl.operation", "ArithmeticSpectrum"),

    # Utilities
    "DataImporter": ("ramanpl.dataImporter", "DataImporter"),
    "BaselineAPI": ("ramanpl.baselineAPI", "BaselineAPI"),

    # Optional: allow `import ramanpl; ramanpl.Mapping.PLMapping`
    "Mapping": ("ramanpl", "Mapping"),
    "operation": ("ramanpl", "operation"),
    "dataImporter": ("ramanpl", "dataImporter"),
    "baselineAPI": ("ramanpl", "baselineAPI"),
    "exporter": ("ramanpl", "exporter"),

}


def __getattr__(name: str) -> Any:
    if name in ("Mapping", "operation", "dataImporter", "baselineAPI"):
        return import_module(f"ramanpl.{name}")

    if name not in _LAZY_MAP:
        raise AttributeError(f"module 'ramanpl' has no attribute '{name}'")

    mod_name, attr_name = _LAZY_MAP[name]
    mod = import_module(mod_name)
    return getattr(mod, attr_name)
