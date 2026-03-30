"""Compatibility façade for mapping workflows.

Public API remains available from:
    from ramanpl.Mapping import ...
"""

from .mapping._diagnostics import fit_summary
from .mapping._io import MappingFileLoader
from .mapping._image import MappingImage
from .mapping._pl_mapping import PLMapping, PL_Integration
from .mapping._raman_mapping import RamanMapping, Raman_Integration

__all__ = [
    "fit_summary",
    "MappingFileLoader",
    "MappingImage",
    "PLMapping",
    "PL_Integration",
    "RamanMapping",
    "Raman_Integration",
]