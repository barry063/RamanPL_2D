"""
Compatibility wrapper for the Raman single-spectrum fitter.

Public behaviour is preserved:
    from ramanpl import RamanFit
    from ramanpl.RamanFit import RamanFit
"""

from .single_fit.RamanFit import RamanFit, DataImporter

__all__ = ["RamanFit", "DataImporter"]