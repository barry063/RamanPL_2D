"""
Compatibility wrapper for the PL single-spectrum fitter.

Public behaviour is preserved:
    from ramanpl import PLfit
    from ramanpl.PLfit import PLfit
"""

from .single_fit.PLfit import PLfit, DataImporter

__all__ = ["PLfit", "DataImporter"]