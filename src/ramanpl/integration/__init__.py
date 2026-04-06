"""
Integration backends for optional third-party interoperability.

Current scope
-------------
- RamanSPy backend infrastructure
- Raman-only container conversion helpers
- RamanSPy translation support for selected preprocessing steps
"""

from .ramanspy_adapter import (
    has_ramanspy,
    get_ramanspy_version,
    can_use_ramanspy,
    require_ramanspy,
    to_ramanspy_spectrum,
    from_ramanspy_spectrum,
    to_ramanspy_image,
    from_ramanspy_image,
)
from .ramanspy_translate import (
    pipeline_supports_ramanspy_translation,
    apply_pipeline_ramanspy_single,
    apply_pipeline_ramanspy_mapping,
)

__all__ = [
    "has_ramanspy",
    "get_ramanspy_version",
    "can_use_ramanspy",
    "require_ramanspy",
    "to_ramanspy_spectrum",
    "from_ramanspy_spectrum",
    "to_ramanspy_image",
    "from_ramanspy_image",
    "pipeline_supports_ramanspy_translation",
    "apply_pipeline_ramanspy_single",
    "apply_pipeline_ramanspy_mapping",
]