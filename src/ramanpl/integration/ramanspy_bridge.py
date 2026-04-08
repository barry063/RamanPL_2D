from __future__ import annotations

from typing import Any, Dict

try:
    from ..schema import (
        normalise_axis_kind,
        normalise_modality,
        normalise_preprocess_backend,
    )
    from .ramanspy_adapter import (
        can_use_ramanspy,
        get_ramanspy_version,
        has_ramanspy,
    )
except Exception:  # pragma: no cover
    from ramanpl.schema import (
        normalise_axis_kind,
        normalise_modality,
        normalise_preprocess_backend,
    )
    from ramanpl.integration.ramanspy_adapter import (
        can_use_ramanspy,
        get_ramanspy_version,
        has_ramanspy,
    )


def resolve_preprocessing_backend(
    *,
    requested_backend: str,
    modality: str,
    axis_kind: str,
) -> Dict[str, Any]:
    """
    Resolve preprocessing backend availability for the current input.

    Notes
    -----
    This function resolves only backend availability and input compatibility.

    It does NOT decide whether a specific RamanPL Pipeline is translatable to
    RamanSPy. Pipeline-step translation support is checked later by the
    preprocessing layer.

    v0.4.1 policy
    -------------
    - native:
        always resolves to native
    - auto:
        may later promote to RamanSPy if:
            * RamanSPy is installed
            * the input is Raman / cm^-1 compatible
            * the pipeline is fully translatable
    - ramanspy:
        valid only when RamanSPy is installed and the input is compatible;
        pipeline translation support is checked separately upstream
    """
    backend = normalise_preprocess_backend(requested_backend)
    mod = normalise_modality(modality)
    ax = normalise_axis_kind(axis_kind)

    available = has_ramanspy()
    supported_for_input, support_reason = can_use_ramanspy(
        modality=mod,
        axis_kind=ax,
    )
    version = get_ramanspy_version()

    if backend == "native":
        resolved = "native"
        execution_ready = True
        reason = None

    elif backend == "auto":
        # Final promotion to RamanSPy is decided later after pipeline translation check
        resolved = "native"
        execution_ready = True
        reason = None

    elif backend == "ramanspy":
        resolved = "ramanspy"
        execution_ready = bool(available and supported_for_input)

        if not available:
            reason = "RamanSPy backend requested, but RamanSPy is not installed."
        elif not supported_for_input:
            reason = support_reason
        else:
            reason = None

    else:  # pragma: no cover
        raise ValueError(f"Unsupported preprocessing backend '{backend}'.")

    return {
        "requested_backend": backend,
        "resolved_backend": resolved,
        "execution_ready": execution_ready,
        "ramanspy_available": available,
        "ramanspy_version": version,
        "supported_for_input": supported_for_input,
        "reason": reason,
        "modality": mod,
        "axis_kind": ax,
    }