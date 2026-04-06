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
    Resolve preprocessing backend selection for the current input.

    v0.4.0 policy
    -------------
    - native: execute natively
    - auto:   currently resolves to native, even if RamanSPy is installed
    - ramanspy:
        accepted as a selector value, but execution is not enabled yet;
        callers should raise a clear NotImplementedError
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
        resolved = "native"
        execution_ready = True
        if available and supported_for_input:
            reason = (
                "RamanSPy is available and supported for this input, "
                "but preprocessing execution is not enabled yet in v0.4.0. "
                "Using native backend."
            )
        else:
            reason = None

    elif backend == "ramanspy":
        resolved = "ramanspy"
        execution_ready = False
        if not available:
            reason = (
                "RamanSPy backend requested, but RamanSPy is not installed."
            )
        elif not supported_for_input:
            reason = support_reason
        else:
            reason = (
                "RamanSPy backend requested, but preprocessing execution "
                "is not enabled yet in v0.4.0."
            )

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