from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def resolve_backend_outcome(
    *,
    requested_backend: str,
    modality: str,
    axis_kind: str,
    translation_supported: bool = True,
    unsupported_steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Canonical backend decision point for all preprocessing workflows.

    Combines availability/input checks (from resolve_preprocessing_backend) with
    pipeline translation support into a single outcome object consumed by
    single-fit, mapping, and batch workflows.

    Parameters
    ----------
    translation_supported
        Whether all pipeline steps can be translated to RamanSPy.
        Pass True when no pipeline is provided (native always suffices).
    unsupported_steps
        Step descriptions that cannot be translated, used in fallback reason.

    Returns
    -------
    dict with canonical fields:
        requested_backend, resolved_backend, execution_ready,
        ramanspy_available, ramanspy_version, supported_for_input,
        translation_supported, unsupported_steps,
        fallback_used, fallback_reason, failure_mode,
        modality, axis_kind

    Raises
    ------
    NotImplementedError
        When requested_backend='ramanspy' but any condition is not met.
    """
    if unsupported_steps is None:
        unsupported_steps = []

    info = resolve_preprocessing_backend(
        requested_backend=requested_backend,
        modality=modality,
        axis_kind=axis_kind,
    )

    info["translation_supported"] = bool(translation_supported)
    info["unsupported_steps"] = list(unsupported_steps)
    info["fallback_used"] = False
    info["fallback_reason"] = None
    info["failure_mode"] = None

    backend = info["requested_backend"]

    if backend == "native":
        # resolve_preprocessing_backend() already set resolved_backend="native",
        # execution_ready=True, reason=None for native — nothing to override.
        return info

    if backend == "auto":
        if (
            info["ramanspy_available"]
            and info["supported_for_input"]
            and translation_supported
        ):
            info["resolved_backend"] = "ramanspy"
            info["execution_ready"] = True
        else:
            info["resolved_backend"] = "native"
            info["execution_ready"] = True
            info["fallback_used"] = True
            info["fallback_reason"] = _build_auto_fallback_reason(info, unsupported_steps)
            info["failure_mode"] = "fallback"
        return info

    if backend == "ramanspy":
        if not info["ramanspy_available"]:
            raise NotImplementedError(
                "RamanSPy backend requested, but RamanSPy is not installed."
            )
        if not info["supported_for_input"]:
            raise NotImplementedError(
                info.get("reason") or "RamanSPy backend requested, but input is not supported."
            )
        if not translation_supported:
            raise NotImplementedError(
                "RamanSPy translation is currently implemented for "
                "CropByRange, SmoothSavGol, and BaselineSubtract "
                "(poly / asls / airpls / arpls). "
                f"Unsupported steps: {unsupported_steps}"
            )
        info["resolved_backend"] = "ramanspy"
        info["execution_ready"] = True
        info["reason"] = None
        return info

    raise ValueError(f"Unsupported preprocessing backend '{backend}'.")  # pragma: no cover


def _build_auto_fallback_reason(
    info: Dict[str, Any],
    unsupported_steps: List[str],
) -> str:
    """Build a clear reason string for auto → native fallback."""
    if not info["ramanspy_available"]:
        return "Automatic fallback to native: RamanSPy is not installed."
    if not info["supported_for_input"]:
        support_reason = info.get("reason") or "input modality or axis is not supported by RamanSPy"
        return f"Automatic fallback to native: {support_reason}"
    if not info["translation_supported"]:
        return (
            "Automatic fallback to native: RamanSPy translation is not yet "
            f"implemented for: {unsupported_steps}"
        )
    return "Automatic fallback to native backend."


def format_backend_fallback_reason(outcome: Dict[str, Any]) -> str:
    """Return a user-facing string describing why auto fallback occurred."""
    reason = outcome.get("fallback_reason")
    if reason:
        return str(reason)
    return "Automatic fallback to native backend."


def format_backend_error_message(outcome: Dict[str, Any]) -> str:
    """Return a consistent error message string for forced ramanspy failures."""
    requested = outcome.get("requested_backend", "ramanspy")
    if not outcome.get("ramanspy_available", True):
        return f"Backend '{requested}' requested, but RamanSPy is not installed."
    if not outcome.get("supported_for_input", True):
        reason = outcome.get("reason", "input modality or axis not supported by RamanSPy")
        return f"Backend '{requested}' requested, but {reason}"
    if not outcome.get("translation_supported", True):
        steps = outcome.get("unsupported_steps", [])
        return (
            f"Backend '{requested}' requested, but pipeline contains steps not supported "
            f"by RamanSPy translation: {steps}"
        )
    return f"Backend '{requested}' is not available for this input."