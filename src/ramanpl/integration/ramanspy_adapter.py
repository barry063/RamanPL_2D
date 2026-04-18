from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.metadata import version as pkg_version, PackageNotFoundError
from typing import Any, Dict, Tuple

import numpy as np

try:
    from ..schema import normalise_axis_kind, normalise_modality
except Exception:  # pragma: no cover
    from ramanpl.schema import normalise_axis_kind, normalise_modality


@lru_cache(maxsize=1)
def has_ramanspy() -> bool:
    """
    Return True if RamanSPy is installed and importable.

    Result is cached after the first call to avoid repeated slow import probes.
    """
    try:
        import_module("ramanspy")
        return True
    except Exception:
        return False


def get_ramanspy_version() -> str | None:
    """
    Return the installed RamanSPy version, or None if unavailable.
    """
    try:
        return pkg_version("ramanspy")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def require_ramanspy() -> Any:
    """
    Import and return the RamanSPy top-level module.

    Raises
    ------
    ImportError
        If RamanSPy is not installed.
    """
    try:
        return import_module("ramanspy")
    except Exception as e:
        raise ImportError(
            "RamanSPy backend requested, but RamanSPy is not installed. "
            "Install it with: pip install RamanPL_2D[ramanspy]"
        ) from e


def can_use_ramanspy(*, modality: str, axis_kind: str) -> Tuple[bool, str | None]:
    """
    Check whether RamanSPy backend is valid for the given data semantics.

    Current policy for v0.4.1
    -------------------------
    - Raman workflows only
    - Raman shift axis only (cm^-1)
    - PL workflows remain on the native backend
    """
    mod = normalise_modality(modality)
    ax = normalise_axis_kind(axis_kind)

    if mod != "Raman":
        return False, "RamanSPy backend currently supports Raman workflows only."
    if ax != "raman_shift_cm-1":
        return False, (
            "RamanSPy backend currently requires axis_kind='raman_shift_cm-1'."
        )

    return True, None


def to_ramanspy_spectrum(
    *,
    x: np.ndarray,
    y: np.ndarray,
    modality: str,
    axis_kind: str,
):
    """
    Convert a RamanPL single spectrum into a RamanSPy Spectrum.

    Parameters
    ----------
    x : ndarray, shape [N]
        Raman shift axis.
    y : ndarray, shape [N]
        Spectrum intensities.
    modality : str
        Must normalise to 'Raman'.
    axis_kind : str
        Must normalise to 'raman_shift_cm-1'.
    """
    ok, reason = can_use_ramanspy(modality=modality, axis_kind=axis_kind)
    if not ok:
        raise ValueError(reason)
    rp = require_ramanspy()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError(f"x and y length mismatch: {x.size} vs {y.size}")

    return rp.Spectrum(y, spectral_axis=x)


def from_ramanspy_spectrum(obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a RamanSPy Spectrum back into (x, y).

    Returns
    -------
    x : ndarray, shape [N]
    y : ndarray, shape [N]
    """
    x = np.asarray(obj.spectral_axis, dtype=float).ravel()
    y = np.asarray(obj.spectral_data, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError(
            f"RamanSPy Spectrum axis/data mismatch: {x.size} vs {y.size}"
        )

    return x, y


def to_ramanspy_image(
    *,
    x: np.ndarray,
    cube_yxn: np.ndarray,
    modality: str,
    axis_kind: str,
):
    """
    Convert a RamanPL mapping cube [Y, X, N] into a RamanSPy SpectralImage.

    RamanSPy uses image-like spectral containers; here we expose a single,
    explicit orientation conversion so that the rest of the codebase does not
    silently mix conventions.
    """
    ok, reason = can_use_ramanspy(modality=modality, axis_kind=axis_kind)
    if not ok:
        raise ValueError(reason)
    rp = require_ramanspy()

    x = np.asarray(x, dtype=float).ravel()
    cube_yxn = np.asarray(cube_yxn, dtype=float)

    if cube_yxn.ndim != 3:
        raise ValueError("cube_yxn must be 3D with shape [Y, X, N].")
    if cube_yxn.shape[2] != x.size:
        raise ValueError(
            f"cube_yxn.shape[2] ({cube_yxn.shape[2]}) must match len(x) ({x.size})."
        )

    # RamanPL convention: [Y, X, N]; RamanSPy image convention: [X, Y, N].
    # The transpose is unavoidable — conversion overhead not further reducible here (v0.4.6).
    cube_xyn = np.transpose(cube_yxn, (1, 0, 2))

    return rp.SpectralImage(cube_xyn, spectral_axis=x)


def from_ramanspy_image(obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a RamanSPy SpectralImage back into RamanPL convention.

    Returns
    -------
    x : ndarray, shape [N]
    cube_yxn : ndarray, shape [Y, X, N]
    """
    x = np.asarray(obj.spectral_axis, dtype=float).ravel()
    cube_xyn = np.asarray(obj.spectral_data, dtype=float)

    if cube_xyn.ndim != 3:
        raise ValueError("RamanSPy SpectralImage data must be 3D.")
    if cube_xyn.shape[2] != x.size:
        raise ValueError(
            f"RamanSPy SpectralImage axis/data mismatch: "
            f"{cube_xyn.shape[2]} vs {x.size}"
        )

    cube_yxn = np.transpose(cube_xyn, (1, 0, 2))
    return x, cube_yxn


def ramanspy_backend_info(*, modality: str, axis_kind: str) -> Dict[str, Any]:
    """
    Small helper for metadata/debugging.
    """
    ok, reason = can_use_ramanspy(modality=modality, axis_kind=axis_kind)
    return {
        "available": has_ramanspy(),
        "version": get_ramanspy_version(),
        "supported_for_input": ok,
        "reason": reason,
        "modality": normalise_modality(modality),
        "axis_kind": normalise_axis_kind(axis_kind),
    }