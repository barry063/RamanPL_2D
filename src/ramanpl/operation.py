# src/ramanpl/operation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal
import numpy as np

from ramanpl.dataImporter import DataImporter


AxisType = Literal["energy", "wavenumber", "auto"]
AlignType = Literal["interp", "strict"]
OpType = Literal["add", "sub"]


@dataclass(frozen=True)
class Spectrum:
    """
    Minimal container for a single spectrum.

    y: intensity
    x: energy (eV) or wavenumber (cm^-1), depending on axis
    axis: "energy" or "wavenumber"
    source: optional provenance string
    """
    y: np.ndarray
    x: np.ndarray
    axis: Literal["energy", "wavenumber"]
    source: str = ""


class ArithmeticSpectrum:
    """
    Arithmetic pre-processing for single spectra (and optionally mapping cubes).

    Core contract:
      - output is (y_out, x_out) that plugs directly into PLfit / RamanFit constructors.
    """

    # ----------------------------
    # Public API
    # ----------------------------
    @staticmethod
    def combine(
        A: Union[str, Tuple[np.ndarray, np.ndarray], Spectrum],
        B: Union[str, Tuple[np.ndarray, np.ndarray], Spectrum],
        *,
        op: OpType = "sub",
        k: float = 1.0,
        axis: AxisType = "auto",
        align: AlignType = "interp",
        save: bool = False,
        out_path: Optional[str] = None,
        clip_nonnegative: bool = False,
        txt_delimiter: str = "\t",
        txt_skiprows: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine two spectra with arithmetic:
            op="sub":  C = A - k*B
            op="add":  C = A + k*B

        Parameters
        ----------
        A, B
            filename (.wdf/.txt) OR (y, x) tuple OR Spectrum
        op
            "sub" or "add"
        k
            coefficient for B
        axis
            "energy", "wavenumber", or "auto" (infer from x-range)
        align
            "interp" (default): interpolate B onto A.x if grids differ
            "strict": require identical x arrays
        save
            if True, write output to out_path (two-column txt)
        out_path
            required if save=True
        clip_nonnegative
            if True, clip output y to >= 0 (optional; default False)

        Returns
        -------
        (y_out, x_out)
        """
        specA = ArithmeticSpectrum._coerce_to_spectrum(A, axis=axis,
                                                      txt_delimiter=txt_delimiter,
                                                      txt_skiprows=txt_skiprows)
        specB = ArithmeticSpectrum._coerce_to_spectrum(B, axis=axis,
                                                      txt_delimiter=txt_delimiter,
                                                      txt_skiprows=txt_skiprows)

        # Enforce type match (PL vs Raman)
        if specA.axis != specB.axis:
            raise ValueError(
                "Spectrum type mismatch: A is '{0}' while B is '{1}'. "
                "Arithmetic requires both to be the same type.".format(specA.axis, specB.axis)
            )

        xA, yA = specA.x, specA.y
        xB, yB = specB.x, specB.y

        # Validate shapes
        if xA.ndim != 1 or yA.ndim != 1 or xB.ndim != 1 or yB.ndim != 1:
            raise ValueError("Only 1D single spectra are supported in combine().")

        if yA.size != xA.size:
            raise ValueError(f"A length mismatch: len(y)={yA.size} vs len(x)={xA.size}")
        if yB.size != xB.size:
            raise ValueError(f"B length mismatch: len(y)={yB.size} vs len(x)={xB.size}")

        # Align x-grids
        if align == "strict":
            if xA.size != xB.size or not np.allclose(xA, xB, rtol=0, atol=0):
                raise ValueError("align='strict' requires identical x arrays for A and B.")
            yB_aligned = yB
            x_out = xA
        elif align == "interp":
            yB_aligned = ArithmeticSpectrum._interp_to(x_src=xB, y_src=yB, x_tgt=xA)
            x_out = xA
        else:
            raise ValueError("align must be 'interp' or 'strict'.")

        # Arithmetic
        k = float(k)
        if op == "sub":
            y_out = yA - k * yB_aligned
        elif op == "add":
            y_out = yA + k * yB_aligned
        else:
            raise ValueError("op must be 'sub' or 'add'.")

        if clip_nonnegative:
            y_out = np.clip(y_out, a_min=0.0, a_max=None)

        # Optional save
        if save:
            if not out_path:
                raise ValueError("out_path must be provided when save=True.")
            ArithmeticSpectrum._save_txt(out_path, x_out, y_out)

        return y_out, x_out

    @staticmethod
    def combine_map_with_reference(
        map_data: Union[str, Tuple[np.ndarray, np.ndarray, int, int]],
        reference: Union[str, Tuple[np.ndarray, np.ndarray], Spectrum],
        *,
        op: OpType = "sub",
        k: float = 1.0,
        axis: AxisType = "auto",
        align: AlignType = "interp",
        x_range: Optional[Tuple[float, float]] = None,
        save: bool = False,
        out_path: Optional[str] = None,
        clip_nonnegative: bool = False,
        txt_delimiter: str = "\t",
        txt_skiprows: int = 1,
        map_txt_skiprows: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Apply arithmetic between a mapping cube and a single reference spectrum:

            op="sub":  cube_out[j,i,:] = cube[j,i,:] - k*ref(:)
            op="add":  cube_out[j,i,:] = cube[j,i,:] + k*ref(:)

        Parameters
        ----------
        map_data
            filename (.wdf/.txt mapping) OR (cube, x, X, Y) tuple
            where cube shape is [Y, X, N]
        reference
            filename (.wdf/.txt single spectrum) OR (y, x) tuple OR Spectrum
        x_range
            optional spectral trimming applied to the mapping import (physical units)
        save
            if True, writes mapping .txt to out_path with columns [X, Y, xaxis, intensity]
            using integer indices for X,Y coordinates (0..X-1, 0..Y-1)

        Returns
        -------
        (cube_out, x_out, X, Y)
        """
        cube, x_map, X, Y = ArithmeticSpectrum._coerce_to_map(
            map_data,
            axis=axis,
            x_range=x_range,
            map_txt_skiprows=map_txt_skiprows,
        )

        # Reference spectrum (single)
        ref_spec = ArithmeticSpectrum._coerce_to_spectrum(
            reference,
            axis=axis,
            txt_delimiter=txt_delimiter,
            txt_skiprows=txt_skiprows,
        )

        # Axis/type consistency (PL vs Raman)
        map_axis = ArithmeticSpectrum._resolve_axis(x_map, axis=axis)
        if map_axis != ref_spec.axis:
            raise ValueError(
                f"Spectrum type mismatch: mapping is '{map_axis}' while reference is '{ref_spec.axis}'. "
                "Arithmetic requires both to be the same type."
            )

        # Align reference onto mapping x-grid
        x_ref = ref_spec.x
        y_ref = ref_spec.y

        if align == "strict":
            if x_ref.size != x_map.size or not np.allclose(x_ref, x_map, rtol=0, atol=0):
                raise ValueError("align='strict' requires identical x arrays for mapping and reference.")
            y_ref_aligned = y_ref
            x_out = x_map
        elif align == "interp":
            y_ref_aligned = ArithmeticSpectrum._interp_to(x_src=x_ref, y_src=y_ref, x_tgt=x_map)
            x_out = x_map
        else:
            raise ValueError("align must be 'interp' or 'strict'.")

        # Vectorised broadcast: [Y, X, N] +/- [N] -> [Y, X, N]
        k = float(k)
        if op == "sub":
            cube_out = cube - k * y_ref_aligned[None, None, :]
        elif op == "add":
            cube_out = cube + k * y_ref_aligned[None, None, :]
        else:
            raise ValueError("op must be 'sub' or 'add'.")

        if clip_nonnegative:
            cube_out = np.clip(cube_out, a_min=0.0, a_max=None)

        if save:
            if not out_path:
                raise ValueError("out_path must be provided when save=True.")
            ArithmeticSpectrum._save_map_txt(out_path, cube_out, x_out, X, Y)

        return cube_out, x_out, X, Y

    @staticmethod
    def reference_from_map(
        map_data: Union[str, Tuple[np.ndarray, np.ndarray, int, int]],
        *,
        ref_x: int,
        ref_y: int,
        axis: AxisType = "auto",
        x_range: Optional[Tuple[float, float]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        map_txt_skiprows: int = 1,
    ) -> Spectrum:
        """
        Build a single reference Spectrum from a mapping dataset.

        Parameters
        ----------
        map_data
            filename (.wdf/.txt mapping) OR (cube, x, X, Y)
        ref_x, ref_y
            pixel coordinate (0-indexed)
        x_range
            optional spectral trimming applied to the mapping import (physical units)
        roi
            Optional ROI (x0, x1, y0, y1) inclusive. If provided, reference spectrum is the
            mean spectrum over the ROI and ref_x/ref_y are ignored.

        Returns
        -------
        Spectrum(y_ref, x_map, axis=...)
        """
        cube, x_map, X, Y = ArithmeticSpectrum._coerce_to_map(
            map_data,
            axis=axis,
            x_range=x_range,
            map_txt_skiprows=map_txt_skiprows,
        )

        if roi is not None:
            x0, x1, y0, y1 = roi
            if not (0 <= x0 <= x1 < X and 0 <= y0 <= y1 < Y):
                raise ValueError(f"ROI out of bounds for map size (X={X}, Y={Y}).")
            y_ref = np.nanmean(cube[y0:y1+1, x0:x1+1, :], axis=(0, 1))
            source = f"map_roi[x={x0}:{x1}, y={y0}:{y1}]"
        else:
            if not (0 <= ref_x < X and 0 <= ref_y < Y):
                raise ValueError(f"Reference pixel (ref_x={ref_x}, ref_y={ref_y}) out of bounds for (X={X}, Y={Y}).")
            y_ref = cube[ref_y, ref_x, :].astype(float, copy=False)
            source = f"map_pixel[x={ref_x}, y={ref_y}]"

        ax = ArithmeticSpectrum._resolve_axis(x_map, axis=axis)
        return Spectrum(y=np.asarray(y_ref, dtype=float).ravel(),
                        x=np.asarray(x_map, dtype=float).ravel(),
                        axis=ax,
                        source=source)

    @staticmethod
    def combine_map_with_pixel_reference(
        map_data: Union[str, Tuple[np.ndarray, np.ndarray, int, int]],
        *,
        ref_x: int,
        ref_y: int,
        op: OpType = "sub",
        k: float = 1.0,
        axis: AxisType = "auto",
        align: AlignType = "strict",
        x_range: Optional[Tuple[float, float]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        save: bool = False,
        out_path: Optional[str] = None,
        clip_nonnegative: bool = False,
        map_txt_skiprows: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, Spectrum]:
        """
        Compute a new mapping cube by applying arithmetic to each pixel using a reference
        spectrum taken from a selected map pixel (or ROI mean).

        Returns
        -------
        cube_out, x_out, X, Y, ref_spectrum
        """
        # Load map
        cube, x_map, X, Y = ArithmeticSpectrum._coerce_to_map(
            map_data,
            axis=axis,
            x_range=x_range,
            map_txt_skiprows=map_txt_skiprows,
        )

        # Extract reference spectrum from map (pixel or ROI)
        ref_spec = ArithmeticSpectrum.reference_from_map(
            (cube, x_map, X, Y),
            ref_x=ref_x,
            ref_y=ref_y,
            axis=axis,
            x_range=None,              # already applied above
            roi=roi,
            map_txt_skiprows=map_txt_skiprows,
        )

        # Since ref_spec.x is x_map, strict alignment is valid by default
        # But keep the same interface as combine_map_with_reference
        if align == "strict":
            y_ref_aligned = ref_spec.y
            x_out = x_map
        elif align == "interp":
            # This is mostly useful only if you later allow ref spectra from other sources
            y_ref_aligned = ArithmeticSpectrum._interp_to(ref_spec.x, ref_spec.y, x_map)
            x_out = x_map
        else:
            raise ValueError("align must be 'interp' or 'strict'.")

        k = float(k)
        if op == "sub":
            cube_out = cube - k * y_ref_aligned[None, None, :]
        elif op == "add":
            cube_out = cube + k * y_ref_aligned[None, None, :]
        else:
            raise ValueError("op must be 'sub' or 'add'.")

        if clip_nonnegative:
            cube_out = np.clip(cube_out, a_min=0.0, a_max=None)

        if save:
            if not out_path:
                raise ValueError("out_path must be provided when save=True.")
            ArithmeticSpectrum._save_map_txt(out_path, cube_out, x_out, X, Y)

        return cube_out, x_out, X, Y, ref_spec

    # ----------------------------
    # Internals
    # ----------------------------
    @staticmethod
    def _coerce_to_spectrum(
        obj: Union[str, Tuple[np.ndarray, np.ndarray], Spectrum],
        *,
        axis: AxisType,
        txt_delimiter: str,
        txt_skiprows: int,
    ) -> Spectrum:
        if isinstance(obj, Spectrum):
            # If user forced axis, validate
            if axis != "auto" and obj.axis != axis:
                raise ValueError(f"Axis override '{axis}' conflicts with provided Spectrum.axis='{obj.axis}'.")
            return obj

        if isinstance(obj, str):
            # Load using shared importer (single spectrum)
            # axis is used only for messaging there, so we do our own validation/inference here.
            y, x = DataImporter.data_import(
                filename=obj,
                readlines=None,      # IMPORTANT: disable default trimming
                x_range=None,
                axis="auto",
                txt_delimiter=txt_delimiter,
                txt_skiprows=txt_skiprows,
            )
            y = np.asarray(y, dtype=float).ravel()
            x = np.asarray(x, dtype=float).ravel()
            ax = ArithmeticSpectrum._resolve_axis(x, axis=axis)
            return Spectrum(y=y, x=x, axis=ax, source=obj)

        # tuple (y, x)
        if isinstance(obj, tuple) and len(obj) == 2:
            y, x = obj
            y = np.asarray(y, dtype=float).ravel()
            x = np.asarray(x, dtype=float).ravel()
            ax = ArithmeticSpectrum._resolve_axis(x, axis=axis)
            return Spectrum(y=y, x=x, axis=ax, source="tuple")

        raise ValueError("Unsupported input type for spectrum. Use filename, (y,x) tuple, or Spectrum.")

    @staticmethod
    def _resolve_axis(x: np.ndarray, *, axis: AxisType) -> Literal["energy", "wavenumber"]:
        """
        Robust-enough heuristic:
          - PL energy is typically < ~20 (eV scale; in your code it is ~1–3 eV)
          - Raman shift is typically tens to thousands (cm^-1)
        """
        if axis in ("energy", "wavenumber"):
            return axis

        # auto
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("Cannot infer axis from empty x array.")

        xmax = float(np.nanmax(x))
        xmin = float(np.nanmin(x))

        # If x is in eV, range and magnitudes should be small (order 1–5 typically).
        # If x is Raman shift, it is usually > 50 cm^-1.
        if xmax <= 20.0 and xmin >= -5.0:
            return "energy"
        if xmax > 50.0:
            return "wavenumber"

        # Ambiguous band (e.g., 20–50); force user to declare.
        raise ValueError(
            f"Axis inference ambiguous from x-range [{xmin:.3g}, {xmax:.3g}]. "
            "Please pass axis='energy' or axis='wavenumber'."
        )

    @staticmethod
    def _interp_to(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        x_src = np.asarray(x_src, dtype=float).ravel()
        y_src = np.asarray(y_src, dtype=float).ravel()
        x_tgt = np.asarray(x_tgt, dtype=float).ravel()

        if x_src.size < 2:
            raise ValueError("Cannot interpolate from fewer than 2 points.")

        # Ensure monotonic x for interpolation
        if x_src[0] > x_src[-1]:
            x_src = x_src[::-1]
            y_src = y_src[::-1]

        # Overlap check
        lo = max(np.min(x_src), np.min(x_tgt))
        hi = min(np.max(x_src), np.max(x_tgt))
        if not (lo < hi):
            raise ValueError(
                "A and B have no overlapping x-range; cannot align by interpolation."
            )

        # Interpolate (np.interp requires ascending x_src)
        return np.interp(x_tgt, x_src, y_src)

    @staticmethod
    def _save_txt(path: str, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size != y.size:
            raise ValueError("Cannot save: x and y must be same length.")

        header = "x\tintensity"
        data = np.column_stack([x, y])
        np.savetxt(path, data, delimiter="\t", header=header, comments="")

    @staticmethod
    def _coerce_to_map(
        obj: Union[str, Tuple[np.ndarray, np.ndarray, int, int]],
        *,
        axis: AxisType,
        x_range: Optional[Tuple[float, float]],
        map_txt_skiprows: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Returns (cube, x_map, X, Y) with cube shape [Y, X, N].
        """
        if isinstance(obj, str):
            cube, x_map, X, Y = DataImporter.map_import(
                filename=obj,
                x_range=x_range,
                axis="auto",
                txt_skiprows=map_txt_skiprows,
            )
            cube = np.asarray(cube, dtype=float)
            x_map = np.asarray(x_map, dtype=float).ravel()
            return cube, x_map, int(X), int(Y)

        if isinstance(obj, tuple) and len(obj) == 4:
            cube, x_map, X, Y = obj
            cube = np.asarray(cube, dtype=float)
            x_map = np.asarray(x_map, dtype=float).ravel()
            X = int(X)
            Y = int(Y)
            if cube.ndim != 3:
                raise ValueError("Mapping cube must be 3D with shape [Y, X, N].")
            if cube.shape[0] != Y or cube.shape[1] != X:
                raise ValueError(f"Cube shape {cube.shape[:2]} inconsistent with provided (X,Y)=({X},{Y}).")
            if cube.shape[2] != x_map.size:
                raise ValueError(f"Spectral length mismatch: cube N={cube.shape[2]} vs len(x)={x_map.size}.")
            return cube, x_map, X, Y

        raise ValueError("Unsupported input type for mapping. Use filename or (cube, x, X, Y) tuple.")

    @staticmethod
    def _save_map_txt(path: str, cube: np.ndarray, x: np.ndarray, X: int, Y: int) -> None:
        """
        Save mapping cube to .txt with columns [X, Y, xaxis, intensity].
        Uses integer indices for X and Y coordinates (0..X-1, 0..Y-1).
        """
        cube = np.asarray(cube, dtype=float)
        x = np.asarray(x, dtype=float).ravel()

        if cube.ndim != 3:
            raise ValueError("cube must be 3D [Y, X, N].")
        if cube.shape[0] != Y or cube.shape[1] != X:
            raise ValueError("cube shape and (X,Y) do not match.")
        if cube.shape[2] != x.size:
            raise ValueError("cube spectral length must match x length.")

        # Vectorised build: meshgrid of pixel indices broadcast to [Y, X, N]
        yi_grid, xi_grid = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")  # [Y, X]
        Xi_all = np.broadcast_to(xi_grid[:, :, None], cube.shape).reshape(-1).astype(float)
        Yj_all = np.broadcast_to(yi_grid[:, :, None], cube.shape).reshape(-1).astype(float)
        x_all  = np.broadcast_to(x[None, None, :],    cube.shape).reshape(-1)
        I_all  = cube.reshape(-1)
        data = np.column_stack([Xi_all, Yj_all, x_all, I_all])
        header = "X\tY\txaxis\tintensity"
        np.savetxt(path, data, delimiter="\t", header=header, comments="")
