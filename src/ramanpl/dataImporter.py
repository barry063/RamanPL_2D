"""
Shared data importer for Raman/PL single-spectrum and mapping files.

Supports:
- .wdf (Renishaw WiRE) via renishawWiRE.WDFReader
- .txt two-column single spectrum: xdata, intensity
- .txt mapping: columns [X, Y, xaxis, intensity] (header row skipped)

Preferred range selection:
- x_range=(xmin, xmax) in physical units

Legacy selection (deprecated):
- readlines=[start_index, end_index]
"""

from __future__ import annotations

import warnings
import numpy as np


class DataImporter:
    """
    Shared importer used by PL/Raman modules.

    axis:
        - "energy"      -> PL convention (eV)
        - "wavenumber"  -> Raman convention (cm^-1)
        - "auto"        -> no validation; used only for messaging
    """

    # --------------------------------------------------------------------------------------
    # Public API: single spectrum
    # --------------------------------------------------------------------------------------
    @staticmethod
    def data_import(
        filename: str,
        readlines=None,
        x_range=None,
        axis: str = "auto",
        txt_delimiter: str = "\t",
        txt_skiprows: int = 1,
    ):
        spectra, xdata = DataImporter._read_single_file(
            filename=filename,
            txt_delimiter=txt_delimiter,
            txt_skiprows=txt_skiprows,
        )

        spectra = np.asarray(spectra, dtype=float).ravel()
        xdata = np.asarray(xdata, dtype=float).ravel()

        if spectra.size != xdata.size:
            raise RuntimeError("Imported spectrum and x-axis must have the same length.")

        # Prefer physical trimming if x_range is provided
        if x_range is not None:
            x_sel, y_sel = DataImporter.select_by_xrange(xdata, spectra, x_range, axis=axis)
            return y_sel, x_sel

        # Legacy trimming only if readlines is not None
        if readlines is not None:
            import warnings
            warnings.warn(
                "DataImporter.data_import(..., readlines=...) is deprecated and will be removed in a future version. "
                "Please use x_range=(xmin, xmax) in physical units instead.",
                category=FutureWarning,
                stacklevel=2,
            )
            x_sel, y_sel = DataImporter._select_by_readlines(xdata=xdata, ydata=spectra, readlines=readlines)
            return y_sel, x_sel

        # No trimming
        return spectra, xdata

    # --------------------------------------------------------------------------------------
    # Public API: mapping import
    # --------------------------------------------------------------------------------------
    @staticmethod
    def map_import(
        filename: str,
        x_range=None,
        axis: str = "auto",
        txt_skiprows: int = 1,
    ):
        """
        Import mapping data from .wdf or .txt.

        .wdf:
            Uses reader.map_shape, reader.xdata and reader.spectra[j][i][:]

        .txt expected columns:
            [X, Y, xaxis, intensity]
            where X and Y are coordinates (numeric), xaxis repeats per pixel.

        Returns
        -------
        spectra_cube : np.ndarray
            Shape [Y, X, N]
        xdata : np.ndarray
            Shape [N]
        X : int
        Y : int
        """
        fn = filename.lower()

        if fn.endswith(".wdf"):
            spectra_cube, xdata, X, Y = DataImporter._read_wdf_map(filename)
        elif fn.endswith(".txt"):
            spectra_cube, xdata, X, Y = DataImporter._read_txt_map(
                filename=filename,
                txt_skiprows=txt_skiprows,
            )
        else:
            raise RuntimeError("Unsupported format for mapping import. Supported formats: .wdf, .txt")

        xdata = np.asarray(xdata, dtype=float).ravel()
        spectra_cube = np.asarray(spectra_cube, dtype=float)

        # Optional spectral trimming by x_range (saves memory and speeds up downstream)
        if x_range is not None:
            mask = DataImporter.mask_by_xrange(xdata, x_range)
            if not np.any(mask):
                unit = DataImporter._axis_unit(axis)
                raise RuntimeError(
                    f"x_range {x_range} selects no points. "
                    f"Available range: [{xdata.min():.6g}, {xdata.max():.6g}] {unit}"
                )
            xdata = xdata[mask]
            spectra_cube = spectra_cube[:, :, mask]

        return spectra_cube, xdata, X, Y

    # --------------------------------------------------------------------------------------
    # Public utilities for consistent masking/slicing
    # --------------------------------------------------------------------------------------
    @staticmethod
    def mask_by_xrange(xdata: np.ndarray, x_range):
        """
        Return a boolean mask selecting xdata within x_range, robust to ascending/descending xdata.
        """
        xdata = np.asarray(xdata, dtype=float).ravel()
        if not isinstance(x_range, (list, tuple)) or len(x_range) != 2:
            raise RuntimeError("x_range must be a list/tuple of length 2: (xmin, xmax).")

        xmin = float(x_range[0])
        xmax = float(x_range[1])
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)

        if xdata.size == 0:
            return np.zeros((0,), dtype=bool)

        if xdata[0] <= xdata[-1]:
            return (xdata >= lo) & (xdata <= hi)
        return (xdata <= hi) & (xdata >= lo)

    @staticmethod
    def select_by_xrange(xdata: np.ndarray, ydata: np.ndarray, x_range, axis: str = "auto"):
        """
        Return (x_selected, y_selected) sliced by x_range.
        """
        xdata = np.asarray(xdata, dtype=float).ravel()
        ydata = np.asarray(ydata, dtype=float).ravel()

        if xdata.size != ydata.size:
            raise RuntimeError(f"xdata and ydata length mismatch: {xdata.size} vs {ydata.size}")

        mask = DataImporter.mask_by_xrange(xdata, x_range)
        if not np.any(mask):
            unit = DataImporter._axis_unit(axis)
            raise RuntimeError(
                f"x_range {x_range} selects no points. "
                f"Available range: [{xdata.min():.6g}, {xdata.max():.6g}] {unit}"
            )

        return xdata[mask], ydata[mask]

    # --------------------------------------------------------------------------------------
    # Internal readers
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _read_single_file(filename: str, txt_delimiter: str, txt_skiprows: int):
        fn = filename.lower()

        if fn.endswith(".wdf"):
            try:
                from renishawWiRE import WDFReader
            except Exception as e:
                raise RuntimeError(
                    "Failed to import renishawWiRE.WDFReader. Install/verify renishawWiRE to read .wdf files."
                ) from e

            reader = WDFReader(filename)
            return reader.spectra, reader.xdata

        if fn.endswith(".txt"):
            try:
                data = np.loadtxt(filename, delimiter=txt_delimiter, skiprows=txt_skiprows)
            except Exception:
                data = np.loadtxt(filename, delimiter=None, skiprows=txt_skiprows)

            if data.ndim != 2 or data.shape[1] != 2:
                raise RuntimeError("Text file must contain exactly 2 columns: xdata and intensity.")

            xdata = data[:, 0]
            spectra = data[:, 1]
            return spectra, xdata

        raise RuntimeError("Unsupported format. Supported formats: .wdf, .txt")

    @staticmethod
    def _read_wdf_map(filename: str):
        try:
            from renishawWiRE import WDFReader
        except Exception as e:
            raise RuntimeError(
                "Failed to import renishawWiRE.WDFReader. Install/verify renishawWiRE to read .wdf files."
            ) from e

        reader = WDFReader(filename)
        X = int(reader.map_shape[0])
        Y = int(reader.map_shape[1])

        xdata = np.asarray(reader.xdata[:], dtype=float).ravel()
        spectra_cube = np.zeros((Y, X, xdata.size), dtype=float)

        for j in range(Y):
            for i in range(X):
                spectra_cube[j, i, :] = np.asarray(reader.spectra[j][i][:], dtype=float).ravel()

        return spectra_cube, xdata, X, Y

    @staticmethod
    def _read_txt_map(filename: str, txt_skiprows: int = 1):
        data = np.loadtxt(filename, skiprows=txt_skiprows)

        if data.ndim != 2 or data.shape[1] < 4:
            raise RuntimeError(
                "Mapping .txt must contain at least 4 columns: [X, Y, xaxis, intensity]."
            )

        x_coords = np.unique(data[:, 0])
        y_coords = np.unique(data[:, 1])
        X = int(len(x_coords))
        Y = int(len(y_coords))

        # Points per location inferred from first pixel
        pts = int(np.sum((data[:, 0] == x_coords[0]) & (data[:, 1] == y_coords[0])))
        if pts <= 0:
            raise RuntimeError("Failed to infer points_per_location from mapping .txt format.")

        xdata = np.asarray(data[:pts, 2], dtype=float).ravel()
        spectra_cube = np.zeros((Y, X, xdata.size), dtype=float)

        index = 0
        for j in range(Y):
            for i in range(X):
                spectra_cube[j, i, :] = np.asarray(data[index:index + pts, 3], dtype=float).ravel()
                index += pts

        return spectra_cube, xdata, X, Y

    @staticmethod
    def _select_by_readlines(xdata: np.ndarray, ydata: np.ndarray, readlines):
        if not isinstance(readlines, (list, tuple)) or len(readlines) != 2:
            raise RuntimeError("readlines must be a list/tuple of length 2: [start, end].")

        n = ydata.size
        start = max(0, min(int(readlines[0]), n))
        end = max(start, min(int(readlines[1]), n))
        return xdata[start:end], ydata[start:end]

    @staticmethod
    def _axis_unit(axis: str) -> str:
        axis = (axis or "auto").lower()
        if axis == "energy":
            return "eV"
        if axis == "wavenumber":
            return "cm^-1"
        return ""
