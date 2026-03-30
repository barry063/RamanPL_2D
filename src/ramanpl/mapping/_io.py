try:
    from ..dataImporter import DataImporter
except Exception:  # pragma: no cover
    from dataImporter import DataImporter


class MappingFileLoader:
    """Loader for spectroscopic mapping data from .wdf and .txt files."""

    def __init__(self, filename, x_range=None, axis="auto", txt_skiprows=1):
        self.filename = filename
        self.axis = axis
        self.data_format = "wdf" if filename.lower().endswith(".wdf") else "txt" if filename.lower().endswith(".txt") else "unknown"

        if self.data_format == "unknown":
            raise RuntimeError("Unsupported mapping file format. Supported formats: .wdf, .txt")

        spectra_cube, xdata, X, Y = DataImporter.map_import(
            filename=filename,
            x_range=x_range,
            axis=axis,
            txt_skiprows=txt_skiprows,
        )

        self.X = X
        self.Y = Y
        self.xdata = xdata
        self.spectra = spectra_cube