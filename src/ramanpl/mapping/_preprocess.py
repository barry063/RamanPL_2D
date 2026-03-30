import numpy as np

try:
    from ..preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )
except Exception:  # pragma: no cover
    from preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )


class _MappingPreprocessMixin:
    """
    Shared preprocessing helpers for PLMapping and RamanMapping.
    """

    def _build_default_preprocessing_pipeline(self) -> Pipeline:
        return build_legacy_mapping_pipeline(
            data_range=self.data_range,
            smoothing=self.smoothing,
            smooth_window=self.smooth_window,
            smooth_order=self.smooth_poly,
            background_remove=self.background_remove,
            baseline_method=self.baseline_method,
            poly_degree=self.poly_degree,
            gaussian_sigma=self.gaussian_sigma,
        )

    def _initialise_preprocessing(self, preprocessing=None):
        if preprocessing is not None and not isinstance(preprocessing, Pipeline):
            raise TypeError("preprocessing must be a preprocessing.Pipeline or None.")

        self.preprocessing = preprocessing if preprocessing is not None else self._build_default_preprocessing_pipeline()
        self._preprocessed_cube_cache = None
        self._preprocessed_x_cache = None
        self._preprocess_meta = {}

    def _get_processed_mapping_cube(self):
        if self._preprocessed_cube_cache is not None and self._preprocessed_x_cache is not None:
            return self._preprocessed_x_cache, self._preprocessed_cube_cache

        x_attr = "xdata" if hasattr(self, "xdata") else "wavenumber"
        x_raw = np.asarray(getattr(self, x_attr), dtype=float).ravel()
        cube_raw = np.asarray(self.spectra, dtype=float)

        axis_kind = "energy_eV" if x_attr == "xdata" else "raman_shift_cm-1"
        modality = "PL" if x_attr == "xdata" else "Raman"

        result = apply_pipeline_to_mapping_cube(
            x=x_raw,
            cube=cube_raw,
            pipeline=self.preprocessing,
            modality=modality,
            axis_kind=axis_kind,
            meta={
                "x_trimmed_on_load": bool(getattr(self, "_x_trimmed_on_load", False)),
                "filename": getattr(self, "filename", None),
            },
        )

        self._preprocessed_x_cache = np.asarray(result.x, dtype=float).ravel()
        self._preprocessed_cube_cache = np.asarray(result.cube, dtype=float)
        self._preprocess_meta = dict(result.meta)

        return self._preprocessed_x_cache, self._preprocessed_cube_cache

    def _prepare_fit_spectrum(self, xdata, spec, *, fit_normalize=True):
        y = np.asarray(spec, dtype=float).ravel()

        scale = np.nanmax(y)
        if (not np.isfinite(scale)) or scale <= 0:
            return None, None

        if fit_normalize:
            return y / scale, float(scale)
        else:
            return y, float(scale)

    def _preprocess_single_spectrum(self, xdata, spec, *, fit_normalize=True):
        cube = np.asarray(spec, dtype=float).reshape(1, 1, -1)

        axis_kind = "energy_eV" if hasattr(self, "xdata") else "raman_shift_cm-1"
        modality = "PL" if hasattr(self, "xdata") else "Raman"

        result = apply_pipeline_to_mapping_cube(
            x=np.asarray(xdata, dtype=float).ravel(),
            cube=cube,
            pipeline=self.preprocessing,
            modality=modality,
            axis_kind=axis_kind,
            meta={
                "x_trimmed_on_load": bool(getattr(self, "_x_trimmed_on_load", False)),
                "filename": getattr(self, "filename", None),
            },
        )

        y_proc = np.asarray(result.cube[0, 0, :], dtype=float).ravel()
        x_proc = np.asarray(result.x, dtype=float).ravel()
        return self._prepare_fit_spectrum(x_proc, y_proc, fit_normalize=fit_normalize)