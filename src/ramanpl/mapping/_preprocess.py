import numpy as np

try:
    from ..preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )
    from ..schema import normalise_preprocess_backend
except Exception:  # pragma: no cover
    from preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )
    from schema import normalise_preprocess_backend


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

    def _schema_x_attr(self) -> str:
        return "xdata" if hasattr(self, "xdata") else "wavenumber"

    def _schema_axis_kind(self) -> str:
        return "energy_eV" if self._schema_x_attr() == "xdata" else "raman_shift_cm-1"

    def _schema_modality(self) -> str:
        return "PL" if self._schema_x_attr() == "xdata" else "Raman"

    def _initialise_preprocessing(self, preprocessing=None):
        if preprocessing is not None and not isinstance(preprocessing, Pipeline):
            raise TypeError("preprocessing must be a preprocessing.Pipeline or None.")

        self.preprocessing = (
            preprocessing if preprocessing is not None
            else self._build_default_preprocessing_pipeline()
        )

        try:
            self.preprocessing_recipe = self.preprocessing.to_dict()
        except Exception:
            self.preprocessing_recipe = None

        self._preprocessed_cube_cache = None
        self._preprocessed_x_cache = None
        self._preprocess_meta = {}

    def _get_processed_mapping_cube(self):
        if self._preprocessed_cube_cache is not None and self._preprocessed_x_cache is not None:
            return self._preprocessed_x_cache, self._preprocessed_cube_cache

        x_attr = self._schema_x_attr()
        x_raw = np.asarray(getattr(self, x_attr), dtype=float).ravel()
        cube_raw = np.asarray(self.spectra, dtype=float)

        result = apply_pipeline_to_mapping_cube(
            x=x_raw,
            cube=cube_raw,
            pipeline=self.preprocessing,
            modality=self._schema_modality(),
            axis_kind=self._schema_axis_kind(),
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
        return y, float(scale)

    def _preprocess_single_spectrum(self, xdata, spec, *, fit_normalize=True):
        cube = np.asarray(spec, dtype=float).reshape(1, 1, -1)

        result = apply_pipeline_to_mapping_cube(
            x=np.asarray(xdata, dtype=float).ravel(),
            cube=cube,
            pipeline=self.preprocessing,
            modality=self._schema_modality(),
            axis_kind=self._schema_axis_kind(),
            meta={
                "x_trimmed_on_load": bool(getattr(self, "_x_trimmed_on_load", False)),
                "filename": getattr(self, "filename", None),
            },
        )

        y_proc = np.asarray(result.cube[0, 0, :], dtype=float).ravel()
        x_proc = np.asarray(result.x, dtype=float).ravel()
        return self._prepare_fit_spectrum(x_proc, y_proc, fit_normalize=fit_normalize)