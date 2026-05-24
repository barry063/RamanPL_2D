import numpy as np

try:
    from ..preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )
    from ..schema import (
        baseline_spec_to_runtime,
        normalise_coord_mode,
        normalise_preprocess_backend,
    )
except Exception:  # pragma: no cover
    from preprocessing import (
        Pipeline,
        build_legacy_mapping_pipeline,
        apply_pipeline_to_mapping_cube,
    )
    from schema import (
        baseline_spec_to_runtime,
        normalise_coord_mode,
        normalise_preprocess_backend,
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
            backend=self.preprocessing_backend,
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

        self.preprocessing_backend = normalise_preprocess_backend(
            getattr(self.preprocessing, "backend", self.preprocessing_backend)
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

    def _iter_coords(self, coord_mode: str = "pixel"):
        """
        Yield (x, y, j, i) for every pixel.

        Parameters
        ----------
        coord_mode : {'pixel', 'real'}
            - 'pixel': return integer pixel indices
            - 'real' : return physical coordinates using self.step_size

        Yields
        ------
        tuple
            (x, y, j, i) where:
            - x, y are exported coordinates
            - j, i are array indices into [Y, X]
        """
        coord_mode = normalise_coord_mode(coord_mode)
        step = float(self.step_size)

        for j in range(self.Y):
            for i in range(self.X):
                if coord_mode == "real":
                    yield (i * step, j * step, j, i)
                else:
                    yield (i, j, j, i)

    # ------------------------------------------------------------------
    # v0.6.2 — Baseline autotune façade methods
    # ------------------------------------------------------------------

    def autotune_baseline(
        self,
        *,
        seed_coord: tuple,
        method_grids=None,
        plot: bool = True,
        fit_spectrum_kwargs=None,
    ):
        """
        Score a grid of baseline candidates on a seed pixel. Does NOT modify self.

        Call self.apply_choice(result.winner) to commit the winning configuration.

        Parameters
        ----------
        seed_coord : (j, i)
            Row-major pixel index (j=row, i=col).
        method_grids : dict or None
            Per-method parameter sweep, e.g.
            ``{"arpls": {"lam": [1e4, 1e5], "niter": [50, 100]}}``.
            None → full 24-candidate default grid.
        plot : bool
            If True, return a comparison figure in result.figure.
        fit_spectrum_kwargs : dict or None
            Extra kwargs for the internal fitter (e.g. n_starts).

        Returns
        -------
        BaselineAutotuneResult
        """
        try:
            from .._autotune import autotune_baseline_for_object
        except Exception:
            from ramanpl._autotune import autotune_baseline_for_object

        result = autotune_baseline_for_object(
            self,
            seed_coord=seed_coord,
            method_grids=method_grids,
            plot=plot,
            fit_spectrum_kwargs=fit_spectrum_kwargs,
        )
        self._last_autotune_result = result
        return result

    def apply_choice(self, choice: dict) -> None:
        """
        Commit a baseline spec to self.preprocessing.

        Invalidates the preprocessed-cube cache so the next fit_spectra() call
        uses the new baseline.

        Parameters
        ----------
        choice : dict
            Baseline spec with at least a 'method' key, e.g.
            {'method': 'airpls', 'lam': 1e5, 'niter': 50}.

        Raises
        ------
        ValueError
            If choice is not a valid spec dict, or if the pipeline has zero or
            more than one BaselineSubtract steps.
        """
        if not isinstance(choice, dict) or "method" not in choice:
            raise ValueError(
                "choice must be a baseline spec dict with at least a 'method' key."
            )

        try:
            from .._autotune import _swap_baseline_step_in_pipeline
            from ..schema import baseline_spec_to_runtime, normalise_baseline_spec
        except Exception:
            from ramanpl._autotune import _swap_baseline_step_in_pipeline
            from ramanpl.schema import baseline_spec_to_runtime, normalise_baseline_spec

        new_pipe = _swap_baseline_step_in_pipeline(self.preprocessing, choice)

        # Refresh pipeline state
        self.preprocessing = new_pipe
        try:
            self.preprocessing_recipe = new_pipe.to_dict()
        except Exception:
            self.preprocessing_recipe = None

        # Refresh legacy baseline attributes
        self.baseline_method = normalise_baseline_spec(choice)
        self._baseline_method, self._baseline_kwargs = baseline_spec_to_runtime(
            self.baseline_method
        )

        # Invalidate cube cache so next fit_spectra() re-preprocesses
        self._preprocessed_cube_cache = None
        self._preprocessed_x_cache = None
        self._preprocess_meta = {}

    def _build_mapping_fit_export_meta(self, *, coord_mode: str, scaled: bool, peak_labels):
        """
        Build shared export metadata for mapping fit exports.

        This keeps PLMapping and RamanMapping export metadata aligned.
        """
        coord_mode = normalise_coord_mode(coord_mode)

        try:
            from ..exporter import build_export_meta, serialise_backend_provenance
        except Exception:  # pragma: no cover
            from exporter import build_export_meta, serialise_backend_provenance

        # Use canonical backend provenance from preprocessing result.
        # apply_pipeline_to_mapping_cube() always sets preprocessing_backend_info.
        backend_outcome = self._preprocess_meta.get("preprocessing_backend_info") if self._preprocess_meta else None
        provenance = serialise_backend_provenance(backend_outcome)

        meta = build_export_meta(
            export_kind="mapping_fit",
            map_kind="fit_params",
            spectrum_type=getattr(self, "spectrum_type", None),
            x_quantity=getattr(self, "x_quantity", None),
            x_unit=getattr(self, "x_unit", None),
            coord_mode=coord_mode,
            step_size=getattr(self, "step_size", None),
            step_unit=getattr(self, "step_unit", "um"),
            scaled=scaled,
            peak_labels=peak_labels,
            peak_profile=getattr(self, "peak_profile", None),
            params_per_peak=getattr(self, "params_per_peak", None),
            baseline_spec=getattr(self, "baseline_method", None),
            preprocessing_recipe=getattr(self, "preprocessing_recipe", None),
            background_remove=getattr(self, "background_remove", None),
            smoothing=getattr(self, "smoothing", None),
            smooth_window=getattr(self, "smooth_window", None),
            smooth_poly=getattr(self, "smooth_poly", None),
            **provenance,
        )

        ar = getattr(self, "_last_autotune_result", None)
        if ar is not None:
            meta["baseline_autotune"] = {
                "methods": list({r["method"] for r in ar.ranking}),
                "n_candidates": len(ar.ranking),
                "seed_coord": list(ar.seed_coord) if ar.seed_coord is not None else None,
                "winner": ar.winner,
                "winner_rmse": float(ar.ranking[0]["rmse"]),
                "ranking_top5": ar.ranking[:5],
            }

        return meta

    def _params_to_export_dict(self, xaxis, peak_labels, params, intensity_scale=1.0):
        """
        Convert a parameter vector into per-peak export dict entries.

        Conventions
        -----------
        - Lorentzian:
            width parameter is HWHM
            FWHM = 2 * HWHM
            peak_height_norm = amp_area / (pi * HWHM)
        - pVoigt:
            width parameter is treated as FWHM
            peak_height_norm is computed numerically from single_peak(...)
        """
        try:
            from ..peak_models import single_peak
        except Exception:  # pragma: no cover
            from peak_models import single_peak

        profile = self.peak_profile
        stride = int(self.params_per_peak)
        p = np.asarray(params, dtype=float).ravel()
        xaxis = np.asarray(xaxis, dtype=float).ravel()

        out = {}
        for i, name in enumerate(peak_labels):
            block = p[stride * i: stride * (i + 1)]
            centre = float(block[0])

            if profile == "lorentzian":
                hwhm = float(block[1])
                amp_area = float(block[2])
                fwhm = 2.0 * hwhm
                peak_height_norm = (amp_area / (np.pi * hwhm)) if hwhm != 0 else np.nan
                peak_height = float(peak_height_norm * intensity_scale)

                out[name] = dict(
                    centre=centre,
                    fwhm=fwhm,
                    peak_height=peak_height,
                    peak_height_norm=float(peak_height_norm),
                    amp=amp_area,
                    scale=hwhm,
                )

            else:
                fwhm = float(block[1])   # pVoigt width treated as FWHM
                amp_area = float(block[2])
                eta = float(block[3])

                y_norm = single_peak(xaxis, block, profile="pvoigt")
                peak_height_norm = float(np.nanmax(y_norm))
                peak_height = float(peak_height_norm * intensity_scale)

                out[name] = dict(
                    centre=centre,
                    fwhm=fwhm,
                    peak_height=peak_height,
                    peak_height_norm=peak_height_norm,
                    amp=amp_area,
                    scale=fwhm,
                    eta=eta,
                )

        return out
    
    def _allocate_basic_fit_outputs(self, *, num_peaks: int):
        """
        Allocate fit-related output arrays shared by PLMapping and RamanMapping.

        Parameters
        ----------
        num_peaks : int
            Number of fitted peaks.
        """
        self.peak_positions = np.full((self.Y, self.X, num_peaks), np.nan, dtype=float)
        self.peak_intensities = np.full((self.Y, self.X, num_peaks), np.nan, dtype=float)
        self.fitted_params = np.full(
            (self.Y, self.X, num_peaks * self.params_per_peak),
            np.nan,
            dtype=float,
        )

        self.residual_map = np.full((self.Y, self.X), np.nan, dtype=float)
        self.norm_scale_map = np.full((self.Y, self.X), np.nan, dtype=float)

    def _qa_columns_for_pixel(self, j: int, i: int, n_peaks: int) -> dict:
        """
        Return per-pixel QA columns for export.

        Always returns the same four keys regardless of fit success or
        diagnostics mode: rmse, ok, n_starts, n_params_at_bounds.

        Failed pixels return rmse=NaN, ok=False, n_starts=NaN,
        n_params_at_bounds=NaN.  When fit_diagnostics_map is None
        (diagnostics='none'), n_starts and n_params_at_bounds are NaN
        but rmse and ok still resolve from residual_map.
        """
        _nan = float("nan")

        rmse = float(self.residual_map[j, i])
        ok = bool(np.isfinite(rmse))

        diag = None
        if getattr(self, "fit_diagnostics_map", None) is not None:
            diag = self.fit_diagnostics_map[j, i]

        if diag is not None and isinstance(diag, dict):
            n_starts = diag.get("n_starts", _nan)
            n_at_bounds = (
                diag.get("n_params_at_lower_bounds", 0)
                + diag.get("n_params_at_upper_bounds", 0)
            )
        else:
            n_starts = _nan
            n_at_bounds = _nan

        return {
            "rmse": rmse,
            "ok": ok,
            "n_starts": n_starts,
            "n_params_at_bounds": n_at_bounds,
        }

    def feature_table(self, *, coord_mode="pixel", scaled=True, ratios=None, separations=None):
        """
        Return a wide-format DataFrame of per-pixel peak descriptors and QA columns.

        Parameters
        ----------
        coord_mode : {'pixel', 'real'}
            Coordinate convention for ``x``/``y`` columns.
        scaled : bool
            If True, peak heights are in physical intensity units (scaled by the
            per-pixel normalisation factor). If False, normalised-space heights.
        ratios : list of (str, str) or None
            Each ``(P1, P2)`` adds a ``{P1}_{P2}_ratio`` column
            (peak_height[P1] / peak_height[P2]; zero denominator → NaN).
        separations : list of (str, str) or None
            Each ``(P1, P2)`` adds a ``{P1}_{P2}_separation`` column
            (position[P1] − position[P2]).

        Returns
        -------
        pandas.DataFrame
            One row per pixel; columns: x, y, per-peak descriptors, derived
            columns (if requested), rmse, ok, n_starts, n_params_at_bounds.

        Raises
        ------
        ValueError
            If no fit has been run, or if a pair references an unknown peak label.
        """
        if not hasattr(self, "fitted_params") or self.fitted_params is None:
            raise ValueError("No fitted_params; run fit_spectra() first.")

        try:
            from .. import descriptors
        except Exception:  # pragma: no cover
            import descriptors

        import pandas as pd

        peak_labels = list(self.peak_params)
        descriptors.validate_peak_pairs(
            list(ratios or []) + list(separations or []), peak_labels
        )

        xaxis = np.asarray(getattr(self, self._schema_x_attr()), dtype=float).ravel()

        rows = []
        for x, y_coord, j, i in self._iter_coords(coord_mode=coord_mode):
            params = np.asarray(self.fitted_params[j, i, :], dtype=float)
            qa = self._qa_columns_for_pixel(j, i, len(peak_labels))
            failed = bool(np.any(np.isnan(params)))

            if failed:
                row = {"x": x, "y": y_coord}
                for name in peak_labels:
                    row[f"{name}_position"] = float("nan")
                    row[f"{name}_fwhm"] = float("nan")
                    row[f"{name}_peak_height"] = float("nan")
                    row[f"{name}_peak_height_norm"] = float("nan")
                for p1, p2 in (separations or []):
                    row[f"{p1}_{p2}_separation"] = float("nan")
                for p1, p2 in (ratios or []):
                    row[f"{p1}_{p2}_ratio"] = float("nan")
                row.update(qa)
            else:
                intensity_scale = 1.0
                if (
                    scaled
                    and hasattr(self, "norm_scale_map")
                    and np.isfinite(self.norm_scale_map[j, i])
                ):
                    intensity_scale = float(self.norm_scale_map[j, i])

                per_peak = self._params_to_export_dict(
                    xaxis, peak_labels, params, intensity_scale=intensity_scale
                )
                feat = descriptors.build_feature_row(
                    per_peak, qa, peak_labels,
                    ratios=ratios, separations=separations,
                )
                row = {"x": x, "y": y_coord, **feat}

            rows.append(row)

        return pd.DataFrame.from_records(rows)

    def _initialise_mapping_fit_common(
        self,
        *,
        filename,
        custom_peaks,
        data_range,
        step_size,
        poly_degree,
        normalize,
        background_remove,
        baseline_method,
        smoothing,
        smooth_window,
        smooth_poly,
        gaussian_sigma,
        peak_profile,
        preprocessing_backend,
        preprocessing,
        spectrum_type,
        x_quantity,
        x_unit,
    ):
        """
        Initialise shared state for PLMapping and RamanMapping fit classes.

        This helper only sets common attributes and preprocessing/model state.
        It does not load data, validate array shapes, or allocate outputs.
        """
        try:
            from ..baselineAPI import BaselineAPI
            from ..schema import (
                normalise_baseline_spec,
                normalise_peak_profile,
                normalise_preprocess_backend,
            )
        except Exception:  # pragma: no cover
            from baselineAPI import BaselineAPI
            from schema import (
                normalise_baseline_spec,
                normalise_peak_profile,
                normalise_preprocess_backend,
            )

        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.background_remove = background_remove
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma,
        )
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.preprocessing_backend = normalise_preprocess_backend(preprocessing_backend)
        self.peak_params = list(custom_peaks.keys())

        # identity metadata
        self.spectrum_type = spectrum_type
        self.x_quantity = x_quantity
        self.x_unit = x_unit
        self.step_unit = "um"

        if str(self.baseline_method.get("method", "")).lower() == "poly":
            self.poly_order = int(self.baseline_method.get("poly_order", poly_degree))
        else:
            self.poly_order = None

        # baseline runtime config (baseline_method is already normalised above)
        self._baseline_method, self._baseline_kwargs = baseline_spec_to_runtime(
            self.baseline_method
        )

        # peak model
        self.peak_profile = normalise_peak_profile(peak_profile)
        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        # preprocessing pipeline
        self._initialise_preprocessing(preprocessing=preprocessing)
        self.preprocessing_backend_resolved = None
        self.preprocessing_backend_info = None