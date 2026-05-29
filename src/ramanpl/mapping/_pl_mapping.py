from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import simpson
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

try:
    from ..baselineAPI import BaselineAPI
    from ..dataImporter import DataImporter
    from ..exporter import build_export_meta, write_table
    from ..peak_models import single_peak, sum_peaks
    from ..schema import (
        baseline_spec_to_runtime,
        normalise_baseline_spec,
        normalise_coord_mode,
        normalise_peak_profile,
        normalise_preprocess_backend,
    )
    from ._diagnostics import fit_summary as _fit_summary
    from ._fit_utils import (
        _params_at_bounds,
        _run_mapping_curve_fit_trials,
        seed_p0_from_coord,
    )
    from ._io import MappingFileLoader
    from ._image import MappingImage
    from ._preprocess import _MappingPreprocessMixin
    from ._parallel import (
        _split_rows,
        _validate_parallel_kwargs,
        _merge_band_outputs,
        _pl_fit_band,
    )
    from ._cluster_seeds import (
        _normalise_cluster_seed_config,
        _cluster_spectra,
        _representative_pixels,
        _build_cluster_schedule,
    )
except Exception:  # pragma: no cover
    from ramanpl.baselineAPI import BaselineAPI
    from ramanpl.dataImporter import DataImporter
    from ramanpl.exporter import build_export_meta, write_table
    from ramanpl.peak_models import single_peak, sum_peaks
    from ramanpl.schema import (
        baseline_spec_to_runtime,
        normalise_baseline_spec,
        normalise_coord_mode,
        normalise_peak_profile,
        normalise_preprocess_backend,
    )
    from ramanpl.mapping._diagnostics import fit_summary as _fit_summary
    from ._fit_utils import (
        _params_at_bounds,
        _run_mapping_curve_fit_trials,
        seed_p0_from_coord,
    )
    from ramanpl.mapping._io import MappingFileLoader
    from ramanpl.mapping._image import MappingImage
    from ramanpl.mapping._preprocess import _MappingPreprocessMixin
    from ramanpl.mapping._parallel import (
        _split_rows,
        _validate_parallel_kwargs,
        _merge_band_outputs,
        _pl_fit_band,
    )
    from ramanpl.mapping._cluster_seeds import (
        _normalise_cluster_seed_config,
        _cluster_spectra,
        _representative_pixels,
        _build_cluster_schedule,
    )


#########################################################################################################################
#################################################### PL Mapping #########################################################
#########################################################################################################################

class PLMapping(_MappingPreprocessMixin):
    """Photoluminescence mapping analysis through Lorentzian peak fitting.
    
    Attributes:
        filename (str): Path to .wdf file
        custom_peaks (dict): Peak parameters for fitting
        data_range (tuple): Spectral analysis range (min, max) in eV
        step_size (float): Physical step size in micrometers
        poly_degree (int): Polynomial degree for background removal
        normalize (bool): Enable spectrum normalization
        background_remove (bool): Enable background subtraction
        baseline_method (str): Background method ('poly' or 'gaussian')
        smoothing (bool): Enable spectral smoothing
        smooth_window (int): Savitzky-Golay window size
        smooth_poly (int): Savitzky-Golay polynomial order
        gaussian_sigma (int): Gaussian filter width
        peak_params (list): Peak names from custom_peaks
        X (int): Map width in pixels
        Y (int): Map height in pixels
        xdata (ndarray): Spectral axis in eV
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        peak_positions (ndarray): Fitted peak centers [Y, X, peaks]
        peak_intensities (ndarray): Fitted peak amplitudes [Y, X, peaks]
        fitted_params (ndarray): Full fitting parameters [Y, X, 3*peaks]
        residual_map (ndarray): Fitting residuals [Y, X]
    """

    def __init__(
        self,
        filename,
        custom_peaks,
        data_range=None,
        step_size=0.3,
        poly_degree=3,
        normalize=True,
        background_remove=True,
        baseline_method="poly",
        smoothing=True,
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
        peak_profile: str = "lorentzian",
        preprocessing_backend: str = "native",
        preprocessing=None,
    ):
        """Initialize PL mapping analyzer.
        
        Args:
            filename: Path to .wdf PL mapping file
            custom_peaks: Peak definitions with bounds {name: (min_params, max_params)}
            data_range: Spectral range (min, max) in eV (default: full spectrum)
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            normalize (bool): Controls display/output scaling only. Fitting is always performed on peak-normalised spectra.
            background_remove: Enable background subtraction
            baseline_method: 'poly' or 'gaussian' background
            smoothing: Enable spectral smoothing
            smooth_window: Savitzky-Golay window size
            smooth_poly: Savitzky-Golay polynomial order
            gaussian_sigma: Gaussian filter width
            peak_profile: str = "lorentzian" or "pvoigt"
            preprocessing: Optional custom preprocessing pipeline (overrides legacy flag-based pipeline if provided)
        """
        self._initialise_mapping_fit_common(
            filename=filename,
            custom_peaks=custom_peaks,
            data_range=data_range,
            step_size=step_size,
            poly_degree=poly_degree,
            normalize=normalize,
            background_remove=background_remove,
            baseline_method=baseline_method,
            smoothing=smoothing,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
            gaussian_sigma=gaussian_sigma,
            peak_profile=peak_profile,
            preprocessing_backend=preprocessing_backend,
            preprocessing=preprocessing,
            spectrum_type="Photoluminescence",
            x_quantity="Photon energy",
            x_unit="eV",
        )

        # ---- load mapping data (optionally trimmed) ----
        if self.data_range is not None:
            loader = MappingFileLoader(filename, x_range=self.data_range, axis="energy")
            self._x_trimmed_on_load = True
        else:
            loader = MappingFileLoader(filename, axis="energy")
            self._x_trimmed_on_load = False

        self.X = loader.X
        self.Y = loader.Y
        self.xdata = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None

        if self.data_range is None:
            self.data_range = (float(np.min(self.xdata)), float(np.max(self.xdata)))

        # ---- allocate output arrays ----
        self._allocate_fit_outputs()

    ### NEW METHOD IN v0.3.5 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        custom_peaks,
        data_range=None,
        step_size=0.3,
        poly_degree=3,
        normalize=True,
        background_remove=True,
        baseline_method='poly',
        smoothing=True,
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
        peak_profile: str = "lorentzian",
        preprocessing_backend: str = "native",
        preprocessing=None,
    ):
        """
        Create a PLMapping instance from in-memory mapping arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Spectral axis with shape [N] (energy in eV)
        X, Y : int
            Map dimensions
        """

        obj = cls.__new__(cls)

        # ---- mirror __init__ fields ----
        obj._initialise_mapping_fit_common(
            filename=None,
            custom_peaks=custom_peaks,
            data_range=data_range,
            step_size=step_size,
            poly_degree=poly_degree,
            normalize=normalize,
            background_remove=background_remove,
            baseline_method=baseline_method,
            smoothing=smoothing,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
            gaussian_sigma=gaussian_sigma,
            peak_profile=peak_profile,
            preprocessing_backend=preprocessing_backend,
            preprocessing=preprocessing,
            spectrum_type="Photoluminescence",
            x_quantity="Photon energy",
            x_unit="eV",
        )

        # ---- assign data ----
        obj.X = int(X)
        obj.Y = int(Y)
        obj.xdata = np.asarray(xdata, dtype=float).ravel()
        obj.spectra = np.asarray(spectra, dtype=float)

        # Validate shapes
        if obj.spectra.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if obj.spectra.shape[0] != obj.Y or obj.spectra.shape[1] != obj.X:
            raise ValueError(f"spectra shape {obj.spectra.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if obj.spectra.shape[2] != obj.xdata.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # No optical image when constructed from arrays
        obj.image_viewer = None

        # Load-time trimming flag: False, because we did not trim during import here
        obj._x_trimmed_on_load = False

        # Default range: full axis if not supplied
        if obj.data_range is None:
            obj.data_range = (float(np.min(obj.xdata)), float(np.max(obj.xdata)))

        # obj.peak_profile = normalise_peak_profile(peak_profile)
        # obj.params_per_peak = 3 if obj.peak_profile == "lorentzian" else 4
        # obj._initialise_preprocessing(preprocessing=preprocessing)
        # obj.preprocessing_backend_resolved = None
        # obj.preprocessing_backend_info = None

        # Allocate output arrays (same as __init__)
        obj._allocate_fit_outputs()

        return obj

    ### New in v0.2.7 ###
    def get_reference_spectrum(self, *, x: int, y: int, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Return a reference spectrum from this already-loaded mapping object.

        Parameters
        ----------
        x, y : int
            Pixel coordinate (0-indexed)
        roi : (x0, x1, y0, y1) inclusive, optional
            If provided, returns the mean spectrum over the ROI.

        Returns
        -------
        (y_ref, xdata)
        """
        if roi is not None:
            x0, x1, y0, y1 = roi
            if not (0 <= x0 <= x1 < self.X and 0 <= y0 <= y1 < self.Y):
                raise ValueError("ROI out of bounds.")
            y_ref = np.nanmean(self.spectra[y0:y1+1, x0:x1+1, :], axis=(0, 1))
        else:
            if not (0 <= x < self.X and 0 <= y < self.Y):
                raise ValueError("Pixel out of bounds.")
            y_ref = self.spectra[y, x, :]

        return np.asarray(y_ref, dtype=float).ravel(), np.asarray(self.xdata, dtype=float).ravel()

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def fit_summary(self, **kwargs):
        """Backward-compatible instance wrapper for mapping fit diagnostics."""
        return _fit_summary(self, **kwargs)
    
    def _allocate_fit_outputs(self):
        """
        Allocate fit-related output arrays so file-based and array-based
        constructors produce the same object state before fitting.
        """
        self._allocate_basic_fit_outputs(num_peaks=len(self.custom_peaks))
    
    def lorentzian(self, x, *params):
        """Multi-Lorentzian function for curve fitting."""
        return sum_peaks(np.asarray(x), params, profile="lorentzian", stride=3)


    ### UPDATED METHOD in v0.2.5 ##
    def remove_background(self, xdata, intensity):
        """Remove spectral background via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=xdata,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected

    # New in v0.3.3
    def _model_dispatch(self):
        if self.peak_profile == "lorentzian":
            return self.lorentzian
        return self.pvoigt  # you must implement pvoigt analogous to lorentzian using peak_models.sum_peaks

    def pvoigt(self, x, *params):
        return sum_peaks(np.asarray(x), params, profile="pvoigt", stride=4)

    ## UPDATED METHOD in v0.3.3 ##
    def fit_spectra(
        self,
        initial_p0=None,
        warm_start=False,
        seed_coord=None,
        seed_roi=None,
        reset_on_fail=True,
        row_reset=True,
        warm_start_rmse_gate=0.06,
        maxfev = 6400,
        fit_spectrum_kwargs=None,
        fit_normalize=True,
        compute_peak_maps=True,
        show_progress=True,
        n_jobs=1,
        cluster_seeds=False,
    ):
        """
        Fit all map spectra using self.custom_peaks as bounds.

        Parameters
        ----------
        initial_p0 : array-like or dict or None
            Optional initial guess vector (e.g., from a single-point PLfit result),
            or dict package {"p0": <vector>, "peak_order": <list>}.
            Must match parameter ordering implied by self.custom_peaks.
        warm_start : bool
            If True, use previous successful fit parameters as p0 for next pixel.
        reset_on_fail : bool
            If True, on fit failure reset p0 to baseline (midpoint/initial_p0).
        maxfev : int
            curve_fit maximum function evaluations.
        warm_start_rmse_gate : float
            RMSE threshold (normalised space) for accepting warm-start propagation.
        fit_normalize : bool
            If True, internal scaling for optimisation is normalised
        compute_peak_maps : bool
            If True, compute the peak maps and store them internally
        n_jobs : int
            Number of parallel loky workers for the row loop (v0.6.4+).
            Default 1 preserves byte-parity with v0.6.3.
            n_jobs > 1 with warm_start=True requires row_reset=True.
        Returns
        -------
        params_map : ndarray
            Fitted parameter cube with shape [Y, X, n_params].
            In notebooks, assign the return value or use `_ = fit_spectra(...)`
            to avoid auto-display.

        """
        if fit_spectrum_kwargs is None:
            fit_spectrum_kwargs = {}
        diagnostics_mode = str(
            fit_spectrum_kwargs.get("diagnostics", "full")
        ).lower().strip()
        if diagnostics_mode not in {"full", "light", "none"}:
            raise ValueError("fit_spectrum_kwargs['diagnostics'] must be 'full', 'light', or 'none'.")
        self.diagnostics_mode = diagnostics_mode

        if not hasattr(self, "custom_peaks") or not isinstance(self.custom_peaks, dict) or len(self.custom_peaks) == 0:
            raise ValueError("custom_peaks is not set or empty. Provide custom_peaks when initialising PLMapping.")

        cluster_seed_cfg = _normalise_cluster_seed_config(cluster_seeds, X=self.X, Y=self.Y)
        if cluster_seed_cfg is not None and n_jobs != 1:
            raise ValueError(
                "cluster_seeds=True requires n_jobs=1 in v0.6.5. "
                "Either set n_jobs=1, or set cluster_seeds=False."
            )

        # --- Shared preprocessing path (crop + smoothing + baseline) ---
        xdata, spectra_fit_cube = self._get_processed_mapping_cube()

        # expose resolved preprocessing backend on the mapping object
        self.preprocessing_backend_resolved = self._preprocess_meta.get("preprocessing_backend", None)
        self.preprocessing_backend_info = self._preprocess_meta.get("preprocessing_backend_info", None)

        # --- Build bounds from custom_peaks (in insertion order)
        lower_bound, upper_bound = [], []
        for params_range in self.custom_peaks.values():
            lower_bound.extend(params_range[0])
            upper_bound.extend(params_range[1])

        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        n_params = lower_bound.size

        # Default p0: midpoint of bounds
        p0_base = (lower_bound + upper_bound) / 2.0

        # --- optional: seed from a coordinate inside this mapping object ---
        if cluster_seed_cfg is not None and seed_coord is not None:
            raise ValueError(
                "cluster_seeds=True and seed_coord are mutually exclusive. "
                "Either drop seed_coord, or set cluster_seeds=False."
            )
        if seed_coord is not None:
            if initial_p0 is not None:
                raise ValueError("Provide either initial_p0 or seed_coord, not both.")

            initial_p0 = seed_p0_from_coord(self, seed_coord, seed_roi, maxfev=maxfev)
            warm_start = True

        # Optional: seed from single-point PLfit
        if initial_p0 is not None:

            if isinstance(initial_p0, dict):
                peak_order_pkg = initial_p0.get("peak_order", None)
                p0_vec = initial_p0.get("p0", None)

                if p0_vec is None:
                    raise ValueError("initial_p0 dict must contain key 'p0' with a numeric vector.")

                if peak_order_pkg is not None:
                    if [p.lower() for p in peak_order_pkg] != [p.lower() for p in self.peak_params]:
                        raise ValueError(
                            "peak_order mismatch between PLfit and PLMapping.\n"
                            f"PLfit: {list(peak_order_pkg)}\n"
                            f"PLMapping: {list(self.peak_params)}\n"
                            "Ensure both use the same custom_peaks ordering (or pass peak_order explicitly)."
                        )

                initial_p0 = p0_vec

            initial_p0 = np.asarray(initial_p0, dtype=float)
            if initial_p0.shape != p0_base.shape:
                raise ValueError(f"initial_p0 shape {initial_p0.shape} does not match expected {p0_base.shape}")

            p0_base = np.clip(initial_p0, lower_bound, upper_bound)

        p0_current = p0_base.copy()

        # Output arrays (added in v0.3.0)
        fitted_params = np.full((self.Y, self.X, n_params), np.nan)
        if diagnostics_mode == "none":
            self.fit_diagnostics_map = None
        else:
            self.fit_diagnostics_map = np.empty((self.Y, self.X), dtype=object)
            self.fit_diagnostics_map[:, :] = None

        # Defensive fallback only; normally these are allocated by constructor/from_arrays
        if not hasattr(self, "norm_scale_map"):
            self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        if not hasattr(self, "residual_map"):
            self.residual_map = np.full((self.Y, self.X), np.nan)
        
        model_fn = self._model_dispatch()
        stride = int(self.params_per_peak)

        # ------------------------------------------------------------
        # Adaptive fitting policy
        # ------------------------------------------------------------
        adaptive_multistart = bool(fit_spectrum_kwargs.get("adaptive_multistart", True))

        # Fast first pass
        fast_n_starts = max(1, int(fit_spectrum_kwargs.get("fast_n_starts", 1)))
        fast_p0_strategy = fit_spectrum_kwargs.get("fast_p0_strategy", "midpoint")
        fast_random_state = fit_spectrum_kwargs.get(
            "fast_random_state",
            fit_spectrum_kwargs.get("random_state", None),
        )

        # Fallback pass
        fallback_n_starts = max(1, int(fit_spectrum_kwargs.get("n_starts", 4)))
        fallback_p0_strategy = fit_spectrum_kwargs.get("p0_strategy", "jitter")
        fallback_random_state = fit_spectrum_kwargs.get("random_state", None)

        retry_on_fail = bool(fit_spectrum_kwargs.get("retry_on_fail", True))
        retry_on_high_rmse = bool(fit_spectrum_kwargs.get("retry_on_high_rmse", True))
        retry_on_bound_hit = bool(fit_spectrum_kwargs.get("retry_on_bound_hit", True))
        retry_rmse_gate = float(
            fit_spectrum_kwargs.get("retry_rmse_gate", warm_start_rmse_gate)
        )

        width_penalty = float(fit_spectrum_kwargs.get("width_penalty", 0.0))
        prefer_nonbound = bool(fit_spectrum_kwargs.get("prefer_nonbound", False))
        score_tie_tol = float(fit_spectrum_kwargs.get("score_tie_tol", 1e-6))

        n_jobs = _validate_parallel_kwargs(n_jobs, warm_start, row_reset, self.Y)

        _common_pixel_kwargs = dict(
            fitted_params=fitted_params, spectra_fit_cube=spectra_fit_cube, xdata=xdata,
            lower_bound=lower_bound, upper_bound=upper_bound, n_params=n_params,
            p0_base=p0_base, model_fn=model_fn, stride=stride,
            warm_start=warm_start, warm_start_rmse_gate=warm_start_rmse_gate,
            reset_on_fail=reset_on_fail, fit_normalize=fit_normalize, maxfev=maxfev,
            adaptive_multistart=adaptive_multistart,
            fast_n_starts=fast_n_starts, fast_p0_strategy=fast_p0_strategy,
            fast_random_state=fast_random_state,
            fallback_n_starts=fallback_n_starts, fallback_p0_strategy=fallback_p0_strategy,
            fallback_random_state=fallback_random_state,
            retry_on_fail=retry_on_fail, retry_on_high_rmse=retry_on_high_rmse,
            retry_on_bound_hit=retry_on_bound_hit, retry_rmse_gate=retry_rmse_gate,
            width_penalty=width_penalty, prefer_nonbound=prefer_nonbound,
            score_tie_tol=score_tie_tol, diagnostics_mode=diagnostics_mode,
        )

        if cluster_seed_cfg is not None:
            labels, cs_meta = _cluster_spectra(
                spectra_fit_cube,
                n_clusters=cluster_seed_cfg["n_clusters"],
                n_components=cluster_seed_cfg["n_components"],
                random_state=cluster_seed_cfg["random_state"],
            )
            reps = _representative_pixels(spectra_fit_cube, labels, cs_meta)
            schedule = _build_cluster_schedule(labels, reps)
            pbar = tqdm(
                total=self.Y * self.X,
                desc="Fitting (PL mapping, cluster seeds)",
                disable=not show_progress,
                mininterval=0.5,
            )
            for entry in schedule:
                sx, sy = entry["seed"]
                self._fit_single_pixel(sy, sx, p0_current=p0_base.copy(), **_common_pixel_kwargs)
                pbar.update(1)
                rep_params = fitted_params[sy, sx, :]
                rep_rmse = self.residual_map[sy, sx]
                cluster_p0 = (
                    rep_params.copy()
                    if (np.all(np.isfinite(rep_params))
                        and np.isfinite(rep_rmse)
                        and rep_rmse <= warm_start_rmse_gate)
                    else p0_base.copy()
                )
                p0_member = cluster_p0.copy()
                for mx, my in entry["members"]:
                    start_p0 = p0_member.copy() if warm_start else cluster_p0.copy()
                    p0_member = self._fit_single_pixel(my, mx, p0_current=start_p0, **_common_pixel_kwargs)
                    pbar.update(1)
            pbar.close()
        elif n_jobs == 1:
            pbar = tqdm(
                total=self.Y * self.X,
                desc="Fitting (PL mapping)",
                disable=not show_progress,
                mininterval=0.5,
            )
            self._fit_rows(
                0, self.Y,
                fitted_params=fitted_params,
                spectra_fit_cube=spectra_fit_cube,
                xdata=xdata,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                n_params=n_params,
                p0_base=p0_base,
                p0_start=p0_current,
                model_fn=model_fn,
                stride=stride,
                warm_start=warm_start,
                row_reset=row_reset,
                warm_start_rmse_gate=warm_start_rmse_gate,
                reset_on_fail=reset_on_fail,
                fit_normalize=fit_normalize,
                maxfev=maxfev,
                adaptive_multistart=adaptive_multistart,
                fast_n_starts=fast_n_starts,
                fast_p0_strategy=fast_p0_strategy,
                fast_random_state=fast_random_state,
                fallback_n_starts=fallback_n_starts,
                fallback_p0_strategy=fallback_p0_strategy,
                fallback_random_state=fallback_random_state,
                retry_on_fail=retry_on_fail,
                retry_on_high_rmse=retry_on_high_rmse,
                retry_on_bound_hit=retry_on_bound_hit,
                retry_rmse_gate=retry_rmse_gate,
                width_penalty=width_penalty,
                prefer_nonbound=prefer_nonbound,
                score_tie_tol=score_tie_tol,
                diagnostics_mode=diagnostics_mode,
                pbar=pbar,
            )
            pbar.close()
        else:
            from joblib import Parallel, delayed
            bands = _split_rows(self.Y, n_jobs)
            cfg = dict(
                X=self.X,
                n_params=n_params,
                n_peaks=len(self.peak_params),
                p0_base=p0_base,
                p0_start=p0_current,
                peak_profile=self.peak_profile,
                stride=stride,
                normalize=self.normalize,
                warm_start=warm_start,
                row_reset=row_reset,
                warm_start_rmse_gate=warm_start_rmse_gate,
                reset_on_fail=reset_on_fail,
                fit_normalize=fit_normalize,
                maxfev=maxfev,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                adaptive_multistart=adaptive_multistart,
                fast_n_starts=fast_n_starts,
                fast_p0_strategy=fast_p0_strategy,
                fast_random_state=fast_random_state,
                fallback_n_starts=fallback_n_starts,
                fallback_p0_strategy=fallback_p0_strategy,
                fallback_random_state=fallback_random_state,
                retry_on_fail=retry_on_fail,
                retry_on_high_rmse=retry_on_high_rmse,
                retry_on_bound_hit=retry_on_bound_hit,
                retry_rmse_gate=retry_rmse_gate,
                width_penalty=width_penalty,
                prefer_nonbound=prefer_nonbound,
                score_tie_tol=score_tie_tol,
                diagnostics_mode=diagnostics_mode,
            )
            _gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                delayed(_pl_fit_band)(
                    j_start, j_end,
                    spectra_fit_cube[j_start:j_end],
                    xdata,
                    cfg,
                )
                for j_start, j_end in bands
            )
            band_results = list(tqdm(
                _gen,
                total=len(bands),
                desc=f"Fitting (PL mapping, {n_jobs} workers)",
                disable=not show_progress,
                mininterval=0.5,
            ))
            _merge_band_outputs(self, fitted_params, bands, band_results)

        n_fit = np.sum(~np.isnan(self.residual_map))
        print(f"Successful fits: {n_fit} / {self.X * self.Y}")

        self.fitted_params = fitted_params
        return fitted_params

    def _fit_single_pixel(
        self, j, i, *,
        fitted_params, spectra_fit_cube, xdata,
        lower_bound, upper_bound, n_params,
        p0_current, p0_base,
        model_fn, stride,
        warm_start, warm_start_rmse_gate, reset_on_fail,
        fit_normalize, maxfev,
        adaptive_multistart,
        fast_n_starts, fast_p0_strategy, fast_random_state,
        fallback_n_starts, fallback_p0_strategy, fallback_random_state,
        retry_on_fail, retry_on_high_rmse, retry_on_bound_hit, retry_rmse_gate,
        width_penalty, prefer_nonbound, score_tie_tol,
        diagnostics_mode,
    ):
        """Fit a single pixel (j, i), store all results, return new p0_current."""

        def _store(payload):
            if self.fit_diagnostics_map is None:
                return
            if diagnostics_mode == "full":
                self.fit_diagnostics_map[j, i] = payload
                return
            light = {"ok": payload.get("ok")}
            for k in ("reason", "rmse", "n_starts", "n_fail", "p0_strategy",
                      "adaptive_retry_used", "n_params_at_lower_bounds",
                      "n_params_at_upper_bounds"):
                if k in payload:
                    light[k] = payload[k]
            self.fit_diagnostics_map[j, i] = light

        y = np.asarray(spectra_fit_cube[j, i, :], dtype=float)
        y_fitspace, s = self._prepare_fit_spectrum(xdata, y, fit_normalize=fit_normalize)

        if y_fitspace is None:
            self.norm_scale_map[j, i] = np.nan
            self.residual_map[j, i] = np.nan
            _store({"ok": False, "reason": "no_positive_signal"})
            return p0_base.copy() if reset_on_fail else p0_current.copy()

        self.norm_scale_map[j, i] = float(s)

        retry_used = False
        retry_result = None

        if adaptive_multistart:
            quick_result = _run_mapping_curve_fit_trials(
                model_fn=model_fn, x=xdata, y=y_fitspace,
                lower_bound=lower_bound, upper_bound=upper_bound,
                p0_current=p0_current, maxfev=maxfev,
                n_starts=fast_n_starts, p0_strategy=fast_p0_strategy,
                random_state=fast_random_state, width_penalty=width_penalty,
                prefer_nonbound=prefer_nonbound, score_tie_tol=score_tie_tol,
                peak_profile=self.peak_profile, stride=stride,
            )
            fit_result = quick_result
            need_retry = (not quick_result["ok"] and retry_on_fail) or \
                         (quick_result["ok"] and retry_on_high_rmse and quick_result["best_rmse"] > retry_rmse_gate) or \
                         (quick_result["ok"] and retry_on_bound_hit and quick_result["best_hits"] > 0)
            if need_retry and fallback_n_starts > 1:
                retry_used = True
                retry_result = _run_mapping_curve_fit_trials(
                    model_fn=model_fn, x=xdata, y=y_fitspace,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    p0_current=p0_current, maxfev=maxfev,
                    n_starts=fallback_n_starts, p0_strategy=fallback_p0_strategy,
                    random_state=fallback_random_state, width_penalty=width_penalty,
                    prefer_nonbound=prefer_nonbound, score_tie_tol=score_tie_tol,
                    peak_profile=self.peak_profile, stride=stride,
                )
                if retry_result["ok"]:
                    if not fit_result["ok"]:
                        fit_result = retry_result
                    else:
                        better = retry_result["best_score"] < fit_result["best_score"]
                        near_tie = abs(retry_result["best_score"] - fit_result["best_score"]) <= score_tie_tol
                        if better or (prefer_nonbound and near_tie and
                                      retry_result["best_hits"] < fit_result["best_hits"]):
                            fit_result = retry_result
                elif not fit_result["ok"]:
                    fit_result = retry_result
            n_fail_total = quick_result["n_fail"] + (retry_result["n_fail"] if retry_result else 0)
            n_starts_total = fast_n_starts + (fallback_n_starts if retry_used else 0)
        else:
            fit_result = _run_mapping_curve_fit_trials(
                model_fn=model_fn, x=xdata, y=y_fitspace,
                lower_bound=lower_bound, upper_bound=upper_bound,
                p0_current=p0_current, maxfev=maxfev,
                n_starts=fallback_n_starts, p0_strategy=fallback_p0_strategy,
                random_state=fallback_random_state, width_penalty=width_penalty,
                prefer_nonbound=prefer_nonbound, score_tie_tol=score_tie_tol,
                peak_profile=self.peak_profile, stride=stride,
            )
            n_fail_total = fit_result["n_fail"]
            n_starts_total = fit_result["n_starts"]

        strategy = "adaptive" if adaptive_multistart else fallback_p0_strategy

        if not fit_result["ok"]:
            fitted_params[j, i, :] = np.nan
            self.residual_map[j, i] = np.nan
            _store({"ok": False, "n_starts": n_starts_total, "n_fail": n_fail_total,
                    "p0_strategy": strategy, "adaptive_retry_used": bool(retry_used)})
            return p0_base.copy() if reset_on_fail else p0_current.copy()

        best_params = fit_result["best_params"]
        best_rmse = fit_result["best_rmse"]
        best_p0 = fit_result["best_p0"]

        if best_params is None:
            fitted_params[j, i, :] = np.nan
            self.residual_map[j, i] = np.nan
            _store({"ok": False, "n_starts": n_starts_total, "n_fail": n_fail_total,
                    "p0_strategy": strategy, "adaptive_retry_used": bool(retry_used)})
            return p0_base.copy() if reset_on_fail else p0_current.copy()

        fitted_params[j, i, :] = best_params
        self.residual_map[j, i] = float(best_rmse)
        at_lo = _params_at_bounds(best_params, lower_bound, upper_bound, which="lower", rtol=1e-6)
        at_hi = _params_at_bounds(best_params, lower_bound, upper_bound, which="upper", rtol=1e-6)

        payload = {
            "ok": True, "rmse": float(best_rmse), "n_starts": n_starts_total,
            "n_fail": n_fail_total, "p0_strategy": strategy,
            "adaptive_retry_used": bool(retry_used),
            "n_params_at_lower_bounds": int(np.count_nonzero(at_lo)),
            "n_params_at_upper_bounds": int(np.count_nonzero(at_hi)),
        }
        if diagnostics_mode == "full":
            payload["best_p0"] = np.asarray(best_p0, dtype=float)
            payload["params_at_lower_bounds_mask"] = at_lo
            payload["params_at_upper_bounds_mask"] = at_hi
        _store(payload)

        n_peaks = len(self.peak_params)
        for k in range(n_peaks):
            block = np.asarray(best_params[stride * k:stride * (k + 1)], dtype=float)
            centre = float(block[0])
            self.peak_positions[j, i, k] = centre
            if self.peak_profile == "lorentzian":
                hwhm = float(block[1])
                amp_area = float(block[2])
                hf = amp_area / (np.pi * hwhm) if (np.isfinite(hwhm) and hwhm > 0) else np.nan
            else:
                y_one = single_peak(xdata, block, profile="pvoigt")
                hf = float(np.nanmax(y_one))
            if self.normalize:
                self.peak_intensities[j, i, k] = hf
            else:
                self.peak_intensities[j, i, k] = hf * float(self.norm_scale_map[j, i]) if fit_normalize else hf

        if warm_start and best_rmse <= warm_start_rmse_gate:
            return np.asarray(best_params, dtype=float)
        if reset_on_fail:
            return p0_base.copy()
        return p0_current.copy()

    def _fit_rows(
        self,
        j_start,
        j_end,
        *,
        fitted_params,
        spectra_fit_cube,
        xdata,
        lower_bound,
        upper_bound,
        n_params,
        p0_base,
        p0_start,
        model_fn,
        stride,
        warm_start,
        row_reset,
        warm_start_rmse_gate,
        reset_on_fail,
        fit_normalize,
        maxfev,
        adaptive_multistart,
        fast_n_starts,
        fast_p0_strategy,
        fast_random_state,
        fallback_n_starts,
        fallback_p0_strategy,
        fallback_random_state,
        retry_on_fail,
        retry_on_high_rmse,
        retry_on_bound_hit,
        retry_rmse_gate,
        width_penalty,
        prefer_nonbound,
        score_tie_tol,
        diagnostics_mode,
        pbar=None,
    ):
        def _store_diag(j, i, payload):
            if self.fit_diagnostics_map is None:
                return

            if diagnostics_mode == "full":
                self.fit_diagnostics_map[j, i] = payload
                return

            light = {"ok": payload.get("ok", None)}
            for key in (
                "reason",
                "rmse",
                "n_starts",
                "n_fail",
                "p0_strategy",
                "adaptive_retry_used",
                "n_params_at_lower_bounds",
                "n_params_at_upper_bounds",
            ):
                if key in payload:
                    light[key] = payload[key]

            self.fit_diagnostics_map[j, i] = light

        p0_current = p0_start.copy()
        for j in range(j_start, j_end):
            for i in range(self.X):
                if pbar is not None:
                    pbar.update(1)

                # --- get already-preprocessed spectrum for this pixel ---
                y = np.asarray(spectra_fit_cube[j, i, :], dtype=float)
                x = xdata

                # --- final fit-space preparation only ---
                y_fitspace, s = self._prepare_fit_spectrum(x, y, fit_normalize=fit_normalize)

                if y_fitspace is None:
                    self.norm_scale_map[j, i] = np.nan
                    self.residual_map[j, i] = np.nan
                    _store_diag(j, i, {"ok": False, "reason": "no_positive_signal"})
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                self.norm_scale_map[j, i] = float(s)

                # --- adaptive fitting policy ---
                retry_used = False
                retry_result = None

                if adaptive_multistart:
                    quick_result = _run_mapping_curve_fit_trials(
                        model_fn=model_fn,
                        x=x,
                        y=y_fitspace,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        p0_current=p0_current,
                        maxfev=maxfev,
                        n_starts=fast_n_starts,
                        p0_strategy=fast_p0_strategy,
                        random_state=fast_random_state,
                        width_penalty=width_penalty,
                        prefer_nonbound=prefer_nonbound,
                        score_tie_tol=score_tie_tol,
                        peak_profile=self.peak_profile,
                        stride=stride,
                    )

                    fit_result = quick_result

                    need_retry = False
                    if not quick_result["ok"]:
                        need_retry = retry_on_fail
                    else:
                        if retry_on_high_rmse and quick_result["best_rmse"] > retry_rmse_gate:
                            need_retry = True
                        if retry_on_bound_hit and quick_result["best_hits"] > 0:
                            need_retry = True

                    if need_retry and fallback_n_starts > 1:
                        retry_used = True
                        retry_result = _run_mapping_curve_fit_trials(
                            model_fn=model_fn,
                            x=x,
                            y=y_fitspace,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            p0_current=p0_current,
                            maxfev=maxfev,
                            n_starts=fallback_n_starts,
                            p0_strategy=fallback_p0_strategy,
                            random_state=fallback_random_state,
                            width_penalty=width_penalty,
                            prefer_nonbound=prefer_nonbound,
                            score_tie_tol=score_tie_tol,
                            peak_profile=self.peak_profile,
                            stride=stride,
                        )

                        if retry_result["ok"]:
                            if not fit_result["ok"]:
                                fit_result = retry_result
                            else:
                                better = retry_result["best_score"] < fit_result["best_score"]
                                near_tie = abs(retry_result["best_score"] - fit_result["best_score"]) <= score_tie_tol
                                if better or (prefer_nonbound and near_tie and retry_result["best_hits"] < fit_result["best_hits"]):
                                    fit_result = retry_result
                        elif not fit_result["ok"]:
                            fit_result = retry_result

                    n_fail_total = quick_result["n_fail"] + (
                        retry_result["n_fail"] if retry_result is not None else 0
                    )
                    n_starts_total = fast_n_starts + (
                        fallback_n_starts if retry_used else 0
                    )

                else:
                    fit_result = _run_mapping_curve_fit_trials(
                        model_fn=model_fn,
                        x=x,
                        y=y_fitspace,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        p0_current=p0_current,
                        maxfev=maxfev,
                        n_starts=fallback_n_starts,
                        p0_strategy=fallback_p0_strategy,
                        random_state=fallback_random_state,
                        width_penalty=width_penalty,
                        prefer_nonbound=prefer_nonbound,
                        score_tie_tol=score_tie_tol,
                        peak_profile=self.peak_profile,
                        stride=stride,
                    )
                    n_fail_total = fit_result["n_fail"]
                    n_starts_total = fit_result["n_starts"]

                # --- handle fail / success ---
                if not fit_result["ok"]:
                    fitted_params[j, i, :] = np.nan
                    self.residual_map[j, i] = np.nan
                    _store_diag(j, i, {
                        "ok": False,
                        "n_starts": n_starts_total,
                        "n_fail": n_fail_total,
                        "p0_strategy": (
                            "adaptive"
                            if adaptive_multistart
                            else fallback_p0_strategy
                        ),
                        "adaptive_retry_used": bool(retry_used),
                    })

                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                best_params = fit_result["best_params"]
                best_rmse = fit_result["best_rmse"]
                best_p0 = fit_result["best_p0"]

                # --- handle fail / success ---
                if best_params is None:
                    fitted_params[j, i, :] = np.nan
                    self.residual_map[j, i] = np.nan
                    _store_diag(j, i, {
                        "ok": False,
                        "n_starts": n_starts_total,
                        "n_fail": n_fail_total,
                        "p0_strategy": (
                            "adaptive"
                            if adaptive_multistart
                            else fallback_p0_strategy
                        ),
                        "adaptive_retry_used": bool(retry_used),
                    })

                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                # success
                fitted_params[j, i, :] = best_params
                self.residual_map[j, i] = float(best_rmse)
                at_lo = _params_at_bounds(best_params, lower_bound, upper_bound, which="lower", rtol=1e-6)
                at_hi = _params_at_bounds(best_params, lower_bound, upper_bound, which="upper", rtol=1e-6)

                payload = {
                    "ok": True,
                    "rmse": float(best_rmse),
                    "n_starts": n_starts_total,
                    "n_fail": n_fail_total,
                    "p0_strategy": (
                        "adaptive"
                        if adaptive_multistart
                        else fallback_p0_strategy
                    ),
                    "adaptive_retry_used": bool(retry_used),
                    "n_params_at_lower_bounds": int(np.count_nonzero(at_lo)),
                    "n_params_at_upper_bounds": int(np.count_nonzero(at_hi)),
                }

                if diagnostics_mode == "full":
                    payload["best_p0"] = np.asarray(best_p0, dtype=float)
                    payload["params_at_lower_bounds_mask"] = at_lo
                    payload["params_at_upper_bounds_mask"] = at_hi

                _store_diag(j, i, payload)

                # --- derive peak centre + peak height per component ---
                # best_params ordering: [centre, width(scale), amp] repeated
                n_peaks = len(self.peak_params)
                # stride = int(self.params_per_peak)

                for k in range(n_peaks):
                    block = np.asarray(best_params[stride*k:stride*(k+1)], dtype=float)
                    centre = float(block[0])
                    self.peak_positions[j, i, k] = centre

                    # peak height in FIT SPACE
                    if self.peak_profile == "lorentzian":
                        hwhm = float(block[1])
                        amp_area = float(block[2])
                        if (not np.isfinite(hwhm)) or hwhm <= 0:
                            height_fitspace = np.nan
                        else:
                            height_fitspace = amp_area / (np.pi * hwhm)
                    else:
                        # pVoigt: height is not your "amp" parameter; compute from the profile
                        # single_peak returns the (normalised-space) y(x) for that one peak
                        y_one = single_peak(x, block, profile="pvoigt")
                        height_fitspace = float(np.nanmax(y_one))

                    # Convert to display convention:
                    # normalize=True  -> store fit-space height
                    # normalize=False -> store raw height using per-pixel scale when fit_normalize=True
                    if self.normalize:
                        self.peak_intensities[j, i, k] = height_fitspace
                    else:
                        if fit_normalize:
                            self.peak_intensities[j, i, k] = height_fitspace * float(self.norm_scale_map[j, i])
                        else:
                            self.peak_intensities[j, i, k] = height_fitspace


                # warm-start update (respect your RMSE gate)
                if warm_start and best_rmse <= warm_start_rmse_gate:
                    p0_current = np.asarray(best_params, dtype=float)
                else:
                    if reset_on_fail:
                        p0_current = p0_base.copy()


            # Row reset
            if row_reset:
                p0_current = p0_base.copy()

    def plot_spectrum_fit(self, x, y):
        """Plot raw data and fitting results for a single map point.

        Args:
            x (int): X coordinate (0-indexed)
            y (int): Y coordinate (0-indexed)

        Display logic:
            - Fitting is always done in normalised space.
            - normalize=True  -> show normalised, background-removed spectrum + fit
            - normalize=False -> show raw spectrum + fit (+ background overlay if enabled)
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # --- Extract spectrum
        x_full = np.asarray(self.xdata, dtype=float)
        y_full = np.asarray(self.spectra[y, x, :], dtype=float)

        # --- Mask by energy range (eV)
        mask = DataImporter.mask_by_xrange(self.xdata, self.data_range)
        xdata = x_full[mask]
        raw_intensity = y_full[mask]

        # --- Preprocessing consistent with fitting (except final normalisation)
        proc = raw_intensity.copy()

        if self.smoothing:
            proc = savgol_filter(proc, self.smooth_window, self.smooth_poly)

        if self.background_remove:
            bg_removed = self.remove_background(xdata, proc)
            background = proc - bg_removed
        else:
            bg_removed = proc
            background = None

        # --- Scale used for fitting normalisation
        scale = None
        if hasattr(self, "norm_scale_map"):
            scale = self.norm_scale_map[y, x]

        if scale is None or not np.isfinite(scale) or scale <= 0:
            scale = np.max(bg_removed)

        if scale <= 0:
            raise ValueError(f"No positive signal at (X={x}, Y={y}); cannot scale fitted curve.")

        # --- Load fitted parameters (normalised space)
        params = np.asarray(self.fitted_params[y, x, :], dtype=float)
        if np.any(np.isnan(params)):
            raise ValueError(f"Fit parameters are NaN at (X={x}, Y={y}). Fit may have failed.")

        model_fn = self._model_dispatch()
        fitted_norm = model_fn(xdata, *params)
        fitted_raw = fitted_norm * scale

        # --- Plot
        plt.figure(figsize=(10, 6))

        if self.normalize:
            # Normalised display (background-removed)
            spectrum_norm = bg_removed / scale
            plt.plot(xdata, spectrum_norm, "k-", label="Background-removed (normalised)")
            plt.plot(xdata, fitted_norm, "g--", linewidth=2, label="Fitted Curve")
            plt.ylabel("Normalised Intensity (a.u.)")

        else:
            # Raw display
            plt.plot(xdata, raw_intensity, "k-", label="Raw Spectrum")

            if self.background_remove:
                # Show background components
                plt.plot(xdata, background, "r--", label="Estimated Background")
                plt.plot(xdata, bg_removed, "b-", alpha=0.8, label="Background Removed (smoothed)")

                # Peak-only fit
                plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve (peak only)")

                # Best overlay vs raw spectrum
                fitted_plus_bg = fitted_raw + background
                plt.plot(xdata, fitted_plus_bg, "-", linewidth=2, label="Fit + Estimated Background")

            else:
                # No background removal → show fit only once
                plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve")

            plt.ylabel("Intensity (a.u.)")

        plt.xlabel("Energy (eV)")
        plt.title(f"Spectrum Fit at (X={x}, Y={y})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residual_distribution(
        self,
        filter_threshold=None,
        robust=True,
        p_low=5,
        p_high=95,
        hist_bins=50,
        cmap="inferno"
    ):
        """
        Visualise spatial distribution of fitting residuals and their histogram.

        Behaviour:
        - If filter_threshold is None:
            show full residual heatmap (optionally robust-scaled) + histogram.
        - If filter_threshold is set:
            ONLY show pixels with residual >= filter_threshold (others masked out) + histogram.
        """
        # import numpy as np
        # import matplotlib.pyplot as plt
        residuals = np.asarray(self.residual_map, dtype=float)
        valid = ~np.isnan(residuals)
        residuals_flat = residuals[valid]

        if residuals_flat.size == 0:
            raise ValueError("Residual map contains no valid values to plot.")

        # Determine colour scaling
        if robust:
            vmin = np.percentile(residuals_flat, p_low)
            vmax = np.percentile(residuals_flat, p_high)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None

        # If thresholding, focus colour scale on the thresholded region for contrast
        if filter_threshold is not None:
            above = residuals_flat[residuals_flat >= filter_threshold]
            if above.size == 0:
                raise ValueError(
                    f"No pixels found with residual >= {filter_threshold:g}. "
                    "Try lowering filter_threshold."
                )
            # Make the colour scale meaningful for the highlighted pixels
            vmin = filter_threshold
            vmax_thr = np.percentile(above, 99) if above.size > 5 else np.max(above)
            vmax = vmax_thr if (np.isfinite(vmax_thr) and vmax_thr > vmin) else np.max(above)

        # Layout
        fig, (ax_map, ax_hist) = plt.subplots(
            1, 2, figsize=(12, 5),
            gridspec_kw={"width_ratios": [3, 1]}
        )

        # ---- Map panel ----
        if filter_threshold is None:
            # Normal view (like your 2nd image)
            data_masked = np.ma.masked_invalid(residuals)
            title = "Residual Distribution (higher = worse fit)"
        else:
            # Threshold view: only show pixels >= threshold
            keep = (residuals >= filter_threshold) & valid
            data_masked = np.ma.masked_where(~keep, residuals)
            title = f"Residual Distribution (≥ {filter_threshold:g})"

        im = ax_map.imshow(
            data_masked,
            cmap=cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax
        )
        cbar = fig.colorbar(im, ax=ax_map)
        cbar.set_label("Residual Error (RMSE)")

        ax_map.set_title(title)
        ax_map.set_xlabel("X Position")
        ax_map.set_ylabel("Y Position")

        # ---- Histogram panel ----
        ax_hist.hist(
            residuals_flat,
            bins=hist_bins,
            orientation="horizontal",
            color="darkred",
            edgecolor="black"
        )
        ax_hist.set_xlabel("Count")
        ax_hist.set_ylabel("Residual RMSE")
        ax_hist.set_title("Residual Histogram")

        # Add a threshold line to the histogram for clarity
        if filter_threshold is not None:
            ax_hist.axhline(filter_threshold, linestyle="--", linewidth=1)
            # If thresholding, it can help to zoom histogram y-range to the upper tail
            upper = residuals_flat[residuals_flat >= filter_threshold]
            y_lo = max(filter_threshold * 0.98, np.min(upper))
            y_hi = np.max(upper)
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
                ax_hist.set_ylim(y_lo, y_hi)
        else:
            # Match histogram y-range to map scaling if robust scaling is enabled
            if vmin is not None and vmax is not None:
                ax_hist.set_ylim(vmin, vmax)

        plt.tight_layout()
        plt.show()   

    def plot_heatmap(self, data_type='exciton_position', cmap='viridis',
                    filter_range=None, specific_xdata=None,
                    x_range=None, y_range=None):
        """Visualize 2D map of spectral features."""
        import numpy as np
        import matplotlib.pyplot as plt

        if data_type == 'specific_intensity':
            if specific_xdata is None:
                raise ValueError("For 'specific_intensity' data type, 'specific_xdata' must be provided (in eV).")

            data = np.full((self.Y, self.X), np.nan, dtype=float)

            model_fn = self._model_dispatch()
            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]
                    if np.any(np.isnan(params)):
                        continue  # fit failed

                    y_norm = model_fn(np.asarray([specific_xdata], dtype=float), *params)[0]

                    if self.normalize:
                        # display normalised model intensity (dimensionless)
                        data[j, i] = y_norm
                    else:
                        # display raw model intensity using stored per-pixel scale
                        if (not hasattr(self, "norm_scale_map")) or np.isnan(self.norm_scale_map[j, i]):
                            continue
                        data[j, i] = y_norm * self.norm_scale_map[j, i]

            label = (f'Normalised intensity at {specific_xdata} eV (a.u.)'
                    if self.normalize else
                    f'Intensity at {specific_xdata} eV (a.u.)')

        elif data_type == 'exciton_position':
            data = self.peak_positions[:, :, 0]
            label = 'Exciton Position (eV)'

        elif data_type == 'trion_position':
            if self.peak_positions.shape[2] > 1:
                data = self.peak_positions[:, :, 1]
                label = 'Trion Position (eV)'
            else:
                raise ValueError("Trion data not available.")

        elif data_type == 'exciton_intensity':
            data = self.peak_intensities[:, :, 0]
            label = 'Exciton Intensity (a.u.)'  # already scaled in fit_spectra when normalize=False

        elif data_type == 'trion_intensity':
            if self.peak_intensities.shape[2] > 1:
                data = self.peak_intensities[:, :, 1]
                label = 'Trion Intensity (a.u.)'
            else:
                raise ValueError("Trion data not available.")

        else:
            raise ValueError("Invalid data_type. Choose from "
                            "'exciton_position', 'trion_position', "
                            "'exciton_intensity', 'trion_intensity', 'specific_intensity'.")

        # Apply optional range filter (your current behaviour: clip outliers to lower bound)
        if filter_range is not None:
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])

        # Apply optional cropping
        if x_range is not None and y_range is not None:
            x_start, x_end = x_range
            y_start, y_end = y_range
            data = data[y_start:y_end + 1, x_start:x_end + 1]
            Xp = (x_end - x_start + 1)
            Yp = (y_end - y_start + 1)
        else:
            Xp, Yp = self.X, self.Y

        x_length = Xp * self.step_size
        y_length = Yp * self.step_size

        cm = plt.get_cmap(cmap).copy()
        cm.set_bad('gray')

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            data,
            cmap=cm,
            vmin=filter_range[0] if filter_range else None,
            vmax=filter_range[1] if filter_range else None,
            extent=[0, x_length, y_length, 0]
        )
        plt.colorbar(im, label=label)
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.title(f"Heatmap of {label}")
        plt.tight_layout()
        plt.show()

    ### Updated in v0.5.1
    def export_fit_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        scaled: bool = True,
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
        long: bool = False,
    ) -> str:
        """
        Export fit results for every pixel.

        Default (long=False): wide format — x, y, then per-peak parameters
        on the same row, followed by QA columns rmse, ok, n_starts,
        n_params_at_bounds.

        long=True: one row per (pixel, peak) — x, y, peak, then per-peak
        parameters, followed by the same QA columns.
        """
        if not hasattr(self, "fitted_params") or self.fitted_params is None:
            raise ValueError("No fitted_params found. Run fit_spectra() first.")

        coord_mode = normalise_coord_mode(coord_mode)

        peak_labels = list(self.peak_params)

        per_peak_fields = ["centre", "fwhm", "peak_height", "peak_height_norm", "amp", "scale"]
        if self.peak_profile == "pvoigt":
            per_peak_fields.append("eta")

        _qa_fields = ["rmse", "ok", "n_starts", "n_params_at_bounds"]

        if long:
            fields = ["x", "y", "peak"] + per_peak_fields + _qa_fields
        else:
            fields = ["x", "y"]
            for p in peak_labels:
                for f in per_peak_fields:
                    fields.append(f"{p}_{f}")
            fields.extend(_qa_fields)

        rows = []

        if long:
            for x, y, j, i in self._iter_coords(coord_mode=coord_mode):
                params = np.asarray(self.fitted_params[j, i, :], dtype=float)
                qa = self._qa_columns_for_pixel(j, i, len(peak_labels))
                failed = np.any(np.isnan(params))

                intensity_scale = 1.0
                if (not failed) and scaled and hasattr(self, "norm_scale_map") and np.isfinite(self.norm_scale_map[j, i]):
                    intensity_scale = float(self.norm_scale_map[j, i])

                if not failed:
                    per_peak = self._params_to_export_dict(
                        self.xdata, peak_labels, params, intensity_scale=intensity_scale,
                    )

                for name in peak_labels:
                    r = {"x": x, "y": y, "peak": name}
                    if failed:
                        for f in per_peak_fields:
                            r[f] = float("nan")
                    else:
                        d = per_peak[name]
                        r["centre"] = d["centre"]
                        r["fwhm"] = d["fwhm"]
                        r["peak_height"] = d["peak_height"]
                        r["peak_height_norm"] = d["peak_height_norm"]
                        r["amp"] = d["amp"]
                        r["scale"] = d["scale"]
                        if self.peak_profile == "pvoigt":
                            r["eta"] = d["eta"]
                    r.update(qa)
                    rows.append(r)

        else:
            for x, y, j, i in self._iter_coords(coord_mode=coord_mode):
                params = np.asarray(self.fitted_params[j, i, :], dtype=float)
                qa = self._qa_columns_for_pixel(j, i, len(peak_labels))
                if np.any(np.isnan(params)):
                    r = {"x": x, "y": y}
                    r.update(qa)
                    rows.append(r)
                    continue

                intensity_scale = 1.0
                if scaled and hasattr(self, "norm_scale_map") and np.isfinite(self.norm_scale_map[j, i]):
                    intensity_scale = float(self.norm_scale_map[j, i])

                per_peak = self._params_to_export_dict(
                    self.xdata,
                    peak_labels,
                    params,
                    intensity_scale=intensity_scale,
                )

                r = {"x": x, "y": y}
                for name in peak_labels:
                    d = per_peak[name]
                    r[f"{name}_centre"] = d["centre"]
                    r[f"{name}_fwhm"] = d["fwhm"]
                    r[f"{name}_peak_height"] = d["peak_height"]
                    r[f"{name}_peak_height_norm"] = d["peak_height_norm"]
                    r[f"{name}_amp"] = d["amp"]
                    r[f"{name}_scale"] = d["scale"]
                    if self.peak_profile == "pvoigt":
                        r[f"{name}_eta"] = d["eta"]
                r.update(qa)
                rows.append(r)

        export_format = "long" if long else "wide"
        meta = self._build_mapping_fit_export_meta(
            coord_mode=coord_mode,
            scaled=scaled,
            peak_labels=peak_labels,
        )
        meta["export_format"] = export_format

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )


#########################################################################################################################
############################################## PL Integration Mapping ###################################################
#########################################################################################################################

class PL_Integration:
    """Photoluminescence mapping analysis through spectral integration.
    
    Attributes:
        filename (str): Path to input file
        integration_range (tuple): Spectral integration range (min, max) in eV
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        background_remove (bool): Enable background subtraction
        X (int): Map width in pixels
        Y (int): Map height in pixels
        energy (ndarray): Spectral axis in eV
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        integration_area (ndarray): Integrated intensities [Y, X]
    """
    def __init__(self, filename, integration_range, step_size=0.3, poly_degree=3,
             background_remove=True, baseline_method="poly"):
        """Initialize PL integration analyzer.
        
        Args:
            filename: Path to .wdf file
            integration_range: Spectral range (min, max) in eV
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            background_remove: Enable background subtraction
            baseline_method: 'poly' or 'gaussian' background
        """
        self.filename = filename
        self.integration_range = integration_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.background_remove = background_remove

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Photoluminescence"
        self.x_quantity = "Photon energy"
        self.x_unit = "eV"
        self.step_unit = "um"  # keep consistent with your plotting labels "μm"


        # --- Updated in v0.3.8: baseline_method parsing with BaselineAPI ---
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
        )
        self._baseline_method, self._baseline_kwargs = baseline_spec_to_runtime(
            self.baseline_method
        )

        # integration_range is known here; trim at load-time.
        loader = MappingFileLoader(filename, x_range=self.integration_range, axis="energy")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.energy = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

    ### NEW METHOD in v0.2.7 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        integration_range,
        step_size=0.3,
        poly_degree=3,
        background_remove=True,
        baseline_method="poly",
        clip_nonnegative=False,
    ):
        """
        Construct PL_Integration from in-memory arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Energy axis (eV) with shape [N]
        X, Y : int
            Map dimensions
        integration_range : tuple(float, float)
            (min, max) eV; applied immediately (trim at construction, consistent with __init__)
        """
        obj = cls.__new__(cls)

        # Mirror __init__ fields
        obj.filename = None
        obj.integration_range = integration_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.background_remove = background_remove
        obj.spectrum_type = "Photoluminescence"
        obj.x_quantity = "Photon energy"
        obj.x_unit = "eV"
        obj.step_unit = "um"

        # Baseline configuration (same pattern as __init__)
        obj.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
        )
        obj._baseline_method, obj._baseline_kwargs = baseline_spec_to_runtime(
            obj.baseline_method
        )

        obj.X = int(X)
        obj.Y = int(Y)

        energy = np.asarray(xdata, dtype=float).ravel()
        cube = np.asarray(spectra, dtype=float)

        # Validate shapes
        if cube.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if cube.shape[0] != obj.Y or cube.shape[1] != obj.X:
            raise ValueError(f"spectra shape {cube.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if cube.shape[2] != energy.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # Trim at load-time (same semantics as filename-based __init__)
        emin, emax = integration_range
        mask = (energy >= emin) & (energy <= emax)
        if not np.any(mask):
            raise ValueError(
                f"integration_range {integration_range} does not overlap provided energy axis "
                f"[{float(np.min(energy)):.3g}, {float(np.max(energy)):.3g}]."
            )

        obj.energy = energy[mask]
        obj.spectra = cube[:, :, mask]
        obj._x_trimmed_on_load = True

        if clip_nonnegative:
            obj.spectra = np.clip(obj.spectra, a_min=0.0, a_max=None)

        # No optical image when constructed from arrays
        obj.image_viewer = None

        # Output
        obj.integration_area = np.zeros((obj.Y, obj.X), dtype=float)

        return obj

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    ### Updated in v0.2.5 ##
    def remove_background(self, energy, intensity):
        """Background removal via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=energy,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected


    def calculate_integration(self):
        """Calculate integrated area under spectra across all map points.

        Uses Simpson's rule for integration.
        Stores results in integration_area array.
        """
        energy = np.asarray(self.energy, dtype=float).ravel()

        mask = DataImporter.mask_by_xrange(energy, self.integration_range)
        energy_subset = energy[mask]

        for j in range(self.Y):
            for i in range(self.X):
                spectra = np.asarray(self.spectra[j, i, :], dtype=float).ravel()
                spectra_subset = spectra[mask]

                if self.background_remove:
                    spectra_subset = self.remove_background(energy_subset, spectra_subset)

                self.integration_area[j, i] = np.abs(simpson(spectra_subset, energy_subset))


    def plot_integration_heatmap(self, cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """Visualize 2D map of integrated intensities.
        
        Args:
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
        """
        # Filter data range
        data = self.integration_area
        if filter_range is not None:
            # Replace outliers with filter_range[0] instead of NaN
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])

        # If x_range and y_range are specified, only plot data within the specified region
        if x_range is not None and y_range is not None:
            x_start, x_end = x_range
            y_start, y_end = y_range
            data = data[y_start:y_end+1, x_start:x_end+1]
            # Calculate actual length range
            x_length = (x_end - x_start + 1) * self.step_size
            y_length = (y_end - y_start + 1) * self.step_size
        else:
            # Calculate actual length range
            x_length = self.X * self.step_size
            y_length = self.Y * self.step_size

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            data,
            cmap=cmap,
            vmin=filter_range[0] if filter_range else None,  # Anchor color scale
            vmax=filter_range[1] if filter_range else None,  # to filter range
            extent=[0, x_length, y_length, 0])
        cbar = plt.colorbar(im, label='Integration Area (a.u.)')
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.title(f"Integration Area Heatmap ({self.integration_range[0]} - {self.integration_range[1]} eV)")
        plt.show()

    def plot_spectrum(self, x, y):
        """Plot raw and processed spectra for single map point.
        
        Args:
            x (int): X coordinate (0-indexed)
            y (int): Y coordinate (0-indexed)
            
        Shows:
            - Raw spectrum (blue)
            - Background-removed spectrum (red, if enabled)
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # Get the original spectrum data
        energy = self.energy[:]
        spectra = self.spectra[y][x][:]

        # Get data within the integration range
        mask = DataImporter.mask_by_xrange(energy, self.integration_range)
        energy_subset = energy[mask]
        spectra_subset = spectra[mask]


        # If background removal is enabled, remove the background signal
        spectra_raw = spectra_subset.copy()

        if self.background_remove:
            spectra_bg_removed = self.remove_background(energy_subset, spectra_subset)
        else:
            spectra_bg_removed = spectra_subset

        plt.figure(figsize=(10, 6))
        plt.plot(energy_subset, spectra_raw, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(energy_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.show()

    ### Updated v0.3.8: export_integration_map with metadata and flexible formatting ###
    def export_integration_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
        column_name: str = "integration_area",
    ) -> str:
        """
        Export integration_area in wide format:
        x, y, integration_area
        """
        if not hasattr(self, "integration_area") or self.integration_area is None:
            raise ValueError("No integration_area found. Run calculate_integration() first.")

        coord_mode = normalise_coord_mode(coord_mode)

        step = float(self.step_size)
        rows = []
        for j in range(self.Y):
            for i in range(self.X):
                x, y = (i * step, j * step) if coord_mode == "real" else (i, j)
                rows.append({"x": x, "y": y, column_name: float(self.integration_area[j, i])})

        fields = ["x", "y", column_name]

        meta = build_export_meta(
            export_kind="mapping_table",
            map_kind="integration",
            spectrum_type=getattr(self, "spectrum_type", None),
            x_quantity=getattr(self, "x_quantity", None),
            x_unit=getattr(self, "x_unit", None),
            coord_mode=coord_mode,
            step_size=getattr(self, "step_size", None),
            step_unit=getattr(self, "step_unit", "um"),
            integration_range=getattr(self, "integration_range", None),
            baseline_spec=getattr(self, "baseline_method", None),
            background_remove=getattr(self, "background_remove", None),
        )

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )
    