from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import simpson

try:
    from ..baselineAPI import BaselineAPI
    from ..dataImporter import DataImporter
    from ..exporter import build_export_meta, write_table
    from ..peak_models import single_peak, sum_peaks
    from ..schema import normalise_baseline_spec, normalise_coord_mode, normalise_peak_profile
    from ._diagnostics import fit_summary as _fit_summary
    from ._fit_utils import (
        _mapping_generate_p0_trials,
        _params_at_bounds,
        _width_param_to_fwhm,
        seed_p0_from_coord,
    )
    from ._io import MappingFileLoader
    from ._image import MappingImage
    from ._preprocess import _MappingPreprocessMixin
except Exception:  # pragma: no cover
    from ramanpl.baselineAPI import BaselineAPI
    from ramanpl.dataImporter import DataImporter
    from ramanpl.exporter import build_export_meta, write_table
    from ramanpl.peak_models import single_peak, sum_peaks
    from ramanpl.schema import normalise_baseline_spec, normalise_coord_mode, normalise_peak_profile
    from ramanpl.mapping._diagnostics import fit_summary as _fit_summary
    from ramanpl.mapping._fit_utils import (
        _mapping_generate_p0_trials,
        _params_at_bounds,
        _width_param_to_fwhm,
        seed_p0_from_coord,
    )
    from ramanpl.mapping._io import MappingFileLoader
    from ramanpl.mapping._image import MappingImage
    from ramanpl.mapping._preprocess import _MappingPreprocessMixin


########################################################################################################################
#################################################### Raman Mapping #####################################################
########################################################################################################################

class RamanMapping(_MappingPreprocessMixin):
    """Raman mapping analysis through Lorentzian peak fitting.
    
    Attributes:
        filename (str): Path to .wdf file
        custom_peaks (dict): Peak parameters for fitting
        data_range (tuple): Spectral analysis range (min, max) in cm⁻¹
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        normalize (bool): Enable spectrum normalization
        background_remove (bool): Enable background subtraction
        smoothing (bool): Enable spectral smoothing
        baseline_method (str): Background method ('poly' or 'gaussian')
        smooth_window (int): Savitzky-Golay window size
        smooth_poly (int): Savitzky-Golay polynomial order
        gaussian_sigma (int): Gaussian filter width
        peak_params (list): Peak names from custom_peaks
        X (int): Map width in pixels
        Y (int): Map height in pixels
        wavenumber (ndarray): Spectral axis in cm⁻¹
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        peak_positions (ndarray): Fitted peak centers [Y, X, peaks]
        peak_intensities (ndarray): Fitted peak amplitudes [Y, X, peaks]
        fitted_params (ndarray): Full fitting parameters [Y, X, 3*peaks]
        residual_map (ndarray): Fitting residuals [Y, X]
        Peaks_distance (ndarray): A1g-E2g peak distances [Y, X]
        ratio_A1g_E2g (ndarray): A1g/E2g intensity ratios [Y, X]
        ratio_E2g_A1g (ndarray): E2g/A1g intensity ratios [Y, X]
    """
    def __init__(
        self,
        filename,
        custom_peaks,
        data_range,
        step_size=0.3,
        poly_degree=3,
        normalize=False,
        background_remove=True,
        smoothing=True,
        baseline_method="poly",
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
        peak_profile: str = "lorentzian",
        preprocessing=None,
    ):
        """Initialize Raman mapping analyzer.
        
        Args:
            filename: Path to .wdf file
            custom_peaks: Peak definitions with bounds {name: (min_params, max_params)}
            data_range: Spectral range (min, max) in cm⁻¹
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            normalize: Normalize spectra to [0,1] range
            background_remove: Enable background subtraction
            smoothing: Enable spectral smoothing
            baseline_method: 'poly' or 'gaussian' background
            smooth_window: Savitzky-Golay window size
            smooth_poly: Savitzky-Golay polynomial order
            gaussian_sigma: Gaussian filter width
            peak_profile: str = "lorentzian" or "pvoigt"
            preprocessing: Optional list of preprocessing steps to apply before fitting (e.g., ['normalize', 'smooth'])
        """
        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize          # DISPLAY flag only (fit-space can still be normalised)
        self.background_remove = background_remove
        self.smoothing = smoothing
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma,
        )
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.peak_params = list(custom_peaks.keys())

        # --- identity metadata for exports ---
        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"
        self.step_unit = "um"

        if str(self.baseline_method.get("method", "")).lower() == "poly":
            self.poly_order = int(self.baseline_method.get("poly_order", poly_degree))
        else:
            self.poly_order = None

        # Baseline config
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            self.baseline_method
        )

        # ---- model choice ----
        self.peak_profile = normalise_peak_profile(peak_profile)
        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        # Shared preprocessing pipeline (legacy-compatible if None)
        self._initialise_preprocessing(preprocessing=preprocessing)

        # ---- load mapping data (Raman always wavenumber axis) ----
        loader = MappingFileLoader(filename, x_range=self.data_range, axis="wavenumber")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None

        # ---- allocate output arrays ----
        num_peaks = len(self.custom_peaks)
        self.peak_positions = np.full((self.Y, self.X, num_peaks), np.nan, dtype=float)
        self.peak_intensities = np.full((self.Y, self.X, num_peaks), np.nan, dtype=float)
        self.fitted_params = np.full((self.Y, self.X, num_peaks * self.params_per_peak), np.nan, dtype=float)

        self.residual_map = np.full((self.Y, self.X), np.nan, dtype=float)
        self.norm_scale_map = np.full((self.Y, self.X), np.nan, dtype=float)

        self.Peaks_distance = np.full((self.Y, self.X), np.nan, dtype=float)
        self.ratio_A1g_E2g = np.full((self.Y, self.X), np.nan, dtype=float)
        self.ratio_E2g_A1g = np.full((self.Y, self.X), np.nan, dtype=float)

    ### Updated in v0.3.5 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        custom_peaks,
        data_range,
        step_size=0.3,
        poly_degree=3,
        normalize=False,
        background_remove=True,
        smoothing=True,
        baseline_method='poly',
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
        peak_profile: str = "lorentzian",
        preprocessing=None,
    ):
        """
        Create a RamanMapping instance from in-memory mapping arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Spectral axis with shape [N] (wavenumber in cm^-1)
        X, Y : int
            Map dimensions
        data_range : tuple
            (min, max) in cm^-1 (kept as in the filename-based API)
        """
        obj = cls.__new__(cls)

        obj.filename = None
        obj.custom_peaks = custom_peaks
        obj.data_range = data_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.normalize = normalize
        obj.background_remove = background_remove
        obj.smoothing = smoothing
        obj.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma,
        )
        obj.smooth_window = smooth_window
        obj.smooth_poly = smooth_poly
        obj.gaussian_sigma = gaussian_sigma
        obj.peak_params = list(custom_peaks.keys())

        obj.spectrum_type = "Raman"
        obj.x_quantity = "Raman shift"
        obj.x_unit = "cm^-1"
        obj.step_unit = "um"

        if str(obj.baseline_method.get("method", "")).lower() == "poly":
            obj.poly_order = int(obj.baseline_method.get("poly_order", poly_degree))
        else:
            obj.poly_order = None

        # Baseline config (same as __init__)
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            obj.baseline_method
        )

        obj.X = int(X)
        obj.Y = int(Y)
        obj.wavenumber = np.asarray(xdata, dtype=float).ravel()
        obj.spectra = np.asarray(spectra, dtype=float)

        # New in v0.3.3
        obj.peak_profile = normalise_peak_profile(peak_profile)
        obj.params_per_peak = 3 if obj.peak_profile == "lorentzian" else 4
        obj._initialise_preprocessing(preprocessing=preprocessing)

        # Validate shapes
        if obj.spectra.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if obj.spectra.shape[0] != obj.Y or obj.spectra.shape[1] != obj.X:
            raise ValueError(f"spectra shape {obj.spectra.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if obj.spectra.shape[2] != obj.wavenumber.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        obj.image_viewer = None

        # Not trimmed at load time when built from arrays
        obj._x_trimmed_on_load = False

        # Allocate arrays (same as __init__)
        num_peaks = len(obj.custom_peaks)
        obj.peak_positions = np.full((obj.Y, obj.X, num_peaks), np.nan)
        obj.peak_intensities = np.full((obj.Y, obj.X, num_peaks), np.nan)
        obj.fitted_params = np.full((obj.Y, obj.X, num_peaks * obj.params_per_peak), np.nan)

        obj.residual_map = np.full((obj.Y, obj.X), np.nan)
        obj.norm_scale_map = np.full((obj.Y, obj.X), np.nan)

        obj.Peaks_distance = np.full((obj.Y, obj.X), np.nan, dtype=float)
        obj.ratio_A1g_E2g = np.full((obj.Y, obj.X), np.nan, dtype=float)
        obj.ratio_E2g_A1g = np.full((obj.Y, obj.X), np.nan, dtype=float)

        return obj

    ### New in v0.2.7 ###
    def get_reference_spectrum(self, *, x: int, y: int, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Return a reference spectrum from this already-loaded mapping object.

        Returns
        -------
        (y_ref, wavenumber)
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

        return np.asarray(y_ref, dtype=float).ravel(), np.asarray(self.wavenumber, dtype=float).ravel()
   
    @staticmethod
    def custom_peaks_from_ramanfit(raman_fit):
        """
        Build Mapping-compatible custom_peaks dict from a RamanFit instance.

        Supports both lorentzian (3 params/peak) and pvoigt (4 params/peak).
        """
        import numpy as np

        labels = list(raman_fit.peak_labels)
        lb = np.asarray(raman_fit.lower_bound, dtype=float)
        ub = np.asarray(raman_fit.upper_bound, dtype=float)
        stride = int(getattr(raman_fit, "params_per_peak", 3))

        if lb.size != ub.size or lb.size != stride * len(labels):
            raise ValueError("RamanFit bounds length mismatch with peak_labels.")

        out = {}
        for k, name in enumerate(labels):
            s = stride * k
            out[name] = (lb[s:s+stride].tolist(), ub[s:s+stride].tolist())
        return out
    
    def _find_peak_index(self, target):
        """
        Resolve a peak index robustly.
        Matches exact name first, then case-insensitive, then substring containment.
        Returns int index or None.
        """
        if target is None:
            return None

        names = list(self.peak_params)
        # exact
        if target in names:
            return names.index(target)

        # case-insensitive exact
        t = target.lower()
        for i, n in enumerate(names):
            if n.lower() == t:
                return i

        # substring match (handles e.g. 'E2g' vs 'E12g(Γ)')
        for i, n in enumerate(names):
            nl = n.lower()
            if t in nl or nl in t:
                return i

        return None

    def show_optical_image(self):
        """Display optical image with mapping area overlay."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def fit_summary(self, **kwargs):
        """Backward-compatible instance wrapper for mapping fit diagnostics."""
        return _fit_summary(self, **kwargs)

    def lorentzian_raman(self, x, *params):
        """Calculate multi-Lorentzian curve for given parameters."""
        return sum_peaks(np.asarray(x), params, profile="lorentzian", stride=3)

    ### UPDATED METHOD IN v0.2.5 ###
    def remove_background(self, wavenumber, intensity):
        """Remove spectral background via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=wavenumber,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected

    def _model_dispatch(self):
        if self.peak_profile == "lorentzian":
            return self.lorentzian_raman
        return self.pvoigt  # you must implement pvoigt analogous to lorentzian using peak_models.sum_peaks

    def pvoigt(self, x, *params):
        return sum_peaks(np.asarray(x), params, profile="pvoigt", stride=4)

    ### UPDATED METHOD IN v0.3.5 ###
    def fit_spectra(
        self,
        initial_p0=None,
        warm_start=False,
        seed_coord=None,
        seed_roi=None,
        reset_on_fail=True,
        row_reset=True,
        warm_start_rmse_gate=0.06,
        maxfev=6400,
        fit_spectrum_kwargs=None,
        fit_normalize=True,
    ):


        """
        Fit all map spectra using self.custom_peaks as bounds.

        Behaviour notes:
        - fitting is ALWAYS performed in peak-normalised space
        - self.normalize affects DISPLAY only (intensity maps)
        - supports initial_p0 as vector or dict package {"p0":..., "peak_order":...}
        - warm-start propagation is gated (RMSE + plausibility) to reduce scanline ledges
        - optional row_reset prevents row-to-row propagation artefacts

        Returns
        -------
        params_map : ndarray
            Fitted parameter cube with shape [Y, X, n_params].
            In notebooks, assign the return value or use `_ = fit_spectra(...)`
            to avoid auto-display.

        """
        if fit_spectrum_kwargs is None:
            fit_spectrum_kwargs = {}

        if not hasattr(self, "custom_peaks") or not isinstance(self.custom_peaks, dict) or len(self.custom_peaks) == 0:
            raise ValueError("custom_peaks is not set or empty. Provide custom_peaks when initialising RamanMapping.")

        # ---------- helpers ----------
        def _params_plausible(params, lb, ub, n_peaks, stride, profile, tol=1e-10):
            """
            Reject fits likely stuck at bounds or non-physical.
            stride = params_per_peak (3 lorentzian, 4 pvoigt)
            """
            params = np.asarray(params, dtype=float)
            lb = np.asarray(lb, dtype=float)
            ub = np.asarray(ub, dtype=float)

            for k in range(n_peaks):
                base = stride * k
                c = params[base + 0]
                w = params[base + 1]
                a = params[base + 2]

                # width must be positive
                if (not np.isfinite(w)) or w <= 1e-8:
                    return False

                # centre/width at bounds often indicates constrained "fallback"
                if abs(c - lb[base + 0]) < tol or abs(c - ub[base + 0]) < tol:
                    return False
                if abs(w - lb[base + 1]) < tol or abs(w - ub[base + 1]) < tol:
                    return False

                # amplitude at bounds is suspicious in mapping
                if abs(a - lb[base + 2]) < tol or abs(a - ub[base + 2]) < tol:
                    return False

                if profile == "pvoigt":
                    eta = params[base + 3]
                    # eta should be in [0,1] and not stuck at bounds
                    if (not np.isfinite(eta)) or eta < -1e-6 or eta > 1 + 1e-6:
                        return False
                    if abs(eta - lb[base + 3]) < tol or abs(eta - ub[base + 3]) < tol:
                        return False

            return True

        # ---------- shared preprocessing path ----------
        xdata, spectra_fit_cube = self._get_processed_mapping_cube()


        # ---------- bounds ----------
        lower_bound, upper_bound = [], []
        for params_range in self.custom_peaks.values():
            lower_bound.extend(params_range[0])
            upper_bound.extend(params_range[1])

        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        n_params = lower_bound.size
        n_peaks = len(self.peak_params)

        idx_a1g = self._find_peak_index("A1g")
        idx_e2g = self._find_peak_index("E2g")
        if idx_e2g is None:
            idx_e2g = self._find_peak_index("E12g")

        # baseline p0
        p0_base = (lower_bound + upper_bound) / 2.0

        # --- optional: seed from a coordinate inside this mapping object ---
        if seed_coord is not None:
            if initial_p0 is not None:
                raise ValueError("Provide either initial_p0 or seed_coord, not both.")

            initial_p0 = seed_p0_from_coord(self, seed_coord, seed_roi, maxfev=maxfev)
            warm_start = True

        # ---------- optional: seed from RamanFit export ----------
        if initial_p0 is not None:
            if isinstance(initial_p0, dict):
                peak_order_pkg = initial_p0.get("peak_order", None)
                p0_vec = initial_p0.get("p0", None)

                if p0_vec is None:
                    raise ValueError("initial_p0 dict must contain key 'p0' with a numeric vector.")

                if peak_order_pkg is not None:
                    if [p.lower() for p in peak_order_pkg] != [p.lower() for p in self.peak_params]:
                        raise ValueError(
                            "peak_order mismatch between RamanFit and RamanMapping.\n"
                            f"RamanFit: {list(peak_order_pkg)}\n"
                            f"RamanMapping: {list(self.peak_params)}\n"
                            "Ensure both use the same peak ordering."
                        )

                initial_p0 = p0_vec

            initial_p0 = np.asarray(initial_p0, dtype=float)
            if initial_p0.shape != p0_base.shape:
                raise ValueError(f"initial_p0 shape {initial_p0.shape} does not match expected {p0_base.shape}")
            p0_base = np.clip(initial_p0, lower_bound, upper_bound)

        # outputs
        fitted_params = np.full((self.Y, self.X, n_params), np.nan)
        
        # Per-pixel diagnostics (added in v0.3.0)
        self.fit_diagnostics_map = np.empty((self.Y, self.X), dtype=object)
        self.fit_diagnostics_map[:, :] = None

        # ensure maps exist
        if not hasattr(self, "norm_scale_map"):
            self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        if not hasattr(self, "residual_map"):
            self.residual_map = np.full((self.Y, self.X), np.nan)

        # main loop
        p0_current = p0_base.copy()
        model_fn = self._model_dispatch()
        stride = int(self.params_per_peak)
        for j in range(self.Y):

            # IMPORTANT: prevents a bad seed at end of previous row from contaminating next row
            if warm_start and row_reset:
                p0_current = p0_base.copy()

            for i in range(self.X):
                # raw_spec = self.spectra[j, i, :] if mask is None else self.spectra[j, i, :][mask]

                # spec_fit, scale = self._preprocess_single_spectrum(xdata, raw_spec, fit_normalize=fit_normalize)

                y = np.asarray(spectra_fit_cube[j, i, :], dtype=float)
                x = xdata

                spec_fit, scale = self._prepare_fit_spectrum(x, y, fit_normalize=fit_normalize)

                if spec_fit is None:
                    self.norm_scale_map[j, i] = np.nan
                    self.residual_map[j, i] = np.nan
                    self.fit_diagnostics_map[j, i] = {"ok": False, "reason": "no_positive_signal"}
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                self.norm_scale_map[j, i] = float(scale)

                # ---- multi-start setup ----
                n_starts = int(fit_spectrum_kwargs.get("n_starts", 1))
                p0_strategy = fit_spectrum_kwargs.get("p0_strategy", "midpoint")
                random_state = fit_spectrum_kwargs.get("random_state", None)

                p0_trials = _mapping_generate_p0_trials(
                    lower_bound, upper_bound, p0_current,
                    n_starts=n_starts,
                    strategy=p0_strategy,
                    random_state=random_state,
                )

                best_params = None
                best_rmse = np.inf
                best_p0 = None
                n_fail = 0

                best_score = np.inf
                best_hits = np.inf

                width_penalty = float(fit_spectrum_kwargs.get("width_penalty", 0.0))
                prefer_nonbound = bool(fit_spectrum_kwargs.get("prefer_nonbound", False))
                score_tie_tol = float(fit_spectrum_kwargs.get("score_tie_tol", 1e-6))
                
                for p0_try in p0_trials:
                    try:
                        params, _ = optimize.curve_fit(
                            model_fn,
                            xdata,
                            spec_fit,
                            p0=p0_try,
                            bounds=(lower_bound, upper_bound),
                            maxfev=maxfev,
                        )
                    except Exception:
                        n_fail += 1
                        continue

                    y_hat = model_fn(xdata, *params)
                    rmse = float(np.sqrt(np.mean((spec_fit - y_hat) ** 2)))

                    hits = int(np.count_nonzero(_params_at_bounds(params, lower_bound, upper_bound, which="both", rtol=1e-6)))

                    # Penalised objective (score)
                    if width_penalty > 0:
                        widths = np.asarray(params[1::stride], dtype=float)
                        width_ub = np.asarray(upper_bound[1::stride], dtype=float)

                        fwhm = _width_param_to_fwhm(widths, self.peak_profile)
                        fwhm_ub = _width_param_to_fwhm(width_ub, self.peak_profile)

                        fwhm_ub = np.where(fwhm_ub > 0, fwhm_ub, 1.0)
                        pen = float(np.mean((fwhm / fwhm_ub) ** 2))
                        score = rmse + width_penalty * pen
                    else:
                        score = rmse

                    # Select by score (primary), with optional tie-break by bound hits
                    if best_params is None:
                        best_score = score
                        best_rmse = rmse
                        best_params = params
                        best_p0 = p0_try
                        best_hits = hits
                    else:
                        better = score < best_score
                        near_tie = abs(score - best_score) <= score_tie_tol

                        if better or (prefer_nonbound and near_tie and hits < best_hits):
                            best_score = score
                            best_rmse = rmse
                            best_params = params
                            best_p0 = p0_try
                            best_hits = hits

                # ---- fail all starts ----
                if best_params is None:
                    self.residual_map[j, i] = np.nan
                    fitted_params[j, i, :] = np.nan
                    self.fit_diagnostics_map[j, i] = {
                        "ok": False,
                        "n_starts": n_starts,
                        "n_fail": n_fail,
                        "p0_strategy": p0_strategy,
                    }
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                # ---- success ----
                self.residual_map[j, i] = best_rmse
                fitted_params[j, i, :] = best_params
                self.fit_diagnostics_map[j, i] = {
                    "ok": True,
                    "rmse": best_rmse,
                    "n_starts": n_starts,
                    "n_fail": n_fail,
                    "p0_strategy": p0_strategy,
                    "best_p0": np.asarray(best_p0, dtype=float),
                }
                at_lo = _params_at_bounds(best_params, lower_bound, upper_bound, which="lower", rtol=1e-6)
                at_hi = _params_at_bounds(best_params, lower_bound, upper_bound, which="upper", rtol=1e-6)
                self.fit_diagnostics_map[j, i]["n_params_at_lower_bounds"] = int(np.count_nonzero(at_lo))
                self.fit_diagnostics_map[j, i]["n_params_at_upper_bounds"] = int(np.count_nonzero(at_hi))
                self.fit_diagnostics_map[j, i]["params_at_lower_bounds_mask"] = at_lo
                self.fit_diagnostics_map[j, i]["params_at_upper_bounds_mask"] = at_hi

                # ---- store peak centre + intensity (peak height) ----
                stride = int(self.params_per_peak)

                for k in range(n_peaks):
                    block = np.asarray(best_params[stride*k:stride*(k+1)], dtype=float)

                    center = float(block[0])
                    self.peak_positions[j, i, k] = center

                    if self.peak_profile == "lorentzian":
                        width = float(block[1])       # HWHM
                        amp  = float(block[2])        # area-like
                        if (not np.isfinite(width)) or width <= 0:
                            height_fitspace = np.nan
                        else:
                            height_fitspace = amp / (np.pi * width)
                    else:
                        # pVoigt: compute peak height numerically in fit space
                        y_one = single_peak(xdata, block, profile="pvoigt")
                        height_fitspace = float(np.nanmax(y_one))

                    # DISPLAY convention: normalize=True -> store fit-space height
                    # normalize=False -> store raw height using per-pixel scale when fit_normalize=True
                    if self.normalize:
                        self.peak_intensities[j, i, k] = height_fitspace
                    else:
                        if fit_normalize:
                            self.peak_intensities[j, i, k] = height_fitspace * float(scale)
                        else:
                            self.peak_intensities[j, i, k] = height_fitspace



                # ---- derived maps (compute once per pixel) ----
                # idx_a1g = self._find_peak_index("A1g")
                # idx_e2g = self._find_peak_index("E2g")
                if idx_e2g is None:
                    idx_e2g = self._find_peak_index("E12g")

                if (idx_a1g is not None) and (idx_e2g is not None):
                    a1g_pos = self.peak_positions[j, i, idx_a1g]
                    e2g_pos = self.peak_positions[j, i, idx_e2g]
                    a1g_I   = self.peak_intensities[j, i, idx_a1g]
                    e2g_I   = self.peak_intensities[j, i, idx_e2g]

                    self.Peaks_distance[j, i] = (a1g_pos - e2g_pos) if (np.isfinite(a1g_pos) and np.isfinite(e2g_pos)) else np.nan
                    self.ratio_A1g_E2g[j, i]  = (a1g_I / e2g_I) if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and e2g_I > 0) else np.nan
                    self.ratio_E2g_A1g[j, i]  = (e2g_I / a1g_I) if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and a1g_I > 0) else np.nan
                else:
                    self.Peaks_distance[j, i] = np.nan
                    self.ratio_A1g_E2g[j, i]  = np.nan
                    self.ratio_E2g_A1g[j, i]  = np.nan

                # ---- gated warm-start (RMSE + plausibility) ----
                if warm_start:
                    ok_rmse = (best_rmse <= warm_start_rmse_gate)
                    stride = int(self.params_per_peak)
                    ok_params = _params_plausible(best_params, lower_bound, upper_bound, n_peaks, stride, self.peak_profile)

                    if ok_rmse and ok_params:
                        p0_current = np.asarray(best_params, dtype=float)
                    else:
                        if reset_on_fail:
                            p0_current = p0_base.copy()

        self.fitted_params = fitted_params
        n_fit = np.sum(~np.isnan(self.residual_map))
        print(f"Successful fits: {n_fit} / {self.X * self.Y}")
        return fitted_params

    
    def plot_spectrum_fit(self, x, y):
        """Plot raw data and fitting results for a single map point.

        Display logic (PL-equivalent):
            - Fitting is always done in normalised space.
            - normalize=True  -> show normalised, background-removed spectrum + fit
            - normalize=False -> show raw spectrum + fit (+ background overlay if enabled)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter

        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # --- Extract full spectrum
        x_full = np.asarray(self.wavenumber, dtype=float)
        y_full = np.asarray(self.spectra[y, x, :], dtype=float)

        # --- Mask by wavenumber range (cm^-1)
        mask = DataImporter.mask_by_xrange(x_full, self.data_range)
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
        # Prefer stored scale if available (ensures exact match to fit_spectra)
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
                plt.plot(xdata, background, "r--", label="Estimated Background")
                plt.plot(xdata, bg_removed, "b-", alpha=0.8, label="Background Removed (smoothed)")

                # Peak-only fit (raw units)
                plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve (peak only)")

                # Overlay vs raw spectrum: (peak fit + estimated background)
                fitted_plus_bg = fitted_raw + background
                plt.plot(xdata, fitted_plus_bg, "-", linewidth=2, label="Fit + Estimated Background")
            else:
                # No background removal → show fit only once
                plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve")

            plt.ylabel("Intensity (a.u.)")

        plt.xlabel("Wavenumber (cm⁻¹)")
        title = f"Spectrum Fit at (X={x}, Y={y})"
        if hasattr(self, "residual_map") and np.isfinite(self.residual_map[y, x]):
            title += f" | RMSE(norm)={self.residual_map[y, x]:.4g}"
        plt.title(title)

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

        Notes:
        - residual_map is RMSE computed in the *normalised fit space* (dimensionless).
        """
        import numpy as np
        import matplotlib.pyplot as plt

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
            data_masked = np.ma.masked_invalid(residuals)
            title = "Residual Distribution (higher = worse fit)"
        else:
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
        cbar.set_label("Residual Error (RMSE, normalised)")

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
        ax_hist.set_ylabel("Residual RMSE (normalised)")
        ax_hist.set_title("Residual Histogram")

        if filter_threshold is not None:
            ax_hist.axhline(filter_threshold, linestyle="--", linewidth=1)
            upper = residuals_flat[residuals_flat >= filter_threshold]
            y_lo = max(filter_threshold * 0.98, np.min(upper))
            y_hi = np.max(upper)
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
                ax_hist.set_ylim(y_lo, y_hi)
        else:
            if vmin is not None and vmax is not None:
                ax_hist.set_ylim(vmin, vmax)

        plt.tight_layout()
        plt.show()

    def plot_ratio_heatmap(self, ratio_type='A1g/E2g', cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """Visualize 2D map of peak intensity ratios.
        
        Args:
            ratio_type: 'A1g/E2g' or 'E2g/A1g'
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
            
        Raises:
            ValueError: For invalid ratio types or missing peaks
        """
        # Ensure derived maps exist (in case user calls this before fit_spectra)
        if not hasattr(self, "ratio_A1g_E2g") or not hasattr(self, "ratio_E2g_A1g"):
            raise ValueError("Ratio maps not initialised. Run fit_spectra() first.")

        # Choose ratio map
        if ratio_type == 'A1g/E2g':
            data = self.ratio_A1g_E2g
            label = 'A1g/E2g Intensity Ratio'
        elif ratio_type == 'E2g/A1g':
            data = self.ratio_E2g_A1g
            label = 'E2g/A1g Intensity Ratio'
        else:
            raise ValueError("Invalid ratio_type. Choose from 'A1g/E2g' or 'E2g/A1g'.")

        # Filter range: clip outliers only if requested
        if filter_range is not None:
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, np.nan)

        # Crop
        if x_range is not None and y_range is not None:
            x_start, x_end = x_range
            y_start, y_end = y_range
            data = data[y_start:y_end+1, x_start:x_end+1]
            x_length = (x_end - x_start + 1) * self.step_size
            y_length = (y_end - y_start + 1) * self.step_size
        else:
            x_length = self.X * self.step_size
            y_length = self.Y * self.step_size

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

    ### Updated in v0.3.8
    @staticmethod
    def custom_peaks_from_ramanfit(raman_fit):
        """
        Build Mapping-compatible custom_peaks dict from a RamanFit instance.

        Supports both lorentzian (3 params/peak) and pvoigt (4 params/peak).
        """
        import numpy as np

        labels = list(raman_fit.peak_labels)
        lb = np.asarray(raman_fit.lower_bound, dtype=float)
        ub = np.asarray(raman_fit.upper_bound, dtype=float)
        stride = int(getattr(raman_fit, "params_per_peak", 3))

        if lb.size != ub.size or lb.size != stride * len(labels):
            raise ValueError("RamanFit bounds length mismatch with peak_labels.")

        out = {}
        for k, name in enumerate(labels):
            s = stride * k
            out[name] = (lb[s:s+stride].tolist(), ub[s:s+stride].tolist())
        return out

    def _params_to_export_dict(self, xaxis, peak_labels, params, intensity_scale=1.0):
        """
        Convert a parameter vector into per-peak export dict entries.

        Conventions
        -----------
        - Lorentzian: width is HWHM; FWHM = 2*HWHM; peak_height_norm = amp_area/(pi*HWHM)
        - pVoigt: width parameter is treated as FWHM (consistent with your PLfit/RamanFit pVoigt step);
                peak_height_norm is computed numerically as max(single_peak(xaxis)).
        """
        import numpy as np

        profile = self.peak_profile
        stride = int(self.params_per_peak)
        p = np.asarray(params, dtype=float).ravel()
        xaxis = np.asarray(xaxis, dtype=float).ravel()

        out = {}
        for i, name in enumerate(peak_labels):
            block = p[stride*i:stride*(i+1)]
            centre = float(block[0])

            if profile == "lorentzian":
                hwhm = float(block[1])
                amp_area = float(block[2])
                fwhm = 2.0 * hwhm
                peak_height_norm = (amp_area / (np.pi * hwhm)) if hwhm != 0 else np.nan
                peak_height = float(peak_height_norm * intensity_scale)
                out[name] = dict(
                    centre=centre, fwhm=fwhm,
                    peak_height=peak_height,
                    peak_height_norm=float(peak_height_norm),
                    amp=amp_area, scale=hwhm,
                )
            else:
                fwhm = float(block[1])       # pVoigt width treated as FWHM
                amp_area = float(block[2])
                eta = float(block[3])

                y_norm = single_peak(xaxis, block, profile="pvoigt")
                peak_height_norm = float(np.nanmax(y_norm))
                peak_height = float(peak_height_norm * intensity_scale)

                out[name] = dict(
                    centre=centre, fwhm=fwhm,
                    peak_height=peak_height,
                    peak_height_norm=peak_height_norm,
                    amp=amp_area, scale=fwhm, eta=eta,
                )

        return out


    ### Updated in v0.3.8
    def export_fit_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        scaled: bool = True,
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
    ) -> str:
        """
        Export fit results for every pixel in wide format:
        x, y, then per-peak parameters on the same row.
        """
        if not hasattr(self, "fitted_params") or self.fitted_params is None:
            raise ValueError("No fitted_params found. Run fit_spectra() first.")

        coord_mode = normalise_coord_mode(coord_mode)

        peak_labels = list(self.peak_params)
        fields = ["x", "y"]

        per_peak_fields = ["centre", "fwhm", "peak_height", "peak_height_norm", "amp", "scale"]
        if self.peak_profile == "pvoigt":
            per_peak_fields.append("eta")

        for p in peak_labels:
            for f in per_peak_fields:
                fields.append(f"{p}_{f}")

        rows = []
        for x, y, j, i in self._iter_coords(coord_mode=coord_mode):
            params = np.asarray(self.fitted_params[j, i, :], dtype=float)
            if np.any(np.isnan(params)):
                rows.append({"x": x, "y": y})
                continue

            intensity_scale = 1.0
            if scaled and hasattr(self, "norm_scale_map") and np.isfinite(self.norm_scale_map[j, i]):
                intensity_scale = float(self.norm_scale_map[j, i])

            per_peak = self._params_to_export_dict(
                self.wavenumber,
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

            rows.append(r)

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
    
    def plot_heatmap(self, data_type='position', cmap='viridis', filter_range=None, 
                    x_range=None, y_range=None, specific_wavenumber=None, peak_name=None):
        """Visualize 2D map of spectral features.
        
        Args:
            data_type: Plot type ('position', 'intensity', 'specific_intensity', 'distance')
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            specific_wavenumber: Wavenumber for 'specific_intensity' plots
            peak_name: Peak name for position/intensity plots
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
            
        Raises:
            ValueError: For invalid data types or missing parameters
        """
        # Handle input validation dynamically
        if data_type in ['position', 'intensity']:
            if peak_name is None or peak_name not in self.peak_params:
                raise ValueError(f"Must provide valid peak_name for {data_type} plots")
        elif data_type == 'specific_intensity':
            if specific_wavenumber is None:
                raise ValueError("Must provide specific_wavenumber for intensity at spectra")
        elif data_type == 'distance':
            pass  # No peak_name needed
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        ### Updated in v0.2.4 ###
        # Generate data based on data_type
        if data_type == 'specific_intensity':
            data = np.full((self.Y, self.X), np.nan, dtype=float)

            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]
                    if np.any(np.isnan(params)):
                        continue  # fit failed / not available

                    model_fn = self._model_dispatch()
                    y_norm = model_fn(np.asarray([specific_wavenumber], dtype=float), *params)[0]

                    if self.normalize:
                        # display normalised model intensity
                        data[j, i] = y_norm
                    else:
                        # display raw model intensity using stored scale
                        if not hasattr(self, "norm_scale_map") or np.isnan(self.norm_scale_map[j, i]):
                            continue
                        data[j, i] = y_norm * self.norm_scale_map[j, i]

            label = (f'Normalised intensity at {specific_wavenumber} cm⁻¹'
                    if self.normalize else
                    f'Intensity at {specific_wavenumber} cm⁻¹')
        ### End UPDATED METHOD ###

        elif data_type == 'distance':
            data = self.Peaks_distance
            label = 'A1g - E2g Distance (cm⁻¹)'
        else:
            peak_idx = self.peak_params.index(peak_name)
            data = (self.peak_positions[:, :, peak_idx] if data_type == 'position'
                    else self.peak_intensities[:, :, peak_idx])
            label = f'{peak_name} {data_type.capitalize()}'

        # Filter data range
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
        cbar = plt.colorbar(im, label=label)
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.title(f"Heatmap of {label}")
        plt.show()


########################################################################################################################
############################################ Raman Integration without Peak fitting ####################################
########################################################################################################################

class Raman_Integration:
    """Raman mapping analysis through spectral integration.
    
    Attributes:
        filename (str): Path to input file
        integration_range (tuple): Spectral integration range (min, max) in cm⁻¹
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        background_remove (bool): Enable background subtraction
        X (int): Map width in pixels
        Y (int): Map height in pixels
        wavenumber (ndarray): Spectral axis in cm⁻¹
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        integration_area (ndarray): Integrated intensities [Y, X]
    """   
    def __init__(self, filename, integration_range,
                step_size=0.3, header=False,
                poly_degree=3, background_remove=True,
                baseline_method="poly"):
        """Initialize Raman integration analyzer.
        
        Args:
            filename: Path to .wdf file
            integration_range: Spectral range (min, max) in cm⁻¹
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            background_remove: Enable background subtraction
        """
        self.filename = filename
        self.integration_range = integration_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.background_remove = background_remove

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"
        self.step_unit = "um"


        # integration_range is known here; trim at load-time.
        loader = MappingFileLoader(filename, x_range=self.integration_range, axis="wavenumber")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

        # New in v0.3.8 Baseline configuration (single source of truth; backward compatible)
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
        )
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            self.baseline_method
        )

    ### New classmethod in v0.2.7 ###
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
        Construct Raman_Integration from in-memory arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Wavenumber axis (cm^-1) with shape [N]
        X, Y : int
            Map dimensions
        integration_range : tuple(float, float)
            (min, max) cm^-1; applied immediately (trim at construction, consistent with __init__)
        """
        obj = cls.__new__(cls)

        obj.filename = None
        obj.integration_range = integration_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.background_remove = background_remove
        obj.spectrum_type = "Raman"
        obj.x_quantity = "Raman shift"
        obj.x_unit = "cm^-1"
        obj.step_unit = "um"

        # Baseline configuration (same pattern as __init__)
        obj.baseline_method = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
        )
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            obj.baseline_method
        )

        obj.X = int(X)
        obj.Y = int(Y)

        wn = np.asarray(xdata, dtype=float).ravel()
        cube = np.asarray(spectra, dtype=float)

        # Validate shapes
        if cube.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if cube.shape[0] != obj.Y or cube.shape[1] != obj.X:
            raise ValueError(f"spectra shape {cube.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if cube.shape[2] != wn.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # Trim at load-time (same semantics as filename-based __init__)
        wmin, wmax = integration_range
        mask = (wn >= wmin) & (wn <= wmax)
        if not np.any(mask):
            raise ValueError(
                f"integration_range {integration_range} does not overlap provided wavenumber axis "
                f"[{float(np.min(wn)):.3g}, {float(np.max(wn)):.3g}]."
            )

        obj.wavenumber = wn[mask]
        obj.spectra = cube[:, :, mask]
        obj._x_trimmed_on_load = True

        if clip_nonnegative:
            obj.spectra = np.clip(obj.spectra, a_min=0.0, a_max=None)

        obj.image_viewer = None
        obj.integration_area = np.zeros((obj.Y, obj.X), dtype=float)

        return obj

    def show_optical_image(self):
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    ## Updated in v0.2.5 ##
    def remove_background(self, wavenumber, intensity):
        """Background removal via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=wavenumber,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected
    
    ## Updated in v0.2.6 ##
    def calculate_integration(self):
        """Calculate integrated area using Simpson's rule.

        Stores results in integration_area array.
        """
        wavenumber = np.asarray(self.wavenumber, dtype=float).ravel()

        mask = DataImporter.mask_by_xrange(wavenumber, self.integration_range)
        wavenumber_subset = wavenumber[mask]

        for j in range(self.Y):
            for i in range(self.X):
                spectra = np.asarray(self.spectra[j, i, :], dtype=float).ravel()
                spectra_subset = spectra[mask]

                if self.background_remove:
                    spectra_subset = self.remove_background(wavenumber_subset, spectra_subset)

                self.integration_area[j, i] = np.abs(simpson(spectra_subset, wavenumber_subset))


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
        plt.title(f"Integration Area Heatmap ({self.integration_range[0]} - {self.integration_range[1]} cm⁻¹)")
        plt.show()

    def plot_spectrum(self, x, y):
        """Plot raw and processed spectra for single map point.
        
        Args:
            x: X coordinate (0-indexed)
            y: Y coordinate (0-indexed)
            
        Raises:
            ValueError: For invalid coordinates
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # Get the original spectrum data
        wavenumber = self.wavenumber[:]
        spectra = self.spectra[y][x][:]

        # Subset within integration range
        mask = DataImporter.mask_by_xrange(wavenumber, self.integration_range)
        wavenumber_subset = wavenumber[mask]
        spectra_subset = spectra[mask]

        # Keep raw copy for plotting
        spectra_raw = spectra_subset.copy()

        # Background removal (define spectra_bg_removed in all cases)
        if self.background_remove:
            spectra_bg_removed = self.remove_background(wavenumber_subset, spectra_subset)
        else:
            spectra_bg_removed = spectra_subset

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber_subset, spectra_raw, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(wavenumber_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.show()


    ### Updated in v0.3.8
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