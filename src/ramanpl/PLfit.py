"""
A module for analyzing photoluminescence (PL) spectra through Lorentzian curve fitting.

This module provides tools for preprocessing PL data (smoothing, background subtraction),
fitting Exciton and Trion peaks using Lorentzian functions, and visualizing the results.

Classes:
    PLfit: Main class for processing, fitting, and visualizing PL spectra.
    DataImporter: Class for importing Raman data from .wdf and .txt files (single spectrum only)
"""
from renishawWiRE import WDFReader
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ramanpl import BaselineAPI
from ramanpl import DataImporter
from ramanpl.exporter import params_to_rows, write_rows, write_table

class DataImporter:
    """
    Compatibility shim: keeps PLfit.DataImporter.data_import(...) working,
    while delegating to the shared importer.

    Preferred usage going forward:
        from ramanpl.dataImporter import DataImporter
    """
    @staticmethod
    def data_import(filename, readlines=(300, 780), x_range=None):
        from ramanpl.dataImporter import DataImporter as _Shared
        return _Shared.data_import(filename=filename, readlines=readlines, x_range=x_range, axis="energy")

# ----------------------------
# Multi-start + diagnostics helpers (Step 1)
# ----------------------------
def _rng(random_state=None):
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(r * r)))


def _params_at_bounds(params: np.ndarray, lb: np.ndarray, ub: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-12) -> np.ndarray:
    """
    Return boolean mask of parameters that are effectively on their lower/upper bounds.
    """
    params = np.asarray(params, dtype=float).ravel()
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    on_lb = np.isclose(params, lb, rtol=rtol, atol=atol)
    on_ub = np.isclose(params, ub, rtol=rtol, atol=atol)
    return on_lb | on_ub


def _generate_p0_trials(
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    base_p0: np.ndarray,
    n_starts: int,
    strategy: str,
    random_state=None
) -> list:
    """
    Generate a list of p0 vectors within bounds.

    strategy:
        - "midpoint": (default) uses base_p0 only (unless n_starts>1, then jitter about base_p0)
        - "random": uniform random within bounds
        - "jitter": Gaussian jitter about base_p0, then clipped to bounds
    """
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    base_p0 = np.asarray(base_p0, dtype=float).ravel()

    if n_starts is None:
        n_starts = 1
    n_starts = int(n_starts)
    if n_starts < 1:
        n_starts = 1

    strategy = (strategy or "midpoint").lower()
    if strategy not in {"midpoint", "random", "jitter"}:
        raise ValueError("p0_strategy must be one of: 'midpoint', 'random', 'jitter'.")

    trials = [base_p0.copy()]
    if n_starts == 1:
        return trials

    rng = _rng(random_state)
    m = n_starts - 1

    if strategy == "random":
        for _ in range(m):
            trials.append(rng.uniform(lb, ub))
        return trials

    # "midpoint" and "jitter" both jitter about base_p0 for extra starts
    scale = 0.10 * (ub - lb)
    scale = np.where(scale > 0, scale, 1.0)  # guard against zero ranges
    for _ in range(m):
        p = base_p0 + rng.normal(loc=0.0, scale=scale)
        p = np.clip(p, lb, ub)
        trials.append(p)

    return trials


class PLfit:
    """A class for processing and fitting photoluminescence spectra with Lorentzian functions.
    
    Handles data preprocessing (smoothing, background subtraction), curve fitting,
    and visualization of results for Exciton and Trion peaks.

    Attributes:
        raw_spectra (ndarray): Raw intensity values from the input spectrum
        processed_spectra (ndarray): Processed intensity values after preprocessing
        energy (ndarray): Energy values (x-axis) for the spectrum in eV
        peak_intensity (float): Maximum intensity value for normalization
        intensity_normal (ndarray): Normalized intensity values
        lower_bound (list): Lower bounds for fitting parameters
        upper_bound (list): Upper bounds for fitting parameters
        peak_labels (list): Names of peaks being fit (Trion and Exciton)
        p0 (list): Initial parameter guesses for curve fitting

    Methods:
        __init__: Initialize PLfit object with data and preprocessing options
        update_bounds: Modify fitting constraints for specific peaks
        lorentzian_pl: Static Lorentzian function for curve fitting
        fit_spectrum: Perform the curve fitting operation
        plot_fit: Visualize data, fit results, and components
    """

    def __init__(self, spectra, energy, background_remove=False, baseline_method='poly',
             poly_degree=3, gaussian_sigma=50, smoothing=False,
             smooth_window=11, smooth_order=3, normalize=True,
             custom_peaks=None, remove_peaks=None, peak_order=None,
             peak_profile: str = "lorentzian"
             ):
        """Initialize PLfit object with data and processing parameters.

        Parameters:
            spectra (array-like): PL intensity values (y-axis)
            energy (array-like): Corresponding energy values in eV (x-axis)
            background_remove (bool): Enable background subtraction (default: False)
            baseline_method (str): Background method 'poly' or 'gaussian' (default: 'poly')
            poly_degree (int): Polynomial degree for poly background (default: 3)
            gaussian_sigma (int): Sigma for Gaussian filter (default: 50)
            smoothing (bool): Enable Savitzky-Golay smoothing (default: False)
            smooth_window (int): Window size for smoothing filter (default: 11)
            smooth_order (int): Polynomial order for smoothing (default: 3)
            normalize (bool):  controls DISPLAY/OUTPUT scaling only. Fitting is ALWAYS performed in peak-normalised space.

        Raises:
            ValueError: If invalid baseline method is specified
        """
        self.raw_spectra = np.array(spectra)
        self.energy = np.array(energy)
        self.processed_spectra = np.array(spectra.copy())
        
        # Added in build v0.2.7.1
        self._smoothed_spectra = None
        self._baseline = None
        self._corrected_spectra = None

        ## New in v0.2.8: store preprocessing settings
        # --- store preprocessing settings for reproducibility / export metadata ---
        self.spectrum_type = "Photoluminescence"
        self.x_quantity = "Photon energy"
        self.x_unit = "eV"

        self.background_remove = background_remove
        self.baseline_method = baseline_method
        self.poly_degree = poly_degree
        self.gaussian_sigma = gaussian_sigma

        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        self.peak_order = peak_order

        # New in v0.3.3
        self.peak_profile = str(peak_profile).lower().strip()
        if self.peak_profile not in ("lorentzian", "pvoigt"):
            raise ValueError("peak_profile must be 'lorentzian' or 'pvoigt'")

        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        ## Modified in build v0.2.7.1
        # Apply smoothing
        if smoothing:
            self.processed_spectra = savgol_filter(self.processed_spectra,
                                                smooth_window, smooth_order)
            self._smoothed_spectra = self.processed_spectra.copy()

        # Background subtraction (smoothing happens before this; unchanged)
        if background_remove:
            method, bkwargs = BaselineAPI.parse_spec(
                baseline_method,
                poly_degree=poly_degree,
                gaussian_sigma=gaussian_sigma
            )

            result = BaselineAPI.subtract(
                x=self.energy,
                y=self.processed_spectra,
                method=method,
                clip_nonnegative=True,  # always clip
                **bkwargs,
            )

            # --- store intermediates for comparison plotting ---
            self._baseline = np.asarray(result.baseline, dtype=float).ravel()
            self._corrected_spectra = np.asarray(result.y_corrected, dtype=float).ravel()

            # existing behaviour
            self.processed_spectra = result.y_corrected
        else:
            if smoothing:
                self._corrected_spectra = self.processed_spectra.copy()

        # DISPLAY flag (fit is always normalised)
        self.normalize = normalize

        # Peak normalisation for fitting space
        self.peak_intensity = np.max(self.processed_spectra)
        if self.peak_intensity <= 0:
            raise ValueError("Peak intensity is non-positive after preprocessing; cannot normalise for fitting.")
        self.intensity_normal = self.processed_spectra / self.peak_intensity

        # ---- Updated in v0.3.3 ---- #
        self.custom_peaks = custom_peaks  # may be None

        if custom_peaks is None:
            # Backwards-compatible defaults (your existing behaviour)
            if self.peak_profile == "lorentzian":
                self.lower_bound = [1.95, 0, 0,  1.8, 0, 0]
                self.upper_bound = [2.1, 0.05, 10, 2.0, 0.2, 10]
                self.peak_labels = ['trion', 'exciton']
            else:  # pvoigt
                self.lower_bound = [1.95, 0, 0, 0.0,  1.8, 0, 0, 0.0]
                self.upper_bound = [2.1, 0.05, 10, 1.0, 2.0, 0.2, 10, 1.0]
                self.peak_labels = ['trion', 'exciton']
        else:
            if not isinstance(custom_peaks, dict) or len(custom_peaks) == 0:
                raise ValueError("custom_peaks must be a non-empty dict: {name: ([lb...],[ub...])}")

            # Stable ordering contract
            if peak_order is None:
                self.peak_order = list(custom_peaks.keys())
            else:
                self.peak_order = list(peak_order)
                missing = [k for k in self.peak_order if k not in custom_peaks]
                if missing:
                    raise ValueError(f"peak_order contains keys not in custom_peaks: {missing}")

            self.peak_labels = list(self.peak_order)

            self.lower_bound, self.upper_bound = [], []
            expected = self.params_per_peak
            for name in self.peak_labels:
                lb, ub = custom_peaks[name]
                if len(lb) != expected or len(ub) != expected:
                    if expected == 3:
                        raise ValueError(f"Peak '{name}' bounds must be length-3 lists: [centre, width, amp]")
                    else:
                        raise ValueError(f"Peak '{name}' bounds must be length-4 lists: [centre, width, amp, eta]")
                self.lower_bound += list(lb)
                self.upper_bound += list(ub)

        # Initial guess at midpoint of bounds
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

        self.remove_peaks_list = list(remove_peaks) if remove_peaks is not None else []
        if self.remove_peaks_list:
            self.remove_peaks(*self.remove_peaks_list)

        # ---- NEW in v0.2.3: slots for exporting to mapping
        self.params_fit = None
        self.params_cov = None


    ### UPDATED METHOD in v0.3.3 ###
    def update_bounds(self, **kwargs):
        """Update fitting constraints for specific peaks.

        Parameters:
            **kwargs: Peak name and bounds tuple pairs (e.g., Trion=([lb1, lb2, lb3], [ub1, ub2, ub3]))

        Raises:
            ValueError: For unrecognized peak names or invalid bound formats

        Example:
            >>> pl.update_bounds(Trion=([1.9, 0.01, 1], [2.0, 0.1, 5]),
            ...                  Exciton=([1.7, 0.01, 1], [1.9, 0.1, 5]))
        """
        stride = self.params_per_peak

        for peak_name, new_bounds in kwargs.items():
            peak_key = str(peak_name).lower()
            labels_lower = [p.lower() for p in self.peak_labels]

            if peak_key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            if not (isinstance(new_bounds, (tuple, list)) and len(new_bounds) == 2):
                raise ValueError(f"Bounds for '{peak_name}' must be (lb, ub).")

            lb_new, ub_new = new_bounds
            if len(lb_new) != stride or len(ub_new) != stride:
                raise ValueError(
                    f"Peak '{peak_name}' bounds must be length-{stride} lists."
                )

            idx = labels_lower.index(peak_key)
            s = stride * idx

            self.lower_bound[s : s + stride] = list(lb_new)
            self.upper_bound[s : s + stride] = list(ub_new)
            self.p0[s : s + stride] = [(lb_new[i] + ub_new[i]) / 2 for i in range(stride)]

    # UPDATED IN V0.3.3
    def remove_peaks(self, *peak_names):
        """
        Remove peaks from the fitting model.

        Parameters
        ----------
        *peak_names : str
            Peak names to remove (case-insensitive match allowed).

        Raises
        ------
        ValueError
            If a peak name is not present.
        """
        stride = self.params_per_peak

        for peak_name in peak_names:
            labels_lower = [p.lower() for p in self.peak_labels]  # refresh each loop (simplest)
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(
                    f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}"
                )

            idx = labels_lower.index(key)

            # remove blocks
            del self.p0[stride * idx : stride * idx + stride]
            del self.lower_bound[stride * idx : stride * idx + stride]
            del self.upper_bound[stride * idx : stride * idx + stride]

            # remove label to keep everything aligned
            del self.peak_labels[idx]

        # Optional: if you maintain peak_order elsewhere, keep it consistent too
        if hasattr(self, "peak_order") and isinstance(self.peak_order, list):
            self.peak_order = list(self.peak_labels)

    ## Updated in v0.3.0
    def _model(self, x, *params):
        """Internal model dispatcher (Lorentzian or pseudo-Voigt)."""
        try:
            from .peak_models import sum_peaks
        except Exception:  # pragma: no cover
            from peak_models import sum_peaks

        return sum_peaks(
            np.asarray(x),
            params,
            profile=("pvoigt" if self.peak_profile == "pvoigt" else "lorentzian"),
            stride=self.params_per_peak,
        )

    def fit_spectrum(
        self,
        *,
        n_starts: int = 1,
        p0_strategy: str = "midpoint",
        random_state=None,
        diagnose_bounds: bool = True,
        bounds_tol: float = 1e-6,
        return_diagnostics: bool = False,
    ):
        """
        Perform Lorentzian curve fitting with bounded least squares.

        Backwards-compatible default:
            fit_spectrum() behaves as before (single start from self.p0) when n_starts=1.

        Parameters
        ----------
        n_starts
            Number of optimisation restarts. Best solution (lowest RMSE) is retained.
        p0_strategy
            How to generate starting points: "midpoint" (default), "random", or "jitter".
            Note: for n_starts>1, "midpoint" will jitter around the midpoint.
        random_state
            Seed for reproducible multi-start initialisations (used only when n_starts > 1).
        diagnose_bounds
            If True, compute and store whether fitted parameters are on bounds.
        bounds_tol
            Relative tolerance passed to np.isclose for bound-hit detection.
        return_diagnostics
            If True, return (params, params_cov, diagnostics_dict).

        Returns
        -------
        (params, params_cov) or (params, params_cov, diagnostics)
        """
        lb = np.asarray(self.lower_bound, dtype=float).ravel()
        ub = np.asarray(self.upper_bound, dtype=float).ravel()
        base_p0 = np.asarray(self.p0, dtype=float).ravel()

        p0_trials = _generate_p0_trials(
            lb, ub, base_p0=base_p0,
            n_starts=n_starts, strategy=p0_strategy,
            random_state=random_state,
        )

        best_params = None
        best_cov = None
        best_rmse = np.inf
        best_p0 = None
        n_fail = 0

        for p0 in p0_trials:
            try:
                params, params_cov = optimize.curve_fit(
                    self._model,
                    self.energy,
                    self.intensity_normal,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=6400,
                )
            except Exception:
                n_fail += 1
                continue

            y_hat = y_hat = self._model(self.energy, *params)
            rmse = _rmse(self.intensity_normal, y_hat)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_cov = params_cov
                best_p0 = p0

        if best_params is None:
            raise RuntimeError(
                f"PLfit.fit_spectrum failed for all starts (n_starts={n_starts}). "
                "Check bounds/preprocessing, or reduce model complexity."
            )

        self.params_fit = best_params
        self.params_cov = best_cov

        diagnostics = {
            "rmse": float(best_rmse),
            "n_starts": int(n_starts),
            "n_fail": int(n_fail),
            "p0_strategy": str(p0_strategy),
            "best_p0": np.asarray(best_p0, dtype=float),
        }

        if diagnose_bounds:
            at_bounds = _params_at_bounds(np.asarray(best_params, dtype=float), lb, ub, rtol=float(bounds_tol))
            diagnostics["n_params_at_bounds"] = int(np.count_nonzero(at_bounds))
            diagnostics["params_at_bounds_mask"] = at_bounds

        self.fit_diagnostics = diagnostics

        if return_diagnostics:
            return best_params, best_cov, diagnostics
        return best_params, best_cov
    
    ### NEW METHOD in v0.2.9 ###
    def get_fitted_spectrum(self):
        """
        Return fitted spectrum on the same x-grid as the input data.

        Returns
        -------
        x : np.ndarray
            Energy axis (eV)
        y_fit : np.ndarray
            Fitted intensity in the SAME units as self.processed_spectra.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("PLfit has not been fitted yet. Run fit_spectrum() first.")

        x = np.asarray(self.energy, dtype=float).ravel()

        y_fit_norm = self._model(x, *self.params_fit)            # normalised fit space
        y_fit = y_fit_norm * float(self.peak_intensity)          # back to processed intensity scale

        return x.copy(), np.asarray(y_fit, dtype=float).ravel().copy()

    ### Updated v0.3.3 ###
    def get_fitted_parameters(self):
        """
        Return fitted peak parameters as a structured dict.

        Notes
        -----
        Parameter vector layout depends on peak_profile:
        - lorentzian: (centre, HWHM, amp_area) per peak
        - pvoigt:     (centre, FWHM, amp_area, eta) per peak

        Reported peak_height is the *peak maximum* in processed intensity units.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("Fit not available. Run fit_spectrum() first.")
        if not hasattr(self, "peak_labels") or not self.peak_labels:
            raise RuntimeError("No peak labels found; cannot map parameters to peaks.")

        p = np.asarray(self.params_fit, dtype=float).ravel()

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))
        if profile not in ("lorentzian", "pvoigt"):
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in get_fitted_parameters().")

        expected = stride * len(self.peak_labels)
        if p.size < expected:
            raise RuntimeError(
                f"params_fit has length {p.size}, but expected at least {expected} "
                f"for {len(self.peak_labels)} peaks (stride={stride})."
            )

        fit_scale = float(self.peak_intensity) if hasattr(self, "peak_intensity") else 1.0

        if profile == "pvoigt":
            try:
                from .peak_models import single_peak
            except Exception:  # pragma: no cover
                from peak_models import single_peak

        out = {}

        for i, name in enumerate(self.peak_labels):
            block = p[stride * i : stride * (i + 1)]
            loc = float(block[0])
            width = float(block[1])       # HWHM (lorentzian) or FWHM (pvoigt)
            amp_area = float(block[2])    # area-like amplitude (both)

            if profile == "lorentzian":
                fwhm = 2.0 * width
                height_norm = (amp_area / (np.pi * width)) if width != 0 else np.nan
                row = dict(
                    position=loc,
                    fwhm=float(fwhm),
                    scale=float(width),          # HWHM
                    amp=float(amp_area),
                    height_norm=float(height_norm),
                    peak_height=float(height_norm * fit_scale),
                )
            else:
                eta = float(block[3])
                fwhm = width  # pVoigt width parameter is FWHM in your peak_models
                y_comp_norm = single_peak(self.energy, block, profile="pvoigt")
                height_norm = float(np.max(y_comp_norm))
                row = dict(
                    position=loc,
                    fwhm=float(fwhm),
                    fwhm_param=float(width),     # explicit: stored width is FWHM
                    eta=float(eta),
                    amp=float(amp_area),
                    height_norm=float(height_norm),
                    peak_height=float(height_norm * fit_scale),
                )

            out[name] = row

        return out


    ### New in v0.2.8 ###
    def fit_table(self, params=None, *, scaled: bool = True):
        """
        Return per-peak fitted parameters as a list of dicts.

        scaled=True:
            height_scaled is reported in approximate original units by multiplying
            normalised peak height by self.peak_intensity (if available).
        """
        if params is None:
            if not hasattr(self, "params_fit") or self.params_fit is None:
                raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
            params = self.params_fit

        intensity_scale = 1.0
        if scaled and hasattr(self, "peak_intensity") and self.peak_intensity is not None:
            intensity_scale = float(self.peak_intensity)

        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        return [
            {
                "Peak": r.peak,
                "Position(eV)": r.centre,
                "FWHM(eV)": r.fwhm,
                "Scale": r.scale,
                "Amp": r.amp,
                "PeakHeight_norm": r.peak_height_norm,
                "PeakHeight": r.peak_height,
            }
            for r in rows
        ]


    def export_fit(
        self,
        out_path: str,
        *,
        params=None,
        delimiter: str | None = None,
        include_header: bool = True,
        scaled: bool = True,
        headers: bool = True,
    ) -> str:
        """
        Export fitted parameters to CSV or TXT/TSV.

        headers:
            If True, write a metadata header block in TXT/TSV outputs.
            Ignored for CSV.
        """
        if params is None:
            if not hasattr(self, "params_fit") or self.params_fit is None:
                raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
            params = self.params_fit
    
        intensity_scale = 1.0
        if scaled and hasattr(self, "peak_intensity") and self.peak_intensity is not None:
            intensity_scale = float(self.peak_intensity)

        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        # Metadata (parallel to RamanFit)
        meta = {
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_quantity": getattr(self, "x_quantity", None),
            "x_unit": getattr(self, "x_unit", None),

            "background_remove": getattr(self, "background_remove", None),
            "baseline_method": getattr(self, "baseline_method", None),
            "poly_degree": getattr(self, "poly_degree", None),
            "gaussian_sigma": getattr(self, "gaussian_sigma", None),

            "smoothing": getattr(self, "smoothing", None),
            "smooth_window": getattr(self, "smooth_window", None),
            "smooth_order": getattr(self, "smooth_order", None),

            "normalize": getattr(self, "normalize", None),
            "intensity_scale(peak_intensity)": getattr(self, "peak_intensity", None),

            "peak_labels": getattr(self, "peak_labels", None),
            "custom_peaks": "True" if getattr(self, "custom_peaks", None) is not None else "False",
            "remove_peaks": getattr(self, "remove_peaks_list", None),

        }
        meta = {k: v for k, v in meta.items() if v is not None}

        if self.peak_profile == "pvoigt":
            fitted = self.get_fitted_parameters()
            rows_dict = []
            for peak in self.peak_labels:
                d = fitted[peak]
                rows_dict.append({
                    "Peak": peak,
                    "Centre": d["position"],
                    "FWHM": d["fwhm"],
                    "Eta": d.get("eta", ""),
                    "PeakHeight": d["peak_height"],
                })
            return write_table(
                rows_dict,
                out_path,
                fieldnames=["Peak", "Centre", "FWHM", "Eta", "PeakHeight"],
                delimiter=delimiter,
                include_header=include_header,
                meta=meta,
                headers=headers,
                meta_in_csv=False,
            )

        # lorentzian path only:
        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        return write_rows(
            rows,
            out_path,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )

    ### NEW METHOD in v0.2.3 ###
    def export_p0(self):
        """
        Export mapping-ready initial guess vector and ordering metadata.

        Returns
        -------
        dict:
            {
            "p0": np.ndarray,  # normalised-space params
            "peak_order": list[str]
            }
        """
        import numpy as np

        if self.params_fit is None:
            raise ValueError("No fitted parameters found. Run fit_spectrum() first.")

        peak_order = list(self.peak_labels)  # authoritative ordering used in params vector
        return {"p0": np.asarray(self.params_fit, dtype=float).copy(),
                "peak_order": peak_order}
    
    # changed in v0.3.3
    def plot_fit(self, params, offset=0.0, scale=1.0, x_lim=(1.7, 2.2)):
        """Visualise processed spectrum, fitted total curve, and per-peak components.

        Contract:
        - Fitting is ALWAYS performed in peak-normalised space: intensity_normal = processed_spectra / peak_intensity
        - self.normalize controls DISPLAY only:
            * True  -> plot in normalised units
            * False -> plot in processed counts
        - Prints:
            * normalised residual (fit space)
            * per-peak position + (FWHM for Lorentzian, width for pVoigt) + peak height (in display units)
        """
        if params is None:
            raise ValueError("params must be provided (e.g. output from fit_spectrum).")
        
        #TESTING
        print("min step in params:", np.min(np.abs(params - np.round(params, 2))))

        # Preprocessing comparison (your existing feature)
        self._plot_preprocessing_comparison()

        p = np.asarray(params, dtype=float).ravel()
        labels = list(getattr(self, "peak_labels", []))
        if not labels:
            raise RuntimeError("No peak labels available for plotting components.")

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))

        expected = stride * len(labels)
        if p.size < expected:
            raise RuntimeError(
                f"params length {p.size} is insufficient for {len(labels)} peaks with stride={stride} "
                f"(expected >= {expected})."
            )

        # ---- Display scaling: fit-space is normalised; display is either normalised or counts
        if self.normalize:
            data_plot = (self.processed_spectra / self.peak_intensity) * scale + offset
            display_multiplier = 1.0  # keep in normalised units
            y_label = "Intensity (a.u.)"
        else:
            data_plot = self.processed_spectra * scale + offset
            display_multiplier = float(self.peak_intensity)  # convert model from normalised to counts
            y_label = "Intensity (counts)"

        # ---- Total fit in fit space (normalised)
        y_fit_norm = self._model(self.energy, *p)
        y_fit_plot = (y_fit_norm * display_multiplier) * scale + offset

        # ---- Residual in fit space (normalised)
        residual = np.sum((self.intensity_normal - y_fit_norm) ** 2) / np.sum(self.intensity_normal ** 2)
        print(f'Normalized Residual: {residual:.4f} (Perfect fit has R = 0)\n')

        # ---- Plot
        plt.figure()
        plt.plot(self.energy, data_plot, "k-", label="Processed Spectrum")
        plt.plot(self.energy, y_fit_plot, "b--", label="Fitted Total Curve")

        # Components
        try:
            from .peak_models import single_peak
        except Exception:  # pragma: no cover
            from peak_models import single_peak

        # Print per-peak summary header
        if profile == "lorentzian":
            print(f"Per-peak (Lorentzian): position, FWHM, peak height:")
        elif profile == "pvoigt":
            print(f"Per-peak (pseudo-Voigt): position, FWHM, eta, peak height:")
        else:
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in plot_fit().")

        for i, name in enumerate(labels):
            block = p[i * stride : (i + 1) * stride]

            comp_profile = "pvoigt" if profile == "pvoigt" else "lorentzian"
            y_comp_norm = single_peak(self.energy, block, profile=comp_profile)
            y_comp_plot = (y_comp_norm * display_multiplier) * scale + offset

            # Keep legacy colours for Trion/Exciton
            name_l = str(name).lower()
            if name_l == "trion":
                style = "r--"
                label = "Trion"
            elif name_l == "exciton":
                style = "g--"
                label = "Exciton"
            else:
                style = "--"
                label = str(name)

            plt.plot(self.energy, y_comp_plot, style, label=label)

            # Reporting (legacy decimals + rename Amplitude -> Peak height)
            centre = float(block[0])

            if profile == "lorentzian":
                width_hwhm = float(block[1])
                fwhm = 2.0 * width_hwhm

                amp_area = float(block[2])
                height_norm = (amp_area / (np.pi * width_hwhm)) if width_hwhm != 0 else np.nan
                peak_height = float(height_norm * display_multiplier)
                print(f'{label}: {centre:.3f} eV | FWHM: {fwhm:.4f} eV | Peak height: {peak_height:.2f}')
            else:
                fwhm = float(block[1])
                eta = float(block[3])
                peak_height = float(np.max(y_comp_norm) * display_multiplier)
                print(f'{label}: {centre:.2f} eV | FWHM: {fwhm:.2f} eV  | Peak height: {peak_height:.2f} | eta: {eta:.2f}')

        plt.xlabel("Energy (eV)")
        plt.ylabel(y_label)
        plt.xlim(list(x_lim))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # plt.tight_layout()
        plt.show()

    ## Added in build v0.2.7.1
    def _plot_preprocessing_comparison(self):
        """
        Plot raw vs preprocessing outputs on one figure when smoothing/background_remove is enabled.
        """
        do_smooth = self._smoothed_spectra is not None
        do_bg = self._baseline is not None and self._corrected_spectra is not None

        if not (do_smooth or do_bg):
            return

        plt.figure()
        plt.plot(self.energy, self.raw_spectra, label="raw")

        if do_smooth:
            plt.plot(self.energy, self._smoothed_spectra, label="smoothed")

        if do_bg:
            plt.plot(self.energy, self._baseline, label="baseline")
            plt.plot(self.energy, self._corrected_spectra, label="corrected")

        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (counts)")
        plt.title("Preprocessing comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
