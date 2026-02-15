"""
A module for  importing Raman data from .wdf and .txt files, analyzing Raman spectra using multi-peak Lorentzian fitting with material-specific configurations.

This module provides tools for preprocessing Raman data (smoothing, background subtraction),
fitting multiple peaks using Lorentzian functions, and visualizing the results. The code works with selected materials in the raman_materials.json library.

Classes:
    RamanFit: Main class for processing, fitting, and visualizing Raman spectra.
    DataImporter: Class for importing Raman data from .wdf and .txt files (single spectrum only)
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json
import os
from ramanpl import BaselineAPI
from ramanpl import DataImporter
from ramanpl.exporter import params_to_rows, write_rows


class DataImporter:
    """
    Compatibility shim: keeps RamanFit.DataImporter.data_import(...) working,
    while delegating to the shared importer.

    Preferred usage going forward:
        from ramanpl.dataImporter import DataImporter
    """
    @staticmethod
    def data_import(filename, readlines=(300, 780), x_range=None):
        from ramanpl.dataImporter import DataImporter as _Shared
        return _Shared.data_import(filename=filename, readlines=readlines, x_range=x_range, axis="wavenumber")

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


class RamanFit:
    """A class for fitting and analyzing Raman spectra using configurable multi-peak Lorentzian models.
    
    Handles material-specific peak configurations, substrate peaks, preprocessing, and visualization.

    Attributes
    ----------
    raw_spectra : ndarray
        Raw Raman intensity values (counts)
    processed_spectra : ndarray
        Processed intensity values after preprocessing
    wavenumber : ndarray
        Raman shift values (cm⁻¹) for the spectrum
    peak_intensity : float
        Maximum intensity value used for normalization
    intensity_normal : ndarray
        Normalized intensity values
    lower_bound : list
        Lower bounds for fitting parameters [loc, scale, amp] for each peak
    upper_bound : list
        Upper bounds for fitting parameters [loc, scale, amp] for each peak
    peak_labels : list
        Names of peaks being fitted
    p0 : list
        Initial parameter guesses (midpoints between bounds)

    Methods
    -------
    load_material_parameters(materials)
        Load peak parameters from material library
    load_substrate(substrate)
        Load substrate peak parameters from library
    update_bounds(**kwargs)
        Modify fitting bounds for specific peaks
    remove_peaks(*peak_names)
        Remove peaks from fitting model
    fit_spectrum()
        Perform curve fitting
    plot_fit(params, **kwargs)
        Visualize fitting results
    """
    def __init__(
        self,
        spectra,
        wavenumber,
        materials=None,
        substrate=None,
        background_remove=False,
        baseline_method="poly",
        poly_degree=3,
        gaussian_sigma=50,
        smoothing=False,
        smooth_window=11,
        smooth_order=3,
        normalize=False,
        custom_peaks=None,
        remove_peaks=None,
        peak_order=None,
        peak_profile: str = "lorentzian"
    ):
        """Initialize RamanFit analyzer with data and processing parameters.

        Parameters
        ----------
        spectra : array-like
            Raw intensity values (y-axis)
        wavenumber : array-like
            Raman shift values (x-axis) in cm⁻¹
        materials : list of str, optional
            Material identifiers from library (e.g., ['WS2', 'WO3'])
        substrate : str, optional
            Substrate identifier from library (e.g., 'Si')
        background_remove : bool, optional
            Enable background subtraction (default: False)
        baseline_method : {'poly', 'gaussian'}, optional
            Background removal method (default: 'poly')
        poly_degree : int, optional
            Polynomial degree for poly background (default: 3)
        gaussian_sigma : int, optional
            Sigma for Gaussian filter (default: 50)
        smoothing : bool, optional
            Enable Savitzky-Golay smoothing (default: False)
        smooth_window : int, optional
            Window size for smoothing filter (default: 11)
        smooth_order : int, optional
            Polynomial order for smoothing (default: 3)
        normalize : bool, optional
            Normalize intensity to maximum value (default: False)

        Raises
        ------
        ValueError
            For unrecognized baseline methods or invalid material/substrate IDs
        """
        
        # ------------------------------
        # Peak model definition (library-driven)
        # ------------------------------
        # Start empty; populate from library to avoid duplicated / inconsistent defaults.
        self.lower_bound = []
        self.upper_bound = []
        self.peak_labels = []

        # ------------------------------
        # Peak profile (Lorentzian vs pseudo-Voigt)
        # ------------------------------
        self.peak_profile = str(peak_profile).lower().strip()
        if self.peak_profile not in ("lorentzian", "pvoigt"):
            raise ValueError("peak_profile must be 'lorentzian' or 'pvoigt'")
        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        # Store user intent for reproducibility/metadata
        self.custom_peaks = custom_peaks
        self.remove_peaks_list = list(remove_peaks) if remove_peaks is not None else []

        # Store identity metadata even if custom_peaks replaces defaults
        self.materials = materials
        self.substrate = substrate

        if custom_peaks is not None:
            # "custom replaces defaults"
            if not isinstance(custom_peaks, dict) or len(custom_peaks) == 0:
                raise ValueError("custom_peaks must be a non-empty dict: {name: ([lb...],[ub...])}")

            if peak_order is None:
                peak_order_eff = list(custom_peaks.keys())
            else:
                peak_order_eff = list(peak_order)
                missing = [k for k in peak_order_eff if k not in custom_peaks]
                if missing:
                    raise ValueError(f"peak_order contains keys not in custom_peaks: {missing}")

            self.peak_labels = list(peak_order_eff)

            expected = self.params_per_peak
            for name in self.peak_labels:
                lb, ub = custom_peaks[name]
                if len(lb) != expected or len(ub) != expected:
                    if expected == 3:
                        raise ValueError(f"Peak '{name}' bounds must be length-3 lists: [centre, width(HWHM), amp(area)]")
                    else:
                        raise ValueError(f"Peak '{name}' bounds must be length-4 lists: [centre, FWHM, amp(area), eta]")
                self.lower_bound += list(lb)
                self.upper_bound += list(ub)

        else:
            # Backwards-compatible library behaviour
            if materials is None:
                materials = ["WS2"]

            self.materials = materials
            self.substrate = substrate

            if materials is not None:
                self.load_material_parameters(materials)
            if substrate is not None:
                self.load_substrate(substrate)

            # Allow deterministic ordering if requested
            self._enforce_peak_order_if_requested(peak_order=peak_order)

        # Initial guess at midpoint of bounds
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

        # Apply removals last (always wins)
        if self.remove_peaks_list:
            self.remove_peaks(*self.remove_peaks_list)

        # ------------------------------
        ### End of v.0.2.9.5 update ###

        # Set initial parameters
        self.p0 = [(low + high) / 2 
                 for low, high in zip(self.lower_bound, self.upper_bound)]
            
        # Initialise data loaded                                                                                                                               
        self.raw_spectra = np.array(spectra)
        self.wavenumber = np.array(wavenumber)
        self.processed_spectra = np.array(spectra.copy())
        
        # Added in build v0.2.7.1
        self._smoothed_spectra = None
        self._baseline = None
        self._corrected_spectra = None

        # Added in v0.2.8
        # --- store preprocessing settings for reproducibility / export metadata ---
        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"

        self.materials = materials
        self.substrate = substrate

        self.background_remove = background_remove
        self.baseline_method = baseline_method
        self.poly_degree = poly_degree
        self.gaussian_sigma = gaussian_sigma

        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order

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
                x=self.wavenumber,
                y=self.processed_spectra,
                method=method,
                clip_nonnegative=True,
                **bkwargs,
            )

            # --- store intermediates for comparison plotting ---
            self._baseline = np.asarray(result.baseline, dtype=float).ravel()
            self._corrected_spectra = np.asarray(result.y_corrected, dtype=float).ravel()

            # existing behaviour
            self.processed_spectra = result.y_corrected
        else:
            # If no baseline subtraction but smoothing occurred, corrected == processed.
            if smoothing:
                self._corrected_spectra = self.processed_spectra.copy()
        
        ### Updated in v.0.2.4 ###
        # Fit is ALWAYS performed in peak-normalised space.
        # normalize controls DISPLAY/OUTPUT scaling only.
        self.normalize = normalize

        self.peak_intensity = np.max(self.processed_spectra)
        if self.peak_intensity <= 0:
            raise ValueError("Peak intensity is non-positive after preprocessing; cannot normalise for fitting.")
        self.intensity_normal = self.processed_spectra / self.peak_intensity
        ### End of v.0.2.4 update ###

    def _get_material_lib_path(self):
        """Get absolute path to raman_materials.json in module directory."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(module_dir, 'raman_materials.json')
    
    def load_material_parameters(self, materials):
        """Load peak parameters from JSON material library.

        Parameters
        ----------
        materials : list of str
            Material identifiers from library (e.g., ['WS2', 'MoS2'])

        Raises
        ------
        ValueError
            If material library file is missing or contains invalid data
        """
        json_path = self._get_material_lib_path()
        try:
            with open(json_path, 'r') as f:
                material_lib = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Material library file not found at: {json_path}")
                
        for material in materials:
            if material not in material_lib:
                raise ValueError(f"Material '{material}' not found in library")
            
            ### Updated in v.0.2.4 ###
            params = material_lib[material]['peaks']
            lb = params['lower_bound']
            ub = params['upper_bound']
            labels = params['peak_labels']

            # Updated in v0.3.3
            # Append peaks but avoid duplicated peak labels (prevents WS2 double-loading etc.)
            for k, name in enumerate(labels):
                if name in self.peak_labels:
                    continue

                # library is Lorentzian-style: [centre, HWHM, area_amp]
                lb3 = lb[3*k:3*k+3]
                ub3 = ub[3*k:3*k+3]

                self.peak_labels.append(name)

                if self.peak_profile == "lorentzian":
                    self.lower_bound.extend(lb3)
                    self.upper_bound.extend(ub3)
                else:
                    # pVoigt expects width = FWHM. Convert Lorentzian HWHM -> FWHM by *2.
                    c_lb, hwhm_lb, a_lb = lb3
                    c_ub, hwhm_ub, a_ub = ub3

                    self.lower_bound.extend([c_lb, 2.0*hwhm_lb, a_lb, 0.01])
                    self.upper_bound.extend([c_ub, 2.0*hwhm_ub, a_ub, 0.99])

        # Verify parameter consistency
        stride = self.params_per_peak
        if (len(self.upper_bound) !=  stride * len(self.peak_labels)):
            raise ValueError("Invalid parameter dimensions after loading peaks (stride mismatch).")

    ### updated in v0.3.3 ###
    def _enforce_peak_order_if_requested(self, peak_order=None):
        """
        Reorder bounds/labels to match a user-provided peak_order (case-insensitive).
        If peak_order is None, do nothing.
        """
        if peak_order is None:
            return

        peak_order = list(peak_order)
        labels_lower = [p.lower() for p in self.peak_labels]
        order_lower = [p.lower() for p in peak_order]

        if sorted(order_lower) != sorted(labels_lower):
            raise ValueError(
                "peak_order must be a permutation of the loaded peak labels.\n"
                f"Loaded: {self.peak_labels}\n"
                f"Requested: {peak_order}"
            )

        stride = self.params_per_peak

        def block(arr, idx):
            return arr[stride*idx:stride*idx+stride]

        new_lb, new_ub, new_labels = [], [], []
        for name_lower in order_lower:
            old_idx = labels_lower.index(name_lower)
            new_labels.append(self.peak_labels[old_idx])
            new_lb += block(self.lower_bound, old_idx)
            new_ub += block(self.upper_bound, old_idx)

        self.peak_labels = new_labels
        self.lower_bound = new_lb
        self.upper_bound = new_ub
    
    def export_p0(self):
        """
        Export mapping-ready initial guess (normalised fit space) + ordering metadata.

        Returns:
            dict: {"p0": np.ndarray, "peak_order": list[str]}
        """
        import numpy as np
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
        return {"p0": np.asarray(self.params_fit, dtype=float).copy(),
                "peak_order": list(self.peak_labels)}
    
    ### End of v.0.2.4 update ###

    ### Added in v0.2.9 ###
    def get_fitted_spectrum(self):
        """
        Return fitted spectrum on the same x-grid as the input data.

        Returns
        -------
        x : np.ndarray
            Wavenumber axis (cm^-1)
        y_fit : np.ndarray
            Fitted intensity in the SAME units as self.processed_spectra.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("RamanFit has not been fitted yet. Run fit_spectrum() first.")

        x = np.asarray(self.wavenumber, dtype=float).ravel()

        # Fit is performed in normalised space; convert back to processed intensity scale
        y_fit_norm = self._model(x, *self.params_fit)  # model in normalised space
        y_fit = y_fit_norm * float(self.peak_intensity)

        return x.copy(), np.asarray(y_fit, dtype=float).ravel().copy()

    ### Added in v0.2.9 ###
    def get_fitted_parameters(self):
        """
        Return fitted peak parameters as a structured dict.

        Notes
        -----
        Parameter vector layout depends on peak_profile:

        - lorentzian: (loc, HWHM, amp_area) per peak
        - pvoigt:     (loc, FWHM, amp_area, eta) per peak

        Reported peak_height/intensity is the *peak maximum* in processed intensity units
        (i.e. scaled back by peak_intensity when self.normalize=False elsewhere).
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

        # Convert fit-space (normalised) peak height back to processed units.
        # Note: fitting is always normalised, so this is the factor you use everywhere.
        fit_scale = float(self.peak_intensity) if hasattr(self, "peak_intensity") else 1.0

        # For pVoigt peak height, evaluate numerically using the same component model as plot_fit()
        if profile == "pvoigt":
            try:
                from .peak_models import single_peak
            except Exception:  # pragma: no cover
                from peak_models import single_peak

        out = {}

        for i, name in enumerate(self.peak_labels):
            block = p[stride * i : stride * (i + 1)]

            loc = float(block[0])
            width = float(block[1])          # HWHM (lorentzian) or FWHM (pvoigt)
            amp_area = float(block[2])       # area-like amplitude (both profiles)

            if profile == "lorentzian":
                # Historical convention: width is HWHM, so FWHM = 2*HWHM
                fwhm = 2.0 * width
                height_norm = (amp_area / (np.pi * width)) if width != 0 else np.nan

            else:
                # pVoigt convention in your peak_models: width is already FWHM
                fwhm = width
                eta = float(block[3])
                y_comp_norm = single_peak(self.wavenumber, block, profile="pvoigt")
                height_norm = float(np.max(y_comp_norm))

            height_scaled = float(height_norm * fit_scale)

            row = dict(
                position=loc,
                fwhm=float(fwhm),
                amp=float(amp_area),
                height_norm=float(height_norm),
                peak_height=float(height_scaled),   # preferred name
                intensity=float(height_scaled),     # backwards-compatible alias
            )

            # Keep "scale" for Lorentzian compatibility; for pVoigt, store "eta" and optionally "fwhm_param"
            if profile == "lorentzian":
                row["scale"] = float(width)  # HWHM
            else:
                row["eta"] = float(eta)
                row["fwhm_param"] = float(width)  # explicit: pVoigt width parameter is FWHM

            out[name] = row

        return out


    ### Added in v.0.2.8 ###
    def fit_table(self, params=None, *, scaled: bool = True):
        """
        Return per-peak fitted parameters as a list of dicts.

        scaled=True:
            height_scaled is reported in approximate original units by multiplying
            normalised peak height by self.peak_intensity (if available).
            This matches your plotting logic where you scale fitted curves back
            using peak_intensity when not displaying purely normalised output. :contentReference[oaicite:8]{index=8}
        """
        if params is None:
            if not hasattr(self, "params_fit") or self.params_fit is None:
                raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
            params = self.params_fit

        # Best-effort scale factor: if you always fit in normalised space, this should be peak_intensity.
        intensity_scale = 1.0
        if scaled and hasattr(self, "peak_intensity") and self.peak_intensity is not None:
            intensity_scale = float(self.peak_intensity)

        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        # Convert to plain dicts (easy to consume / unit test)
        return [
            {
                "Peak": r.peak,
                "Position(cm^-1)": r.centre,
                "FWHM(cm^-1)": r.fwhm,
                "Scale": r.scale,
                "Amp": r.amp,
                "Height_norm": r.height_norm,
                "Height_scaled": r.height_scaled,
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

        # Build metadata (best-effort: only include fields that exist)
        meta = {
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_quantity": getattr(self, "x_quantity", None),
            "x_unit": getattr(self, "x_unit", None),

            "materials": getattr(self, "materials", None),
            "substrate": getattr(self, "substrate", None),

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

        return write_rows(
            rows,
            out_path,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )
    ### End of v.0.2.8 ###

    # Updated v0.3.3
    def load_substrate(self, substrate):
        """Load substrate parameters from JSON material library.

        Parameters
        ----------
        substrate : str
            Substrate identifier from library (e.g., 'Si', 'SiO2')

        Raises
        ------
        ValueError
            If substrate not found or not marked as substrate in library
        """
        json_path = self._get_material_lib_path()
        try:
            with open(json_path, 'r') as f:
                material_lib = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Material library file not found at: {json_path}")

        if substrate not in material_lib or not material_lib[substrate].get('substrate', False):
            raise ValueError(f"Invalid substrate '{substrate}' or not marked as substrate in library")
        
        params = material_lib[substrate]['peaks']
        lb = params["lower_bound"]
        ub = params["upper_bound"]
        labels = params["peak_labels"]

        for k, name in enumerate(labels):
            if name in self.peak_labels:
                continue

            lb3 = lb[3*k:3*k+3]
            ub3 = ub[3*k:3*k+3]

            self.peak_labels.append(name)

            if self.peak_profile == "lorentzian":
                self.lower_bound.extend(lb3)
                self.upper_bound.extend(ub3)
            else:
                c_lb, hwhm_lb, a_lb = lb3
                c_ub, hwhm_ub, a_ub = ub3
                self.lower_bound.extend([c_lb, 2.0*hwhm_lb, a_lb, 0.01])
                self.upper_bound.extend([c_ub, 2.0*hwhm_ub, a_ub, 0.99])
    
    def update_bounds(self, **kwargs):
        """
        Update fitting constraints for specific peaks.

        kwargs format:
            peak=([lb...], [ub...])

        For lorentzian: [centre, HWHM, amp(area)]
        For pvoigt:     [centre, FWHM, amp(area), eta]
        """
        stride = self.params_per_peak
        labels_lower = [p.lower() for p in self.peak_labels]

        for peak_name, new_bounds in kwargs.items():
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            if not (isinstance(new_bounds, (tuple, list)) and len(new_bounds) == 2):
                raise ValueError(f"Bounds for '{peak_name}' must be (lb, ub).")

            lb_new, ub_new = new_bounds
            if len(lb_new) != stride or len(ub_new) != stride:
                raise ValueError(f"Bounds for '{peak_name}' must be length-{stride} lists (stride={stride}).")

            idx = labels_lower.index(key)
            s = stride * idx

            self.lower_bound[s:s+stride] = list(lb_new)
            self.upper_bound[s:s+stride] = list(ub_new)
            self.p0[s:s+stride] = [(lb_new[i] + ub_new[i]) / 2 for i in range(stride)]
    
    ## updated in v0.3.3
    def remove_peaks(self, *peak_names):
        """Remove peaks from fitting model.

        Parameters
        ----------
        *peak_names : str
            Names of peaks to remove (e.g., 'E12g(Γ)', 'A1g(Γ)')

        Raises
        ------
        ValueError
            If specified peak names are not in current model
        """
        stride = self.params_per_peak
        for peak_name in peak_names:
            labels_lower = [p.lower() for p in self.peak_labels]
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            idx = labels_lower.index(key)
            del self.p0[stride*idx:stride*idx+stride]
            del self.lower_bound[stride*idx:stride*idx+stride]
            del self.upper_bound[stride*idx:stride*idx+stride]
            del self.peak_labels[idx]

    ## Added in v0.3.3 (replace lorentzian_raman)
    def _model(self, x, *params):
        """Internal model dispatcher (Lorentzian or pseudo-Voigt)."""
        try:
            from .peak_models import sum_peaks
        except Exception:  # pragma: no cover
            from peak_models import sum_peaks

        profile = "pvoigt" if self.peak_profile == "pvoigt" else "lorentzian"
        return sum_peaks(np.asarray(x), params, profile=profile, stride=self.params_per_peak)

    # Updated in v0.3.0
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
                    self.wavenumber,
                    self.intensity_normal,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=6400,
                )
            except Exception:
                n_fail += 1
                continue

            y_hat = self._model(self.wavenumber, *params)
            rmse = _rmse(self.intensity_normal, y_hat)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_cov = params_cov
                best_p0 = p0

        if best_params is None:
            raise RuntimeError(
                f"RamanFit.fit_spectrum failed for all starts (n_starts={n_starts}). "
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

    # Method to plot the fitted spectrum along with components
    def plot_fit(self, params, offset=0, scale=1.0, x_lim = [250, 750], y_lim = [],
                 x_ticks = [300, 350, 400, 450, 500, 550, 600, 650, 700]):
        """Visualise fitting results and components (Lorentzian + pseudo-Voigt).

        Notes
        -----
        - Fit space is always normalised: intensity_normal = processed_spectra / peak_intensity
        - self.normalize controls DISPLAY only:
            * True  -> plot in normalised a.u.
            * False -> plot in counts
        - Reported "Peak height" is the peak maximum in DISPLAY units.

        Parameters
        ----------
        params : array-like
            Fitting parameters from fit_spectrum()
        offset : float, optional
            Vertical offset for plotting multiple spectra (default: 0)
        scale : float, optional
            Vertical scaling factor (default: 1.0)
        x_lim : list, optional
            X-axis range [min, max] in cm⁻¹ (default: [250, 750])
        y_lim : list, optional
            Y-axis range [min, max] (default: auto-scale)
        x_ticks : list, optional
            X-axis tick positions in cm⁻¹ (default: 300-700 in 50 cm⁻¹ steps)

        Displays
        --------
        - Raw and processed spectra
        - Fitted curve and individual components
        - Quality metrics in console output
        """
        ## Added in v0.2.7.1 for creating processing comparison
        self._plot_preprocessing_comparison()

        plt.figure()

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))
        p = np.asarray(params, dtype=float).ravel()

        # Display multiplier: convert model (normalised fit space) to display units
        display_multiplier = 1.0 if self.normalize else float(self.peak_intensity)

        # Display-space spectra
        if self.normalize:
            proc_plot = (self.processed_spectra / self.peak_intensity) * scale + offset
            raw_plot = (self.raw_spectra / self.peak_intensity) * scale + offset
            plt.yticks([])
        else:
            proc_plot = self.processed_spectra * scale + offset
            raw_plot = self.raw_spectra * scale + offset

        # Plot spectra
        plt.plot(self.wavenumber, proc_plot, 'k-', label='Processed Spectrum')
        plt.plot(self.wavenumber, raw_plot, 'g-', label='Original Spectrum')

        # Total fitted curve (fit space -> display space)
        y_fit_norm = self._model(self.wavenumber, *p)
        y_fit_plot = (y_fit_norm * display_multiplier) * scale + offset
        plt.plot(self.wavenumber, y_fit_plot, 'b--', label='Fitted Total Curve')

        # Residual in fit space (normalised)
        residual = np.sum((self.intensity_normal - y_fit_norm) ** 2) / np.sum(self.intensity_normal ** 2)

        # Component evaluation helper
        try:
            from .peak_models import single_peak
        except Exception:  # pragma: no cover
            from peak_models import single_peak

        # Print header (style-preserving)
        if profile == "lorentzian":
            print("\n{:<20} {:<15} {:<13} {:<14} {:<10}".format(
                "Peak", "Position(cm⁻¹)", "FWHM(cm⁻¹)", "Peak height", "Scale"
            ))
        elif profile == "pvoigt":
            print("\n{:<20} {:<15} {:<13} {:<14} {:<10} {:<6}".format(
                "Peak", "Position(cm⁻¹)", "FWHM(cm⁻¹)", "Peak height", "Scale", "eta"
            ))
        else:
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in plot_fit().")

        print("-" * 80)

        # Plot components and calculate parameters
        peak_positions = {}

        for i, name in enumerate(self.peak_labels):
            block = p[i * stride:(i + 1) * stride]
            loc = float(block[0])

            # Store positions for special peaks
            peak_positions[str(name)] = loc

            # Component in fit space, then display space
            comp_profile = "pvoigt" if profile == "pvoigt" else "lorentzian"
            y_comp_norm = single_peak(self.wavenumber, block, profile=comp_profile)
            y_comp_plot = (y_comp_norm * display_multiplier) * scale + offset

            # Plot component (keep your legacy red dashed style)
            plt.plot(self.wavenumber, y_comp_plot, 'r--')

            # Peak height (display units)
            peak_height = float(np.max(y_comp_norm) * display_multiplier)

            if profile == "lorentzian":
                # width stored as HWHM in your Lorentzian convention
                scale_param = float(block[1])
                fwhm = 2.0 * scale_param
                amp_area = float(block[2])

                print("{:<20} {:<15.2f} {:<13.2f} {:<14.2f} {:<10.2f}".format(
                    str(name), loc, fwhm, peak_height, scale_param
                ))

            else:
                # pVoigt width stored as FWHM (per your updated peak_models)
                fwhm = float(block[1])
                amp_area = float(block[2])
                eta = float(block[3])

                # "Scale" column: keep something meaningful and stable for users.
                # For pVoigt we print FWHM again as "Scale" to avoid inventing a new parameter name.
                # (If you prefer, relabel the column to "Width" for pVoigt later.)
                print("{:<20} {:<15.2f} {:<13.2f} {:<14.2f} {:<10.2f} {:<6.2f}".format(
                    str(name), loc, fwhm, peak_height, fwhm, eta
                ))

        # E12g-A1g separation
        if 'E12g' in peak_positions and 'A1g' in peak_positions:
            peak_diff = peak_positions['A1g'] - peak_positions['E12g']
            print(f"\nE12g(Γ)-A1g(Γ) separation: {peak_diff:.2f} cm⁻¹")

        # Residual print (match your legacy wording)
        print(f"\nNormalized Residual: {residual:.4f} (0 = perfect fit)")

        # Plot formatting
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)' if self.normalize else 'Intensity (counts)')
        plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)
        plt.xticks(x_ticks)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    # Added in v0.2.7.1
    def _plot_preprocessing_comparison(self):
        """
        Plot raw vs preprocessing outputs on one figure when smoothing/background_remove is enabled.
        """
        do_smooth = self._smoothed_spectra is not None
        do_bg = self._baseline is not None and self._corrected_spectra is not None

        if not (do_smooth or do_bg):
            return  # nothing to compare

        plt.figure()

        # Always show raw
        plt.plot(self.wavenumber, self.raw_spectra, label="raw")

        # Smoothed (only if smoothing=True)
        if do_smooth:
            plt.plot(self.wavenumber, self._smoothed_spectra, label="smoothed")

        # Baseline + corrected (only if background_remove=True)
        if do_bg:
            plt.plot(self.wavenumber, self._baseline, label="baseline")
            plt.plot(self.wavenumber, self._corrected_spectra, label="corrected")

        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity (counts)")
        plt.title("Preprocessing comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
