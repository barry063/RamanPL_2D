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
from ramanpl.exporter import params_to_rows, write_rows

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
             custom_peaks=None, peak_order=None):
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

        self.custom_peaks = custom_peaks
        self.peak_order = peak_order


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

        # ---- NEW in v0.2.3: allow mapping-consistent bounds/order via custom_peaks
        self.custom_peaks = custom_peaks  # may be None

        if custom_peaks is None:
            # Backwards-compatible defaults (your existing behaviour)
            self.lower_bound = [1.95, 0, 0, 1.8, 0, 0]
            self.upper_bound = [2.1, 0.05, 10, 2.0, 0.2, 10]
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
            for name in self.peak_labels:
                lb, ub = custom_peaks[name]
                if len(lb) != 3 or len(ub) != 3:
                    raise ValueError(f"Peak '{name}' bounds must be length-3 lists: [centre, width, amp]")
                self.lower_bound += list(lb)
                self.upper_bound += list(ub)

        # Initial guess at midpoint of bounds
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

        # ---- NEW in v0.2.3: slots for exporting to mapping
        self.params_fit = None
        self.params_cov = None

    ### UPDATED METHOD in v0.2.3 ###
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
        for peak_name, new_bounds in kwargs.items():
            peak_key = str(peak_name).lower()
            labels_lower = [p.lower() for p in self.peak_labels]

            if peak_key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            idx = labels_lower.index(peak_key)

            # Update the lower and upper bounds for the specified peak
            self.lower_bound[3 * idx:3 * idx + 3] = new_bounds[0]
            self.upper_bound[3 * idx:3 * idx + 3] = new_bounds[1]

            # Update p0 to the midpoint of the new bounds
            self.p0[3 * idx:3 * idx + 3] = [(new_bounds[0][i] + new_bounds[1][i]) / 2 for i in range(3)]

    def fit_spectrum(self):
        """Perform curve fitting using specified bounds and initial parameters.

        Returns:
            tuple: Contains two elements:
                - params (ndarray): Optimized fitting parameters
                - params_cov (ndarray): Covariance matrix of parameters

        Note:
            Uses scipy.optimize.curve_fit with max 6400 function evaluations
        """
        params, params_cov = optimize.curve_fit(
            self.lorentzian_pl,
            self.energy,
            self.intensity_normal,
            p0=self.p0,
            maxfev=6400,
            bounds=(self.lower_bound, self.upper_bound)
        )
        self.params_fit = params
        self.params_cov = params_cov
        return params, params_cov
    ### END UPDATED METHOD ###
    
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

        y_fit_norm = self.lorentzian_pl(x, *self.params_fit)     # normalised fit space
        y_fit = y_fit_norm * float(self.peak_intensity)          # back to processed intensity scale

        return x.copy(), np.asarray(y_fit, dtype=float).ravel().copy()

    ### NEW METHOD in v0.2.9 ###
    def get_fitted_parameters(self):
        """
        Return fitted peak parameters as a structured dict.

        Returns
        -------
        dict
            {
                "<peak_name>": {"position": float, "fwhm": float, "intensity": float},
                ...
            }

        Notes
        -----
        Parameter vector layout is (loc, scale, amp) repeated for each peak
        in the same order as self.peak_labels.
        Intensity reported here is the *peak height* in processed intensity units:
            intensity = (amp / (pi * scale)) * peak_intensity
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("Fit not available. Run fit_spectrum() first.")

        if not hasattr(self, "peak_labels") or not self.peak_labels:
            raise RuntimeError("No peak labels found; cannot map parameters to peaks.")

        p = np.asarray(self.params_fit, dtype=float).ravel()

        expected = 3 * len(self.peak_labels)
        if p.size < expected:
            raise RuntimeError(
                f"params_fit has length {p.size}, but expected at least {expected} "
                f"for {len(self.peak_labels)} peaks."
            )

        out = {}
        fit_scale = float(self.peak_intensity) if hasattr(self, "peak_intensity") else 1.0

        for i, name in enumerate(self.peak_labels):
            idx = 3 * i
            loc = float(p[idx])
            scale = float(p[idx + 1])
            amp = float(p[idx + 2])

            fwhm = 2.0 * scale
            height_norm = (amp / (np.pi * scale)) if scale != 0 else np.nan
            height_scaled = height_norm * fit_scale

            out[name] = {
                "position": loc,
                "fwhm": fwhm,
                "intensity": float(height_scaled),
            }

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


    # Lorentzian function to fit each peak
    @staticmethod
    def lorentzian_pl(x, *params):
        """Sum of Lorentzian distributions for curve fitting.

        Parameters:
            x (array-like): Energy values (x-axis) in eV
            *params: Variable-length parameter list in groups of three:
                loc (float): Peak center position
                scale (float): Lorentzian scale parameter (FWHM = 2*scale)
                amp (float): Peak amplitude

        Returns:
            ndarray: Sum of Lorentzian components evaluated at x positions

        Note:
            Parameter order should alternate between Trion and Exciton parameters:
            [loc1, scale1, amp1, loc2, scale2, amp2]
        """
        L = 0
        for i in range(0, len(params), 3):
            loc, scale, amp = params[i:i+3]
            L += (scale / ((x - loc) ** 2 + scale ** 2)) * amp / np.pi
        return L
    
    def plot_fit(self, params, offset=0, scale=1.0, x_lim=[1.7, 2.2]):
        """Visualize spectrum, fit results, and individual components.

        Parameters:
            params (array-like): Fitting parameters from fit_spectrum
            offset (float): Vertical offset for plotting multiple spectra (default: 0)
            scale (float): Vertical scaling factor (default: 1.0)
            x_lim (list): X-axis limits [min, max] in eV (default: [1.7, 2.2])

        Displays:
            - Original processed spectrum
            - Total fitted curve
            - Individual Trion and Exciton components
            - Quality metrics in console output

        Note:
            Automatically handles unit scaling based on normalization setting
        """
        ## Added in v0.2.7.1
        self._plot_preprocessing_comparison()
        
        plt.figure()

        # Calculate peak amplitudes in original units
        trion_scale = params[1]
        trion_amp = params[2]
        exciton_scale = params[4]
        exciton_amp = params[5]
        
        # Determine scaling factors based on normalization
        data_plot = self.processed_spectra * scale + offset
        if self.normalize:
            fit_scale = 1.0  # Already in original units
            trion_peak = trion_amp / (np.pi * trion_scale)
            exciton_peak = exciton_amp / (np.pi * exciton_scale)
            plt.yticks([])
        else:
            fit_scale = self.peak_intensity
            trion_peak = (trion_amp / (np.pi * trion_scale)) * self.peak_intensity
            exciton_peak = (exciton_amp / (np.pi * exciton_scale)) * self.peak_intensity

        # Plot processed spectrum
        plt.plot(self.energy, data_plot, 'k-', label='Processed Spectrum')

        # Calculate and plot fitted curves
        y_fit = self.lorentzian_pl(self.energy, *params) * fit_scale
        if self.normalize:
            plt.plot(self.energy, y_fit * self.peak_intensity, 'b--', label='Fitted Total Curve')
        else:
            plt.plot(self.energy, y_fit , 'b--', label='Fitted Total Curve')

        # Plot components
        if self.normalize:
            y_fit_trion = (params[1]/((self.energy-params[0])**2+params[1]**2)) * params[2]/np.pi * fit_scale * self.peak_intensity
            y_fit_exciton = (params[4]/((self.energy-params[3])**2+params[4]**2)) * params[5]/np.pi * fit_scale * self.peak_intensity        
        else:
            y_fit_trion = (params[1]/((self.energy-params[0])**2+params[1]**2)) * params[2]/np.pi * fit_scale
            y_fit_exciton = (params[4]/((self.energy-params[3])**2+params[4]**2)) * params[5]/np.pi * fit_scale
        plt.plot(self.energy, y_fit_trion, 'r--', label="Trion")
        plt.plot(self.energy, y_fit_exciton, 'g--', label="Exciton")

        # Calculate normalized residual
        fitted_curve = self.lorentzian_pl(self.energy, *params)
        residual = np.sum((self.intensity_normal - fitted_curve) ** 2) / np.sum(self.intensity_normal ** 2)
        print(f'Normalized Residual: {residual:.4f} (Perfect fit has R = 0)\n')
        
        # Print FWHM and Amplitude of exciton and trion
        print(f'Trion: {params[0]:.2f} eV   | FWHM: {2*trion_scale:.2f} eV  | Amplitude: {trion_peak:.2f}')
        print(f'Exciton: {params[3]:.2f} eV | FWHM: {2*exciton_scale:.2f} eV  | Amplitude: {exciton_peak:.2f}')

        # Plot formatting
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (a.u.)' if self.normalize else 'Intensity (counts)')
        plt.xlim(x_lim)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
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
