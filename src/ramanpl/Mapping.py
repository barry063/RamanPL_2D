"""Module for spectroscopic mapping data analysis and visualization.

Provides classes for loading, processing, and visualizing photoluminescence (PL) 
and Raman mapping data from .wdf and .txt files. Includes peak fitting, spectral 
integration, and 2D heatmap visualization capabilities.

Classes:
    MappingFileLoader: Loads spectroscopic mapping data from files
    MappingImage: Displays optical images from .wdf files
    PLMapping: Analyzes PL data through Lorentzian peak fitting
    PL_Integration: Analyzes PL data through spectral integration
    RamanMapping: Analyzes Raman data through Lorentzian peak fitting
    Raman_Integration: Analyzes Raman data through spectral integration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson
from renishawWiRE import WDFReader
# import os
# import re


class MappingFileLoader:
    """Loader for spectroscopic mapping data from .wdf and .txt files.
    
    Attributes:
        filename (str): Path to input file
        data_format (str): File format ('txt' or 'wdf')
        reader (WDFReader): Renishaw file reader object (for .wdf only)
        X (int): Number of points in X-direction
        Y (int): Number of points in Y-direction
        xdata (ndarray): Spectral axis values
        spectra (ndarray): 3D array of spectra [Y, X, spectral_points]
    """
    def __init__(self, filename):
        """Initialize file loader and detect file format.
        
        Args:
            filename (str): Path to input file (.wdf or .txt)
        
        Raises:
            ValueError: For unsupported file formats
        """
        self.filename = filename
        self.reader = None
        if filename.endswith(".txt"):
            self.data_format = "txt"
            self._load_txt()
        elif filename.endswith(".wdf"):
            self.data_format = "wdf"
            self._load_wdf()
        else:
            raise ValueError("Unsupported file format. Only '.txt' and '.wdf' are supported.")

    def _load_txt(self):
        """Load mapping data from ASCII text file.
        
        Expected format:
        - First row: Headers (skipped)
        - Columns: [X, Y, Wavenumber, Intensity]
        """
        data = np.loadtxt(self.filename, skiprows=1)
        x_coords = np.unique(data[:, 0])
        y_coords = np.unique(data[:, 1])
        self.X = len(x_coords)
        self.Y = len(y_coords)
        points_per_location = np.sum((data[:, 0] == x_coords[0]) & (data[:, 1] == y_coords[0]))
        self.xdata = data[:points_per_location, 2]

        spectra = np.zeros((self.Y, self.X, points_per_location))
        index = 0
        for j in range(self.Y):
            for i in range(self.X):
                spectra[j, i, :] = data[index:index+points_per_location, 3]
                index += points_per_location

        self.spectra = spectra

    def _load_wdf(self):
        """Load mapping data from Renishaw .wdf file using renishawWiRE library."""
        self.reader = WDFReader(self.filename)
        self.X = self.reader.map_shape[0]
        self.Y = self.reader.map_shape[1]
        self.xdata = self.reader.xdata[:]
        spectra = np.zeros((self.Y, self.X, len(self.xdata)))
        for j in range(self.Y):
            for i in range(self.X):
                spectra[j, i, :] = self.reader.spectra[j][i][:]
        self.spectra = spectra


class MappingImage:
    """Displays optical images from .wdf files with mapping region overlay.
    
    Attributes:
        reader (WDFReader): Renishaw file reader object
    """
    def __init__(self, filename):
        """Initialize image viewer for .wdf files.
        
        Args:
            filename (str): Path to .wdf file
            
        Raises:
            ValueError: If non-.wdf file is provided
        """
        if not filename.endswith(".wdf"):
            raise ValueError("MappingImage can only be used with .wdf files.")
        self.reader = WDFReader(filename)

    def show_optical_image(self):
        """Display optical image with mapping area rectangle overlay.
        
        Uses PIL for image handling and matplotlib for visualization.
        """
        from PIL import Image
        import matplotlib.patches as patches

        image = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title("Optical Image with Mapping Area")
        plt.show()


#########################################################################################################################
#################################################### PL Mapping #########################################################
#########################################################################################################################

class PLMapping:
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

    def __init__(self, filename, custom_peaks, data_range=None, step_size=0.3,
                 poly_degree=3, normalize=True, background_remove=True,
                 baseline_method='poly', smoothing=True, smooth_window=11,
                 smooth_poly=3, gaussian_sigma=10):
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
        """

        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.background_remove = background_remove
        self.baseline_method = baseline_method
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.peak_params = list(custom_peaks.keys())

        loader = MappingFileLoader(filename)
        self.X = loader.X
        self.Y = loader.Y
        self.xdata = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None       

        if self.data_range is None:
            self.data_range = (min(self.xdata), max(self.xdata))

        num_peaks = len(self.custom_peaks)
        self.peak_positions = np.zeros((self.Y, self.X, num_peaks))
        self.peak_intensities = np.zeros((self.Y, self.X, num_peaks))
        self.fitted_params = np.zeros((self.Y, self.X, num_peaks * 3))
        
        ### UPDATED INITIALIZATION in v0.2.2 ###
        self.residual_map = np.full((self.Y, self.X), np.nan)
        self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        ### END UPDATED INITIALIZATION ###

    ## NEW METHOD ADDED in v0.2.3 ##
    def _preprocess_single_spectrum(self, xdata, spec):
        """
        Preprocessing for fitting (always normalised):
        1) optional smoothing
        2) optional background removal
        3) normalisation by peak intensity

        Returns
        -------
        y_norm : ndarray
            Peak-normalised spectrum for fitting
        scale : float
            Peak intensity used for scaling back (raw units)
        """
        y = np.asarray(spec, dtype=float)

        # 1) smoothing
        if self.smoothing:
            y = savgol_filter(y, self.smooth_window, self.smooth_poly)

        # 2) background removal
        if self.background_remove:
            y = self.remove_background(xdata, y)

        # 3) peak normalisation (always)
        scale = np.max(y)
        if scale <= 0:
            return None, None

        y_norm = y / scale
        return y_norm, scale
    ## END NEW METHOD ###

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def lorentzian(self, x, *params):
        """Multi-Lorentzian function for curve fitting.
        
        Args:
            x: Spectral axis values
            *params: Fitting parameters (loc, scale, amp) for each peak
            
        Returns:
            Sum of Lorentzian components
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            loc = params[i]
            scale = params[i+1]
            amp = params[i+2]
            result += (scale / ((x - loc)**2 + scale**2)) * amp / np.pi
        return result

    def remove_background(self, xdata, intensity):
        """Remove spectral background using selected method.
        
        Args:
            xdata: Spectral axis values
            intensity: Raw intensity values
            
        Returns:
            Background-subtracted intensity
            
        Raises:
            ValueError: For invalid baseline methods
        """
        if self.baseline_method == 'poly':
            coeffs = Polynomial.fit(xdata, intensity, self.poly_degree).convert().coef
            background = np.polyval(coeffs[::-1], xdata)
        elif self.baseline_method == 'gaussian':
            background = gaussian_filter1d(intensity, sigma=self.gaussian_sigma)
        else:
            raise ValueError(f"Invalid baseline method: {self.baseline_method}")
        bg_removed = intensity - background
        return bg_removed.clip(min=0)

    ## UPDATED METHOD in v0.2.3 ##
    def fit_spectra(self, initial_p0=None, warm_start=False, reset_on_fail=True, maxfev=6400,  warm_start_rmse_gate=0.06):
        """
        Fit all map spectra using self.custom_peaks as bounds.

        Parameters
        ----------
        initial_p0 : array-like or None
            Optional initial guess vector (e.g., from a single-point PLfit result).
            Must match parameter ordering implied by self.custom_peaks.
        warm_start : bool
            If True, use previous successful fit parameters as p0 for next pixel.
        reset_on_fail : bool
            If True, on fit failure reset p0 to baseline (midpoint/initial_p0).
        maxfev : int
            curve_fit maximum function evaluations.
         warm_start_rmse_gate=0.06 : float
            RMSE threshold for warm start acceptance.

        Returns
        -------
        fitted_parameters : np.ndarray
            Array with shape (Y, X, n_params), NaN where fitting failed.
        """
        if not hasattr(self, "custom_peaks") or not isinstance(self.custom_peaks, dict) or len(self.custom_peaks) == 0:
            raise ValueError("custom_peaks is not set or empty. Provide custom_peaks when initialising PLMapping.")

        # --- Build spectral mask from energy range (eV)
        e_min, e_max = self.data_range
        mask = (self.xdata >= e_min) & (self.xdata <= e_max)

        xdata = self.xdata[mask]

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

        # Optional: seed from single-point PLfit
        if initial_p0 is not None:

            # Accept either:
            #   (a) raw numeric vector p0
            #   (b) dict package: {"p0": <vector>, "peak_order": <list>}
            if isinstance(initial_p0, dict):
                peak_order_pkg = initial_p0.get("peak_order", None)
                p0_vec = initial_p0.get("p0", None)

                if p0_vec is None:
                    raise ValueError("initial_p0 dict must contain key 'p0' with a numeric vector.")

                # Validate ordering contract if provided
                if peak_order_pkg is not None:
                    if [p.lower() for p in peak_order_pkg] != [p.lower() for p in self.peak_params]:
                        raise ValueError(
                            "peak_order mismatch between PLfit and PLMapping.\n"
                            f"PLfit: {list(peak_order_pkg)}\n"
                            f"PLMapping: {list(self.peak_params)}\n"
                            "Ensure both use the same custom_peaks ordering (or pass peak_order explicitly)."
                        )

                initial_p0 = p0_vec  # now reduce to numeric vector

            # Now initial_p0 must be a numeric vector
            initial_p0 = np.asarray(initial_p0, dtype=float)

            if initial_p0.shape != p0_base.shape:
                raise ValueError(
                    f"initial_p0 shape {initial_p0.shape} does not match expected {p0_base.shape}"
                )

            p0_base = np.clip(initial_p0, lower_bound, upper_bound)
        p0_current = p0_base.copy()

        # Output: NaN = fit failed
        fitted_params = np.full((self.Y, self.X, n_params), np.nan)

        for j in range(self.Y):
            for i in range(self.X):
                raw_spec = self.spectra[j, i, :][mask]

                spec_norm, scale = self._preprocess_single_spectrum(xdata, raw_spec)
                self.norm_scale_map[j, i] = scale
                if spec_norm is None:
                    self.residual_map[j, i] = np.nan
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                try:
                    params, _ = optimize.curve_fit(
                        self.lorentzian,
                        xdata,
                        spec_norm,
                        p0=p0_current,
                        bounds=(lower_bound, upper_bound),
                        maxfev=maxfev
                    )
                    
                    # Residual (fit space)
                    model_norm = self.lorentzian(xdata, *params)
                    residual_norm = spec_norm - model_norm
                    rmse_norm = np.sqrt(np.mean(residual_norm ** 2))
                    self.residual_map[j, i] = rmse_norm

                    fitted_params[j, i, :] = params

                    # --- Store peak positions and intensities consistently
                    for k, peak_name in enumerate(self.peak_params):
                        idx = 3 * k
                        center, width, amp = params[idx:idx+3]

                        self.peak_positions[j, i, k] = center

                        peak_height_norm = amp / (np.pi * width)

                        if self.normalize:
                            # display normalised intensity
                            self.peak_intensities[j, i, k] = peak_height_norm
                        else:
                            # display raw-count intensity
                            self.peak_intensities[j, i, k] = peak_height_norm * scale

                    # --- GATED warm-start: only propagate good fits
                    if warm_start:
                        if rmse_norm <= warm_start_rmse_gate:
                            p0_current = params
                        else:
                            # Prevent scanline propagation of a bad local minimum
                            if reset_on_fail:
                                p0_current = p0_base.copy()
                # --- Fit failed
                except RuntimeError:
                    self.residual_map[j, i] = np.nan
                    spec_norm, scale = self._preprocess_single_spectrum(xdata, raw_spec)
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue
        
        n_fit = np.sum(~np.isnan(self.residual_map))
        print(f"Successful fits: {n_fit} / {self.X * self.Y}")

        self.fitted_params = fitted_params
        return fitted_params 
    
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
        e_min, e_max = self.data_range
        mask = (x_full >= e_min) & (x_full <= e_max)
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
        scale = np.max(bg_removed)
        if scale <= 0:
            raise ValueError(f"No positive signal at (X={x}, Y={y}); cannot scale fitted curve.")

        # --- Load fitted parameters (normalised space)
        params = np.asarray(self.fitted_params[y, x, :], dtype=float)
        if np.any(np.isnan(params)):
            raise ValueError(f"Fit parameters are NaN at (X={x}, Y={y}). Fit may have failed.")

        fitted_norm = self.lorentzian(xdata, *params)
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
        """Visualize 2D map of spectral features.
        
        Args:
            data_type: Plot type ('exciton_position', 'trion_position', 
                       'exciton_intensity', 'trion_intensity', 'specific_intensity')
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            specific_xdata: Energy value for 'specific_intensity' plots
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
            
        Raises:
            ValueError: For invalid data types or missing parameters
        """
        if data_type == 'specific_intensity':
            if specific_xdata is None:
                raise ValueError("For 'specific_intensity' data type, the 'specific_xdata' parameter must be provided.")

            data = np.full((self.Y, self.X), np.nan, dtype=float)

            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]

                    if np.any(np.isnan(params)):
                        continue  # fit failed

                    y_norm = self.lorentzian(specific_xdata, *params)

                    if self.normalize:
                        # display normalised model intensity
                        data[j, i] = y_norm
                    else:
                        # display raw model intensity using stored scale
                        scale = getattr(self, "norm_scale_map", None)
                        if scale is None or np.isnan(self.norm_scale_map[j, i]):
                            # If scale was not stored, fall back to NaN rather than misleading numbers
                            continue
                        data[j, i] = y_norm * self.norm_scale_map[j, i]

            label = f'Intensity at {specific_xdata} eV (a.u.)'

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
            label = 'Exciton Intensity (a.u.)'
            
        elif data_type == 'trion_intensity':
            if self.peak_intensities.shape[2] > 1:
                data = self.peak_intensities[:, :, 1]
                label = 'Trion Intensity (a.u.)'
            else:
                raise ValueError("Trion data not available.")
        else:
            raise ValueError("Invalid data_type. Choose from 'exciton_position', 'trion_position', 'exciton_intensity', 'trion_intensity', 'specific_intensity'.")

        # Apply optional range filter (clip outliers to lower bound as you requested)
        if filter_range is not None:
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])

        # Apply optional cropping
        if x_range is not None and y_range is not None:
            x_start, x_end = x_range
            y_start, y_end = y_range
            data = data[y_start:y_end+1, x_start:x_end+1]
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

    ### END UPDATED METHOD ###
    
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
    def __init__(self, filename, integration_range, step_size=0.3, poly_degree=3, background_remove=True):
        """Initialize PL integration analyzer.
        
        Args:
            filename: Path to .wdf file
            integration_range: Spectral range (min, max) in eV
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            background_remove: Enable background subtraction
        """
        self.filename = filename
        self.integration_range = integration_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.background_remove = background_remove

        loader = MappingFileLoader(filename)
        self.X = loader.X
        self.Y = loader.Y
        self.energy = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()


    def remove_background(self, energy, intensity, poly_degree=3):
        """Remove background using polynomial fitting.
        
        Args:
            energy (ndarray): Spectral axis values in eV
            intensity (ndarray): Raw intensity values
            poly_degree (int): Polynomial degree for fitting
            
        Returns:
            ndarray: Background-subtracted intensity
        """
        coeffs = Polynomial.fit(energy, intensity, poly_degree).convert().coef
        background = np.polyval(coeffs[::-1], energy)  # Calculate background signal
        return intensity - background  # Subtract background signal

    def calculate_integration(self):
        """Calculate integrated area under spectra across all map points.
        
        Uses Simpson's rule for integration
        Stores results in integration_area array
        """
        energy = self.energy
        mask = (energy >= self.integration_range[0]) & (energy <= self.integration_range[1])
        energy_subset = energy[mask]

        for j in range(self.Y):
            for i in range(self.X):
                # Get the spectrum data
                spectra = self.spectra[j][i][:]
                spectra_subset = spectra[mask]

                # If background removal is enabled, remove the background signal
                if self.background_remove:
                    spectra_subset = self.remove_background(energy_subset, spectra_subset, self.poly_degree)

                # Calculate the integration area
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
        mask = (energy >= self.integration_range[0]) & (energy <= self.integration_range[1])
        energy_subset = energy[mask]
        spectra_subset = spectra[mask]

        # If background removal is enabled, remove the background signal
        if self.background_remove:
            spectra_bg_removed = self.remove_background(energy_subset, spectra_subset, self.poly_degree)
        else:
            spectra_bg_removed = spectra_subset

        # Plot the original spectrum and background-removed spectrum (if enabled)
        plt.figure(figsize=(10, 6))
        plt.plot(energy_subset, spectra_subset, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(energy_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.show()


########################################################################################################################
#################################################### Raman Mapping #####################################################
########################################################################################################################

class RamanMapping:
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
    def __init__(self, filename, custom_peaks, data_range, step_size=0.3, poly_degree=3,
                 normalize=False, background_remove=True, smoothing=True, baseline_method='poly', smooth_window=11,
                 smooth_poly=3, gaussian_sigma=10):
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
        """
        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.background_remove = background_remove
        self.smoothing = smoothing
        self.baseline_method = baseline_method
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.peak_params = list(custom_peaks.keys())

        loader = MappingFileLoader(filename)
        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        
        # Initialize arrays with dynamic dimensions based on number of peaks
        num_peaks = len(custom_peaks)
        self.peak_positions = np.zeros((self.Y, self.X, num_peaks))
        self.peak_intensities = np.zeros((self.Y, self.X, num_peaks))
        self.fitted_params = np.zeros((self.Y, self.X, num_peaks * 3))
        self.residual_map = np.zeros((self.Y, self.X))
       
        # Initialize ratio and distance arrays
        self.Peaks_distance = np.zeros((self.Y, self.X))
        self.ratio_A1g_E2g = np.zeros((self.Y, self.X))
        self.ratio_E2g_A1g = np.zeros((self.Y, self.X))

        # Data processing parameters
        self.normalize = normalize
        self.background_remove = background_remove
        self.smoothing = smoothing
        self.baseline_method = baseline_method
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma

    def show_optical_image(self):
        """Display optical image with mapping area overlay."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def lorentzian_raman(self, x, *params):
        """Calculate multi-Lorentzian curve for given parameters.
        
        Args:
            x (ndarray): Wavenumber values
            *params: Fitting parameters in sequence [loc1, scale1, amp1, loc2,...]
            
        Returns:
            ndarray: Sum of Lorentzian components
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            loc = params[i]
            scale = params[i+1]
            amp = params[i+2]
            result += (scale / ((x - loc)**2 + scale**2)) * amp / np.pi
        return result

    def remove_background(self, wavenumber, intensity):
        """Remove background using specified method.
        
        Args:
            wavenumber (ndarray): Spectral axis values
            intensity (ndarray): Raw intensity values
            
        Returns:
            ndarray: Background-subtracted intensity
        Raises:
            ValueError: For invalid baseline methods
        """
        if self.baseline_method == 'poly':
            coeffs = Polynomial.fit(wavenumber, intensity, self.poly_degree).convert().coef
            background = np.polyval(coeffs[::-1], wavenumber)
        elif self.baseline_method == 'gaussian':
            background = gaussian_filter1d(intensity, sigma=self.gaussian_sigma)
        else:
            raise ValueError(f"Invalid baseline method: {self.baseline_method}. Choose 'poly' or 'gaussian'.")
        
        bg_removed = intensity - background
        bg_removed = bg_removed.clip(min=0)  # Ensure non-negative values
        return bg_removed

    def fit_spectra(self):
        """Perform spectral fitting across all map points.
        
        Processes data through:
        1. Optional normalization
        2. Background removal
        3. Smoothing
        4. Lorentzian peak fitting
    
        Additional calculations:
        - A1g-E2g peak distances
        - A1g/E2g intensity ratios
        - E2g/A1g intensity ratios
        
        Stores results in:
        - peak_positions: Fitted peak centers
        - peak_intensities: Calculated peak heights
        - residual_map: Fitting quality metrics
        """
        lower_bound = []
        upper_bound = []
        for peak, (low, high) in self.custom_peaks.items():
            lower_bound.extend(low)
            upper_bound.extend(high)
        p0 = [(l + h)/2 for l, h in zip(lower_bound, upper_bound)]

        for j in range(self.Y):
            for i in range(self.X):
                try:
                    # Get raw data
                    wavenumber = self.wavenumber[self.data_range[0]:self.data_range[1]]
                    spectra = self.spectra[j][i][self.data_range[0]:self.data_range[1]]

                    # 1. Normalize if enabled
                    if self.normalize:
                        spectra_min = np.min(spectra)
                        spectra = spectra - spectra_min
                        spectra_max = np.max(spectra)
                        if spectra_max != 0:
                            spectra = spectra / spectra_max

                    # 2. Background removal if enabled
                    if self.background_remove:
                        spectra = self.remove_background(wavenumber, spectra)

                    # 3. Smoothing if enabled
                    if self.smoothing:
                        spectra = savgol_filter(spectra, self.smooth_window, self.smooth_poly)
                    params, _ = optimize.curve_fit(
                        self.lorentzian_raman, wavenumber, spectra,
                        p0=p0, maxfev=6400, bounds=(lower_bound, upper_bound)
                    )

                    # Store parameters and calculate intensities
                    for k, peak in enumerate(self.peak_params):
                        self.peak_positions[j, i, k] = params[k*3]
                        scale = params[k*3+1]
                        amp = params[k*3+2]
                        self.peak_intensities[j, i, k] = amp / (np.pi * scale)

                    # Calculate residual
                    fitted_curve = self.lorentzian_raman(wavenumber, *params)
                    self.residual_map[j, i] = np.sum((spectra - fitted_curve)**2) / np.sum(spectra**2)

                    # Calculate E2g-A1g distance and ratios if present
                    if ('E2g' in self.peak_params) and ('A1g' in self.peak_params):
                        e2g_idx = self.peak_params.index('E2g')
                        a1g_idx = self.peak_params.index('A1g')
                        self.Peaks_distance[j, i] = self.peak_positions[j, i, a1g_idx] - self.peak_positions[j, i, e2g_idx]
                        
                        e2g_int = self.peak_intensities[j, i, e2g_idx]
                        a1g_int = self.peak_intensities[j, i, a1g_idx]
                        
                        self.ratio_A1g_E2g[j, i] = a1g_int / e2g_int if e2g_int != 0 else np.nan
                        self.ratio_E2g_A1g[j, i] = e2g_int / a1g_int if a1g_int != 0 else np.nan

                    self.fitted_params[j, i, :] = params

                except RuntimeError:
                    continue

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

        # Generate data based on data_type
        if data_type == 'specific_intensity':
            data = np.zeros((self.Y, self.X))
            for j in range(self.Y):
                for i in range(self.X):
                    data[j, i] = self.lorentzian_raman(specific_wavenumber, *self.fitted_params[j, i])
            label = f'Intensity at {specific_wavenumber} cm⁻¹'
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
        if ratio_type == 'A1g/E2g':
            if 'A1g' not in self.peak_params or 'E2g' not in self.peak_params:
                raise ValueError("Both 'A1g' and 'E2g' peaks are required for ratio calculation.")
            data = self.ratio_A1g_E2g
            label = 'A1g/E2g Intensity Ratio'
        elif ratio_type == 'E2g/A1g':
            if 'A1g' not in self.peak_params or 'E2g' not in self.peak_params:
                raise ValueError("Both 'A1g' and 'E2g' peaks are required for ratio calculation.")
            data = self.ratio_E2g_A1g
            label = 'E2g/A1g Intensity Ratio'
        else:
            raise ValueError("Invalid ratio_type. Choose from 'A1g/E2g' or 'E2g/A1g'.")

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

    def plot_spectrum_fit(self, x, y):
        """Plot raw data and fitting results for single map point.
        
        Args:
            x (int): X coordinate (0-indexed)
            y (int): Y coordinate (0-indexed)
        Shows:
            - Raw spectrum
            - Estimated background
            - Background-removed data
            - Fitted curve
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates.")

        # Get full spectra and intensity
        full_wavenumber = self.wavenumber[:]
        full_intensity = self.spectra[y][x][:]

        # Apply mask to both wavenumber and intensity
        mask = (full_wavenumber >= self.data_range[0]) & (full_wavenumber <= self.data_range[1])
        wavenumber = full_wavenumber[mask]
        intensity = full_intensity[mask]

        # Process background removal on MASKED data
        if self.background_remove:
            bg_removed_intensity = self.remove_background(wavenumber, intensity)
        else:
            bg_removed_intensity = intensity.copy()

        # Calculate background from MASKED data
        background = intensity - bg_removed_intensity

        # Get fitted parameters and calculate curve
        params = self.fitted_params[y, x, :]
        fitted_curve = self.lorentzian_raman(wavenumber, *params)
        if self.normalize:
            fitted_curve = fitted_curve*max(bg_removed_intensity)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber, intensity, 'k-', label='Raw Spectrum')
        if self.background_remove:
            plt.plot(wavenumber, background, 'r--', label='Estimated Background')
            plt.plot(wavenumber, bg_removed_intensity, 'b-', label='Background Removed')
        plt.plot(wavenumber, fitted_curve, 'g--', label='Fitted Curve')
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum Fit at (X={x}, Y={y})")
        plt.legend()
        plt.show()

    def plot_residual_distribution(self, threshold=None):
        """Visualize spatial distribution of fitting residuals.
        
        Args:
            threshold (float): Highlight residuals above this value
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(self.residual_map, cmap='viridis', origin='upper')
        plt.colorbar(label='Normalized Residual')
        plt.title('Fitting Residual Distribution')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        if threshold > 0:
            mask = self.residual_map > threshold
            plt.imshow(mask, cmap='binary', alpha=0.1, origin='upper')
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
                 poly_degree=3, background_remove=True):
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

        loader = MappingFileLoader(filename)
        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

    def show_optical_image(self):
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def remove_background(self, wavenumber, intensity, poly_degree=3):
        """Remove background using polynomial fitting.
        
        Args:
            wavenumber: Spectral axis in cm⁻¹
            intensity: Raw intensity values
            poly_degree: Polynomial degree for fitting
            
        Returns:
            Background-subtracted intensity (negative values clipped)
        """
        # Use polynomial fitting to remove background signal
        coeffs = Polynomial.fit(wavenumber, intensity, poly_degree).convert().coef
        background = np.polyval(coeffs[::-1], wavenumber)  # Calculate background signal
        bg_removed = intensity - background  # Subtract background signal
        bg_removed[bg_removed < 0] = 0  # Set negative values to zero
        return bg_removed

    def calculate_integration(self):
        """Calculate integrated area using Simpson's rule.
        
        Stores results in integration_area array.
        """
        wavenumber = self.wavenumber[:]
        mask = (wavenumber >= self.integration_range[0]) & (wavenumber <= self.integration_range[1])
        wavenumber_subset = wavenumber[mask]

        for j in range(self.Y):
            for i in range(self.X):
                # Get the spectrum data
                spectra = self.spectra[j][i][:]
                spectra_subset = spectra[mask]

                # If background removal is enabled, remove the background signal
                if self.background_remove:
                    spectra_subset = self.remove_background(wavenumber_subset, spectra_subset, self.poly_degree)

                # Calculate the integration area
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

        # Get data within the integration range
        mask = (wavenumber >= self.integration_range[0]) & (wavenumber <= self.integration_range[1])
        wavenumber_subset = wavenumber[mask]
        spectra_subset = spectra[mask]

        # If background removal is enabled, remove the background signal
        if self.background_remove:
            spectra_bg_removed = self.remove_background(wavenumber_subset, spectra_subset, self.poly_degree)
        else:
            spectra_bg_removed = spectra_subset

        # Plot the original spectrum and background-removed spectrum (if enabled)
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber_subset, spectra_subset, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(wavenumber_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.sh