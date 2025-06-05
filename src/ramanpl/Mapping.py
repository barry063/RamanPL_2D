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
                 poly_degree=3, normalize=False, background_remove=True,
                 baseline_method='poly', smoothing=True, smooth_window=11,
                 smooth_poly=3, gaussian_sigma=10):
        """Initialize PL mapping analyzer.
        
        Args:
            filename: Path to .wdf PL mapping file
            custom_peaks: Peak definitions with bounds {name: (min_params, max_params)}
            data_range: Spectral range (min, max) in eV (default: full spectrum)
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            normalize: Normalize spectra to [0,1] range
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
        self.residual_map = np.zeros((self.Y, self.X))


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

    def fit_spectra(self):
        """Perform Lorentzian fitting across all map points.
        
        Processing steps:
        1. Normalization (optional)
        2. Background removal (optional)
        3. Smoothing (optional)
        4. Lorentzian peak fitting
        
        Stores results in:
        - peak_positions in eV
        - peak_intensities in count/a.u.
        - fitted_params
        - residual_map
        """
        lower_bound = []
        upper_bound = []
        for _, (low, high) in self.custom_peaks.items():
            lower_bound.extend(low)
            upper_bound.extend(high)
        p0 = [(l + h) / 2 for l, h in zip(lower_bound, upper_bound)]

        mask = (self.xdata >= self.data_range[0]) & (self.xdata <= self.data_range[1])
        xdata = self.xdata[mask]

        for j in range(self.Y):
            for i in range(self.X):
                spec = self.spectra[j, i, :][mask]
                if self.normalize:
                    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
                if self.background_remove:
                    spec = self.remove_background(xdata, spec)
                if self.smoothing:
                    spec = savgol_filter(spec, self.smooth_window, self.smooth_poly)
                try:
                    params, _ = optimize.curve_fit(
                        self.lorentzian, xdata, spec,
                        p0=p0, bounds=(lower_bound, upper_bound), maxfev=6400
                    )
                    for k, _ in enumerate(self.peak_params):
                        self.peak_positions[j, i, k] = params[k*3]
                        scale = params[k*3+1]
                        amp = params[k*3+2]
                        self.peak_intensities[j, i, k] = amp / (np.pi * scale)
                    fitted_curve = self.lorentzian(xdata, *params)
                    self.residual_map[j, i] = np.sum((spec - fitted_curve)**2) / np.sum(spec**2)
                    self.fitted_params[j, i, :] = params
                except RuntimeError:
                    continue
    
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
            data = np.zeros((self.Y, self.X))
            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]
                    data[j, i] = self.lorentzian(specific_xdata, *params)
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

        # Filter data range
        if filter_range is not None:
            # Replace outliers with filter_range[0] instead of NaN
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])

        # Set up colormap with explicit 'bad' color for masked values
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad('gray')  # Set masked values to gray instead of white

        # Calculate actual length range (existing code remains)
        if x_range is not None and y_range is not None:
            x_start, x_end = x_range
            y_start, y_end = y_range
            masked_data = masked_data[y_start:y_end+1, x_start:x_end+1]
            x_length = (x_end - x_start + 1) * self.step_size
            y_length = (y_end - y_start + 1) * self.step_size
        else:
            x_length = self.X * self.step_size
            y_length = self.Y * self.step_size

        plt.figure(figsize=(8, 6))
        # Plot with enforced color limits
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
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")
        
        # Get full spectra and intensity
        xdata = self.xdata[:]
        intensity = self.spectra[y][x][:]

        # Apply mask to both wavenumber and intensity
        mask = (xdata >= self.data_range[0]) & (xdata <= self.data_range[1])
        xdata = xdata[mask]
        intensity = intensity[mask]
        raw_intensity = intensity.copy()
        
        # Process background removal on MASKED data
        if self.background_remove:
            bg_removed_intensity = self.remove_background(xdata, intensity)
        else:
            bg_removed_intensity = intensity.copy()

        # Calculate background from MASKED data
        background = intensity - bg_removed_intensity

        # Get fitted parameters and calculate curve
        params = self.fitted_params[y, x, :]
        fitted_curve = self.lorentzian(xdata, *params)
        if self.normalize:
            fitted_curve = fitted_curve*max(bg_removed_intensity)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(xdata, raw_intensity, 'k-', label='Raw Spectrum')
        if self.background_remove:
            plt.plot(xdata, background, 'r--', label='Estimated Background')
            plt.plot(xdata, bg_removed_intensity, 'b-', label='Background Removed')
        plt.plot(xdata, fitted_curve, 'g--', label='Fitted Curve')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum Fit and Background at (X={x}, Y={y})")
        plt.legend()
        plt.show()

    def plot_residual_distribution(self, filter_threshold=None):
        """Visualize spatial distribution of fitting residuals.
        
        Args:
            filter_threshold (float): Highlight residuals above this value
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.residual_map, cmap='viridis', origin='upper')
        plt.colorbar(label='Residual Error')
        plt.title('Residual Distribution')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        if filter_threshold > 0:
            mask = self.residual_map >= filter_threshold
            plt.imshow(mask, cmap='binary', alpha=0.1, origin='upper')
        plt.show()


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
        plt.show()