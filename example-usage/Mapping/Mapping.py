import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from renishawWiRE import WDFReader
from PIL import Image
import matplotlib.patches as patches
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import simps
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


########################################################################################################################
#################################################### Raman Mapping #####################################################
########################################################################################################################

class RamanMapping:
    """A class for analyzing and visualizing Raman mapping data through peak fitting.
    
    Attributes:
        filename (str): Path to .wdf Raman mapping file
        custom_peaks (dict): Dictionary defining peak parameters for fitting
        data_range (tuple): Spectral range to analyze (min_wavenumber, max_wavenumber)
        step_size (float): Physical step size in micrometers
        normalize (bool): Whether to normalize spectra to [0,1] range
        background_remove (bool): Enable/disable background subtraction
        baseline_method (str): Background removal method ('poly' or 'gaussian')
        smooth_window (int): Window size for Savitzky-Golay filter
    """
    def __init__(        
        self, 
        filename, 
        custom_peaks, 
        data_range, 
        step_size=0.3, 
        header=False, 
        save=False, 
        poly_degree=3,
        normalize=False, 
        background_remove=True, 
        smoothing=True, 
        baseline_method='poly', 
        smooth_window=11, 
        smooth_poly=3, 
        gaussian_sigma=10
    ):
        """Initialize Raman mapping analysis parameters.
        
        Args:
            filename (str): Path to .wdf Raman mapping file
            custom_peaks (dict): Peak definitions with format:
                {'PeakName': ([min_loc, min_scale, min_amp], [max_loc, max_scale, max_amp])}
            data_range (tuple): Analysis range (min, max) in cm⁻¹
            step_size (float): Physical step size in micrometers (default: 0.3)
            header (bool): Print file metadata if True (default: False)
            poly_degree (int): Polynomial degree for background removal (default: 3)
            normalize (bool): Normalize spectra to [0,1] range (default: False)
            smooth_window (int): Savitzky-Golay window size (default: 11)
            gaussian_sigma (int): Sigma for Gaussian background (default: 10)
        """
        self.filename = filename
        self.save = save      
        self.reader = WDFReader(filename)
        self.header = header
        if self.header:
            print(self.reader.map_info)

        self.X = self.reader.map_shape[0]
        self.Y = self.reader.map_shape[1]
        self.custom_peaks = custom_peaks
        self.data_range = data_range if data_range else (min(self.reader.xdata), max(self.reader.xdata))
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.peak_params = list(custom_peaks.keys())
        
        # Initialize arrays with dynamic dimensions based on number of peaks
        num_peaks = len(custom_peaks)
        self.peak_positions = np.zeros((self.Y, self.X, num_peaks))
        self.peak_intensities = np.zeros((self.Y, self.X, num_peaks))
        self.fitted_params = np.zeros((self.Y, self.X, num_peaks * 3))
        self.residual_map = np.zeros((self.Y, self.X))  # For residual tracking
        
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

    def show_optical_image(self):
        """Display the optical microscope image with mapping area overlay."""
        img = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
        
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
                    wavenumber = self.reader.xdata[self.data_range[0]:self.data_range[1]]
                    spectra = self.reader.spectra[j][i][self.data_range[0]:self.data_range[1]]

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
            data_type (str): Plot type from:
                'position' - Peak center positions
                'intensity' - Peak intensities  
                'specific_intensity' - Intensity at given wavenumber
                'distance' - A1g-E2g peak spacing
            cmap (str): Matplotlib colormap name (default: 'viridis')
            filter_range (tuple): Data range (min, max) to display
            specific_wavenumber (float): Required for 'specific_intensity' type
            peak_name (str): Required for position/intensity plots
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
        """
        Plot a heatmap of intensity ratios.

        :param ratio_type: Ratio type, options: 'A1g/E2g', 'E2g/A1g'
        :param cmap: Color map
        :param filter_range: Filter range, e.g., [min_value, max_value], only values within this range will be displayed
        :param x_range: X range, e.g., (10, 20), indicating X from 10 to 20
        :param y_range: Y range, e.g., (1, 20), indicating Y from 1 to 20
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
        full_wavenumber = self.reader.xdata[:]
        full_intensity = self.reader.spectra[y][x][:]

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
    """A class for Raman mapping analysis through spectral integration.
    
    Attributes:
        integration_range (tuple): Spectral range for integration (min, max)
        background_remove (bool): Enable/disable background subtraction
        poly_degree (int): Polynomial degree for background removal
    """    
    def __init__(self, filename, integration_range, step_size=0.3, header=False, poly_degree=3, background_remove=True):
        """Initialize integration analysis parameters.
        
        Args:
            filename (str): Path to .wdf Raman mapping file
            integration_range (tuple): Spectral range (min, max) in cm⁻¹
            step_size (float): Physical step size in micrometers (default: 0.3)
            poly_degree (int): Background removal polynomial degree (default: 3)
        """
        self.filename = filename
        self.reader = WDFReader(filename)
        ## Print header information
        self.header = header
        if self.header:
            print(self.reader.map_info)

        self.X = self.reader.map_shape[0]
        self.Y = self.reader.map_shape[1]
        self.integration_range = integration_range  # Integration range
        self.step_size = step_size  # Actual length of each step (unit: um)
        self.poly_degree = poly_degree  # Polynomial fitting degree
        self.background_remove = background_remove  # Whether to perform background removal
        self.integration_area = np.zeros((self.Y, self.X))  # Store integration area

    def show_optical_image(self):
        """Display the optical image."""
        img = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def remove_background(self, wavenumber, intensity, poly_degree=3):
        """
        Remove background signal using polynomial fitting.

        :param wavenumber: Wavenumber axis data
        :param intensity: Intensity data
        :param poly_degree: Polynomial fitting degree
        :return: Intensity data with background removed (negative values set to zero)
        """
        # Use polynomial fitting to remove background signal
        coeffs = Polynomial.fit(wavenumber, intensity, poly_degree).convert().coef
        background = np.polyval(coeffs[::-1], wavenumber)  # Calculate background signal
        bg_removed = intensity - background  # Subtract background signal
        bg_removed[bg_removed < 0] = 0  # Set negative values to zero
        return bg_removed

    def calculate_integration(self):
        """Calculate integrated area under spectra across all map points.
        
        Uses Simpson's rule for integration
        Stores results in integration_area array
        """
        wavenumber = self.reader.xdata[:]
        mask = (wavenumber >= self.integration_range[0]) & (wavenumber <= self.integration_range[1])
        wavenumber_subset = wavenumber[mask]

        for j in range(self.Y):
            for i in range(self.X):
                # Get the spectrum data
                spectra = self.reader.spectra[j][i][:]
                spectra_subset = spectra[mask]

                # If background removal is enabled, remove the background signal
                if self.background_remove:
                    spectra_subset = self.remove_background(wavenumber_subset, spectra_subset, self.poly_degree)

                # Calculate the integration area
                self.integration_area[j, i] = np.abs(simps(spectra_subset, wavenumber_subset))

    def plot_integration_heatmap(self, cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """
        Plot a heatmap of the integration area.

        :param cmap: Color map
        :param filter_range: Filter range, e.g., [min_value, max_value], only values within this range will be displayed
        :param x_range: X range, e.g., (10, 20), indicating X from 10 to 20
        :param y_range: Y range, e.g., (1, 20), indicating Y from 1 to 20
        """
        # Filter data range
        if filter_range is not None:
            # Replace outliers with filter_range[0] instead of NaN
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])
        else:
            data = self.integration_area
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
        """
        Plot the original spectrum and background-removed spectrum (if background removal is enabled) for a specific pixel.

        :param x: X coordinate (step)
        :param y: Y coordinate (step)
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # Get the original spectrum data
        wavenumber = self.reader.xdata[:]
        spectra = self.reader.spectra[y][x][:]

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


#########################################################################################################################
#################################################### PL Mapping #########################################################
#########################################################################################################################

class PLMapping:
    """A class for analyzing and visualizing photoluminescence (PL) mapping data through Lorentzian peak fitting.
    
    Attributes:
        filename (str): Path to .wdf PL mapping file
        custom_peaks (dict): Dictionary defining peak parameters for fitting
        data_range (tuple): Spectral range to analyze (min_energy, max_energy) in eV
        step_size (float): Physical step size in micrometers
        poly_degree (int): Polynomial degree for background removal
        normalize (bool): Whether to normalize spectra to [0,1] range
        background_remove (bool): Enable/disable background subtraction
        baseline_method (str): Background removal method ('poly' or 'gaussian')
    """
    def __init__(self, 
                 filename, 
                 custom_peaks, 
                 data_range=None, 
                 step_size=0.3, 
                 header=False, 
                 save=False, 
                 poly_degree=3, 
                 normalize=False, 
                 background_remove=True, 
                 baseline_method='poly',
                 smoothing=True, 
                 smooth_window=11, 
                 smooth_poly=3,
                 gaussian_sigma=10 
                 ):
        """Initialize PL mapping analysis parameters.
        
        Args:
            filename (str): Path to .wdf PL mapping file
            custom_peaks (dict): Peak definitions with format:
                {'PeakName': ([min_loc, min_scale, min_amp], [max_loc, max_scale, max_amp])}
            data_range (tuple): Analysis range (min, max) in eV (default: full spectrum)
            step_size (float): Physical step size in micrometers (default: 0.3)
            header (bool): Print file metadata if True (default: False)
            poly_degree (int): Polynomial degree for background removal (default: 3)
            normalize (bool): Normalize spectra to [0,1] range (default: False)
            smooth_window (int): Savitzky-Golay window size (default: 11)
            gaussian_sigma (int): Sigma for Gaussian background (default: 10)
        """
        self.filename = filename
        self.save = save      
        self.reader = WDFReader(filename)
        self.header = header
        if self.header:
            print(self.reader.map_info)

        self.X = self.reader.map_shape[0]
        self.Y = self.reader.map_shape[1]
        self.custom_peaks = custom_peaks
        self.data_range = data_range if data_range else (min(self.reader.xdata), max(self.reader.xdata))
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.peak_params = list(custom_peaks.keys())
        
        # Initialize arrays with dynamic dimensions based on number of peaks
        num_peaks = len(custom_peaks)
        self.peak_positions = np.zeros((self.Y, self.X, num_peaks))
        self.peak_intensities = np.zeros((self.Y, self.X, num_peaks))
        self.fitted_params = np.zeros((self.Y, self.X, num_peaks * 3))
        self.residual_map = np.zeros((self.Y, self.X))  # For residual tracking
        
        # Data processing parameters
        self.normalize = normalize
        self.background_remove = background_remove
        self.smoothing = smoothing
        self.baseline_method = baseline_method
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma      
    
    def show_optical_image(self):
        """Display the optical image."""
        img = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def remove_background(self, energy, intensity):
        """Remove background using specified method.
        
        Args:
            energy (ndarray): Spectral axis values in eV
            intensity (ndarray): Raw intensity values
            
        Returns:
            ndarray: Background-subtracted intensity
        Raises:
            ValueError: For invalid baseline methods
        """
        if self.baseline_method == 'poly':
            coeffs = Polynomial.fit(energy, intensity, self.poly_degree).convert().coef
            background = np.polyval(coeffs[::-1], energy)
        elif self.baseline_method == 'gaussian':
            background = gaussian_filter1d(intensity, self.gaussian_sigma)
        else:
            raise ValueError(f"Invalid baseline method: {self.baseline_method}. Choose from 'poly', 'gaussian'.")
        return background

    def lorentzian_pl(self, x, *params):
        """Calculate multi-Lorentzian curve for given parameters.
        
        Args:
            x (ndarray): Energy values in eV
            *params: Fitting parameters in sequence [loc1, scale1, amp1, loc2,...]
            
        Returns:
            ndarray: Sum of Lorentzian components
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            loc = params[i]
            scale = params[i+1]
            amp = params[i+2]
            result += (scale / ((x - loc) ** 2 + scale ** 2)) * amp / np.pi
        return result

    def fit_spectra(self):
        """Perform spectral fitting across all map points.
        
        Processes data through:
        1. Optional normalization
        2. Background removal
        3. Smoothing
        4. Lorentzian peak fitting
        
        Stores results in:
        - peak_positions: Fitted peak centers (eV)
        - peak_intensities: Calculated peak intensities (a.u.)
        - residual_map: Fitting quality metrics (normalized MSE)
        """
        for j in range(self.Y):
            for i in range(self.X):
                try:
                    # Get raw data
                    energy = self.reader.xdata[:]
                    intensity = self.reader.spectra[j][i][:]
                    mask = (energy >= self.data_range[0]) & (energy <= self.data_range[1])
                    energy = energy[mask]
                    intensity = intensity[mask]

                    # 1. Normalize if enabled
                    if self.normalize:
                        spectra_min = np.min(spectra)
                        spectra = spectra - spectra_min
                        spectra_max = np.max(spectra)
                        if spectra_max != 0:
                            spectra = spectra / spectra_max
                    
                    # 2. Background removal if enabled
                    if self.background_remove:
                        background = self.remove_background(energy, intensity)
                        intensity -= background
                        intensity -= min(intensity)

                    # 3. Smoothing if enabled
                    if self.smoothing:
                        intensity = savgol_filter(intensity, self.smooth_window, self.smooth_poly)

                    lower_bound = []
                    upper_bound = []
                    p0 = []
                    for peak, (low, high) in self.custom_peaks.items():
                        # Set initial guess and bounds for each peak
                        lower_bound.extend(low)
                        upper_bound.extend(high)
                        p0.extend([(l + h) / 2 for l, h in zip(low, high)])

                    weights = 1 / (intensity + 0.01)

                    params, _ = optimize.curve_fit(self.lorentzian_pl, energy, intensity, p0=p0, maxfev=6400, bounds=(lower_bound, upper_bound), sigma=weights)

                    # Store peak positions and intensities
                    for k, peak in enumerate(self.peak_params):
                        self.peak_positions[j, i, k] = params[k * 3]
                        # Calculate peak intensity using amp/(π * scale)
                        amp = params[k * 3 + 2]
                        scale = params[k * 3 + 1]
                        self.peak_intensities[j, i, k] = amp / (np.pi * scale)
                    self.fitted_params[j, i, :] = params

                    # Residual check
                    fitted_curve = self.lorentzian_pl(energy, *params)
                    residual = np.sum((intensity - fitted_curve) ** 2) / np.sum(intensity ** 2)
                    self.residual_map[j, i] = residual

                except RuntimeError:
                    continue


    def plot_heatmap(self, data_type='exciton_position', cmap='viridis', 
                     filter_range=None, specific_energy=None,
                     x_range=None, y_range=None):
        """Visualize 2D map of spectral features.
        
        Args:
            data_type (str): Plot type from:
                'exciton_position' - Exciton peak center positions
                'trion_position' - Trion peak center positions
                'exciton_intensity' - Exciton peak intensities
                'trion_intensity' - Trion peak intensities
                'specific_intensity' - Intensity at specified energy
            cmap (str): Matplotlib colormap name (default: 'viridis')
            filter_range (tuple): Data range (min, max) to display
            specific_energy (float): Required for 'specific_intensity' type (eV)
            x_range (tuple): X axis range to display (start, end)
            y_range (tuple): Y axis range to display (start, end)
        """
        if data_type == 'specific_intensity':
            if specific_energy is None:
                raise ValueError("For 'specific_intensity' data type, the 'specific_energy' parameter must be provided.")
            data = np.zeros((self.Y, self.X))
            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]
                    data[j, i] = self.lorentzian_pl(specific_energy, *params)
            label = f'Intensity at {specific_energy} eV (a.u.)'
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
        energy = self.reader.xdata[:]
        intensity = self.reader.spectra[y][x][:]

        # Apply mask to both wavenumber and intensity
        mask = (energy >= self.data_range[0]) & (energy <= self.data_range[1])
        energy = energy[mask]
        intensity = intensity[mask]
        raw_intensity = intensity.copy()
        
        # Process background removal on MASKED data
        if self.background_remove:
            background = self.remove_background(energy, intensity)
            bg_removed_intensity = intensity - background
            bg_removed_intensity -= min(bg_removed_intensity)
        else:
            bg_removed_intensity = intensity.copy()
            background = intensity - bg_removed_intensity

        params = self.fitted_params[y, x, :]
        fitted_curve = self.lorentzian_pl(energy, *params)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(energy, raw_intensity, 'k-', label='Raw Spectrum')
        plt.plot(energy, background, 'r--', label='Estimated Background')
        plt.plot(energy, bg_removed_intensity, 'b-', label='Background Removed')
        plt.plot(energy, fitted_curve, 'g--', label='Fitted Curve')
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
    """A class for PL mapping analysis through spectral integration.
    
    Attributes:
        integration_range (tuple): Spectral range for integration (min_energy, max_energy)
        background_remove (bool): Enable/disable background subtraction
        poly_degree (int): Polynomial degree for background removal
    """
    def __init__(self, filename, integration_range, step_size=0.3, header=False, poly_degree=3, background_remove=True):
        """Initialize integration analysis parameters.
        
        Args:
            filename (str): Path to .wdf PL mapping file
            integration_range (tuple): Spectral range (min, max) in eV
            step_size (float): Physical step size in micrometers (default: 0.3)
            poly_degree (int): Background removal polynomial degree (default: 3)
            background_remove (bool): Enable background subtraction (default: True)
        """
        self.filename = filename
        self.reader = WDFReader(filename)
        self.step_size = step_size  # Actual length of each step (unit: μm)
        self.poly_degree = poly_degree  # Polynomial fitting degree
        self.background_remove = background_remove  # Whether to perform background removal
        self.integration_range = integration_range  # Integration range
        self.header = header
        if self.header:
            print(self.reader.map_info)

        self.X = self.reader.map_shape[0]  # Number of steps in X direction
        self.Y = self.reader.map_shape[1]  # Number of steps in Y direction
        self.integration_area = np.zeros((self.Y, self.X))  # Store integration area

    def show_optical_image(self):
        """Display the optical image."""
        img = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

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
        energy = self.reader.xdata[:]
        mask = (energy >= self.integration_range[0]) & (energy <= self.integration_range[1])
        energy_subset = energy[mask]

        for j in range(self.Y):
            for i in range(self.X):
                # Get the spectrum data
                spectra = self.reader.spectra[j][i][:]
                spectra_subset = spectra[mask]

                # If background removal is enabled, remove the background signal
                if self.background_remove:
                    spectra_subset = self.remove_background(energy_subset, spectra_subset, self.poly_degree)

                # Calculate the integration area
                self.integration_area[j, i] = np.abs(simps(spectra_subset, energy_subset))

    def plot_integration_heatmap(self, cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """Visualize 2D map of integrated intensities.
        
        Args:
            cmap (str): Matplotlib colormap name (default: 'viridis')
            filter_range (tuple): Data range (min, max) to display
            x_range (tuple): X axis range to display (start, end)
            y_range (tuple): Y axis range to display (start, end)
        """
        # Filter data range
        if filter_range is not None:
            # Replace outliers with filter_range[0] instead of NaN
            data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])
        else:
            data = self.integration_area

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
        energy = self.reader.xdata[:]
        spectra = self.reader.spectra[y][x][:]

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