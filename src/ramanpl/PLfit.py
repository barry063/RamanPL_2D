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
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial


class DataImporter:
    """Class for importing Raman data from .wdf and .txt files (single spectrum only)"""
    @staticmethod
    def data_import(filename, readlines=[300, 780]):
        """Import Raman data from single spectrum files
        
        Parameters
        ----------
        filename : str
            Path to .wdf or .txt file
        readlines : list, optional
            Data range indices [start, end] to select subset of data points
            
        Returns
        -------
        tuple
            (spectra, xdata) as numpy arrays
            
        Raises
        ------
        RuntimeError
            If file cannot be imported
        ValueError
            For unsupported formats or invalid data
        """
        try:
            if filename.lower().endswith('.wdf'):
                
                # Handle WDF files
                reader = WDFReader(filename)               
                spectra = reader.spectra
                xdata = reader.xdata

            elif filename.lower().endswith('.txt'):
                # Handle text files with numpy
                data = np.loadtxt(filename, delimiter="\t", skiprows=1)
                
                # Validate text file format
                if data.shape[1] != 2:
                    raise ValueError("Text file must contain exactly 2 columns: xdata and intensity")
                
                xdata = data[:, 0]
                spectra = data[:, 1]

            else:
                raise ValueError("Supported formats: .wdf, .txt")

            # Apply safe range selection
            n_points = len(spectra)
            start = max(0, min(readlines[0], n_points))
            end = max(start, min(readlines[1], n_points))
            
            return spectra[start:end], xdata[start:end]

        except Exception as e:
            raise RuntimeError(f"Failed to import {filename}: {str(e)}")

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
                 smooth_window=11, smooth_order=3, normalize=True):
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
            normalize (bool): Normalize intensity to maximum value (default: True)

        Raises:
            ValueError: If invalid baseline method is specified
        """
        self.raw_spectra = np.array(spectra)
        self.energy = np.array(energy)
        self.processed_spectra = np.array(spectra.copy())

        # Apply smoothing
        if smoothing:
            self.processed_spectra = savgol_filter(self.processed_spectra, 
                                                 smooth_window, smooth_order)

        # Apply background subtraction
        if background_remove:
            if baseline_method == 'poly':
                # Polynomial background removal
                coeffs = Polynomial.fit(self.energy, self.processed_spectra, poly_degree).convert().coef
                background = np.polyval(coeffs[::-1], self.energy)  # Reverse coefficients for np.polyval
                self.processed_spectra -= background
                
            elif baseline_method == 'gaussian':
                # Gaussian background removal
                background = gaussian_filter1d(self.processed_spectra, sigma=gaussian_sigma)
                self.processed_spectra -= background
            else:
                raise ValueError(f"Baseline method '{baseline_method}' not recognized. Use 'poly' or 'gaussian'.")

        # Normalization
        self.normalize = normalize
        self.peak_intensity = np.max(self.processed_spectra)
        self.intensity_normal = self.processed_spectra / self.peak_intensity

        # Default fitting bounds (Exciton and Trion)
        self.lower_bound = [1.95, 0, 0, 1.8, 0, 0]
        self.upper_bound = [2.1, 0.05, 10, 2.0, 0.2, 10]
        self.peak_labels = ['trion', 'exciton']
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

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
            if peak_name not in self.peak_labels:
                raise ValueError(f"Peak '{peak_name}' is not a recognized peak name. Available peaks are: {self.peak_labels}")

            if not (isinstance(new_bounds, tuple) and len(new_bounds) == 2 and 
                    isinstance(new_bounds[0], list) and isinstance(new_bounds[1], list) and 
                    len(new_bounds[0]) == 3 and len(new_bounds[1]) == 3):
                raise ValueError(f"Bounds for '{peak_name}' must be a tuple of two lists with three elements each.")

            idx = self.peak_labels.index(peak_name)

            # Update the lower and upper bounds for the specified peak
            self.lower_bound[3 * idx:3 * idx + 3] = new_bounds[0]
            self.upper_bound[3 * idx:3 * idx + 3] = new_bounds[1]

            # Update p0 to the midpoint of the new bounds
            self.p0[3 * idx:3 * idx + 3] = [(new_bounds[0][i] + new_bounds[1][i]) / 2 for i in range(3)]

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
    
    
    # Method to fit the spectrum
    def fit_spectrum(self):
        """Perform curve fitting using specified bounds and initial parameters.

        Returns:
            tuple: Contains two elements:
                - params (ndarray): Optimized fitting parameters
                - params_cov (ndarray): Covariance matrix of parameters

        Note:
            Uses scipy.optimize.curve_fit with max 6400 function evaluations
        """
        # Perform curve fitting
        params, params_cov = optimize.curve_fit(self.lorentzian_pl, 
                                                self.energy, self.intensity_normal,
                                                p0=self.p0, maxfev=6400, 
                                                bounds=(self.lower_bound, self.upper_bound))
        return params, params_cov
    

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