"""
A module for  importing Raman data from .wdf and .txt files, analyzing Raman spectra using multi-peak Lorentzian fitting with material-specific configurations.

This module provides tools for preprocessing Raman data (smoothing, background subtraction),
fitting multiple peaks using Lorentzian functions, and visualizing the results. The code works with selected materials in the raman_materials.json library.

Classes:
    RamanFit: Main class for processing, fitting, and visualizing Raman spectra.
    DataImporter: Class for importing Raman data from .wdf and .txt files (single spectrum only)
"""
from renishawWiRE import WDFReader
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial
import json
import os


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
    lorentzian_raman(x, *params)
        Lorentzian distribution function for fitting
    fit_spectrum()
        Perform curve fitting
    plot_fit(params, **kwargs)
        Visualize fitting results
    """
    def __init__(self, spectra, wavenumber, materials=None, substrate=None,
                 background_remove=False, baseline_method='poly',
                 poly_degree=3, gaussian_sigma=50, smoothing=False, 
                 smooth_window=11, smooth_order=3, normalize=False):
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
        
        # Initialize default parameters (WS2 core peaks)
        self.lower_bound = [
            353, 0, 0,   # E12g(Γ)
            418, 0, 0,    # A1g(Γ)
        ]
        self.upper_bound = [
            358, 5, 10,   # E12g(Γ)
            424, 5, 10,   # A1g(Γ)
        ]
        self.peak_labels = [
            'E12g(Γ)', 'A1g(Γ)'
        ]

        # Load material parameters if specified
        if materials is not None:
            self.load_material_parameters(materials)
        else:
            # Keep default WS2 parameters but clear them if no materials wanted
            # (Add this if you want empty defaults when materials=None)
            pass

        # Load substrate parameters if specified
        if substrate is not None:
            self.load_substrate(substrate)

        # Set initial parameters
        self.p0 = [(low + high) / 2 
                 for low, high in zip(self.lower_bound, self.upper_bound)]
            
        # Initialise data loaded                                                                                                                               
        self.raw_spectra = np.array(spectra)
        self.wavenumber = np.array(wavenumber)
        self.processed_spectra = np.array(spectra.copy())

        # Apply smoothing
        if smoothing:
            self.processed_spectra = savgol_filter(self.processed_spectra, 
                                                 smooth_window, smooth_order)

        # Apply background subtraction
        if background_remove:
            if baseline_method == 'poly':
                # Polynomial background removal
                coeffs = Polynomial.fit(self.wavenumber, self.processed_spectra, poly_degree).convert().coef
                background = np.polyval(coeffs[::-1], self.wavenumber)  # Reverse coefficients for np.polyval
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
        
        # Clear defaults when materials are specified
        self.lower_bound = []
        self.upper_bound = []
        self.peak_labels = []
        
        for material in materials:
            if material not in material_lib:
                raise ValueError(f"Material '{material}' not found in library")
            
            params = material_lib[material]['peaks']
            self.lower_bound.extend(params['lower_bound'])
            self.upper_bound.extend(params['upper_bound'])
            self.peak_labels.extend(params['peak_labels'])
        
        # Verify parameter consistency
        param_count = len(self.lower_bound)
        if (len(self.upper_bound) != param_count or 
            3 * len(self.peak_labels) != param_count):
            raise ValueError("Invalid parameter dimensions in material library")
    
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
        self.lower_bound.extend(params['lower_bound'])
        self.upper_bound.extend(params['upper_bound'])
        self.peak_labels.extend(params['peak_labels'])
    
    def update_bounds(self, **kwargs):
        """Update fitting constraints for specific peaks.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments in format {peak_name: ([lb1, lb2, lb3], [ub1, ub2, ub3])}
            Example: update_bounds(E12g=([350, 0, 0], [360, 5, 10]))

        Raises
        ------
        ValueError
            For unrecognized peak names or invalid bound formats
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
        for peak_name in peak_names:
            if peak_name not in self.peak_labels:
                raise ValueError(f"Peak '{peak_name}' is not a recognized peak name. Available peaks are: {self.peak_labels}")
            
            idx = self.peak_labels.index(peak_name)

            # Remove the corresponding parameters, bounds, and label
            del self.p0[3 * idx:3 * idx + 3]
            del self.lower_bound[3 * idx:3 * idx + 3]
            del self.upper_bound[3 * idx:3 * idx + 3]
            del self.peak_labels[idx]


    # Lorentzian function to fit each peak
    @staticmethod
    def lorentzian_raman(x, *params):
        """Sum of Lorentzian distributions for Raman peak fitting.

        Parameters
        ----------
        x : array-like
            Raman shift values (cm⁻¹)
        *params : array-like
            Fitting parameters in groups of three per peak:
            (loc, scale, amp) × number_of_peaks

        Returns
        -------
        ndarray
            Sum of Lorentzian components evaluated at x positions
        """
        L = 0
        for i in range(0, len(params), 3):
            loc, scale, amp = params[i:i+3]
            L += (scale / ((x - loc) ** 2 + scale ** 2)) * amp / np.pi
        return L

    # Method to fit the spectrum
    def fit_spectrum(self):
        """Perform curve fitting using configured parameters.

        Returns
        -------
        tuple
            (params, params_cov) - Optimized parameters and covariance matrix

        Notes
        -----
        Uses scipy.optimize.curve_fit with:
        - Maximum 6400 function evaluations
        - Bounds configured from material/substrate parameters
        """
        # Perform curve fitting
        params, params_cov = optimize.curve_fit(
            self.lorentzian_raman, 
            self.wavenumber, 
            self.intensity_normal, 
            p0=self.p0, 
            bounds=(self.lower_bound, self.upper_bound), 
            maxfev=6400
        )
        return params, params_cov

    # Method to plot the fitted spectrum along with components
    def plot_fit(self, params, offset=0, scale=1.0, x_lim = [250, 750], y_lim = [],
                 x_ticks = [300, 350, 400, 450, 500, 550, 600, 650, 700]):
        """Visualize fitting results and components.

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
        plt.figure()

        # Determine scaling factors based on normalization
        data_plot = self.processed_spectra * scale + offset
        if self.normalize:
            fit_scale = 1.0           
            plt.yticks([])
        else:
            fit_scale = self.peak_intensity

        # Plot spectra
        plt.plot(self.wavenumber, data_plot, 'k-', label='Processed Spectrum')
        plt.plot(self.wavenumber, self.raw_spectra * scale + offset, 'g-', label='Original Spectrum')

        # Calculate and plot fitted curves
        y_fit = self.lorentzian_raman(self.wavenumber, *params) * fit_scale
        if self.normalize:
            plt.plot(self.wavenumber, y_fit * self.peak_intensity, 'b--', label='Fitted Total Curve')
        else:
            plt.plot(self.wavenumber, y_fit , 'b--', label='Fitted Total Curve')

        # Print header
        print("\n{:<20} {:<15} {:<13} {:<12} {:<10}".format(
            "Peak", "Position(cm⁻¹)", "FWHM(cm⁻¹)", "Intensity", "Scale"))
        print("-" * 70)

        # Plot components and calculate parameters
        peak_positions = {}
        for i in range(len(self.peak_labels)):
            idx = i * 3
            loc = params[idx]
            scale_param = params[idx+1]
            amp = params[idx+2]
            
            # Calculate peak properties
            fwhm = 2 * scale_param
            amplitude = (amp / (np.pi * scale_param)) * fit_scale
            
            # Store positions for special peaks
            peak_positions[self.peak_labels[i]] = loc
            
            # Print parameters for all peaks
            print("{:<20} {:<15.2f} {:<13.2f} {:<12.2f} {:<10.2f}".format(
                self.peak_labels[i], loc, fwhm, amplitude, scale_param))

            # Plot component
            y_fit_single = (scale_param / ((self.wavenumber - loc)**2 + scale_param**2)) * amp / np.pi * fit_scale
            # plt.plot(self.wavenumber, y_fit_single, 'r--')
            if self.normalize:
                plt.plot(self.wavenumber, y_fit_single * self.peak_intensity, 'r--')
            else:
                plt.plot(self.wavenumber, y_fit_single, 'r--')

        # Calculate and print E12g-A1g difference if both exist
        if 'E12g' in peak_positions and 'A1g' in peak_positions:
            peak_diff = peak_positions['A1g'] - peak_positions['E12g']
            print(f"\nE12g(Γ)-A1g(Γ) separation: {peak_diff:.2f} cm⁻¹")

        # Print residual
        fitted_curve = self.lorentzian_raman(self.wavenumber, *params)
        residual = np.sum((self.intensity_normal - fitted_curve)**2) / np.sum(self.intensity_normal**2)
        print(f"\nNormalized Residual: {residual:.4f} (0 = perfect fit)")

        # Plot formatting
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)' if self.normalize else 'Intensity (counts)')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xticks(x_ticks)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
