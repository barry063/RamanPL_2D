import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks
from scipy.special import wofz


class Compare:
    @classmethod
    def display_PL_plot(cls, *entities, labels=None, peaks=None,
                           title="PL Spectrum", x_label="Energy (eV)", y_label="Intensity (a.u.)",
                           x_lim=[1.5, 2.5], y_distance = 1.2, save=False, save_name=None):
        """
        Plot and compare multiple PL entities on a single diagram.

        Parameters:
        - entities (list, or Dataframe, or Raman): A single entity or a list/tuple of entities.
        - labels (list, optional): Labels for each entity.
        - title (str, optional): Title for the plot.
        - x_label (str, optional): Label for the x-axis.
        - y_label (str, optional): Label 
        for the y-axis.
        - peaks (list, optional): List of x-values where vertical lines will be drawn.
        - x_lim (list, optional): x-axis limits. Default [1.5, 2.5]
        - save (bool, optional): Save image if True
        - save_name (str, optional): filename of the saved image

        Raises:
        - ValueError: If the number of entities and labels don't match.
        """
        if len(entities) < 2:
            entities = tuple(entities[0])
            if len(entities) < 2:
                raise ValueError("At least two entities are required for comparison.")
            
        if labels is None:
            labels = [f"Curve {i + 1}" for i in range(len(entities))]
        elif len(entities) != len(labels):
            raise ValueError("The number of entities and labels must match.")
        
        # Plot the curve on one diagram
        for i, (entity, label) in enumerate(zip(entities, labels)):
            x = entity['Energy']
            y = entity['Intensity']

            # Shift and normalize the curve
            y_normalized = y - i * y_distance

            plt.plot(x, y_normalized, label=label)

        if peaks is not None:
            for peak in peaks:
                plt.axvline(x=peak, color='grey', linestyle='--', linewidth=0.5, 
                            label=f'x={peak:.1f}')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(x_lim)
        plt.yticks([])  # Hide numeric values on the y-axis
        plt.legend(labels=labels, loc='upper left', bbox_to_anchor=(1, 1))  # Show labels on the legend    
        # OPTIONAl: save the image
        if save:
            if save_name == None:
                save_name = title + '.png'
            plt.savefig(save_name)  # use savefig() before show()  
        plt.show()

    @classmethod
    def display_raman_plot(cls, *entities, labels=None, peaks=None,
                           title="Raman Spectrum", x_label="Wavenumber (1/cm)", y_label="Intensity (a.u.)",
                           x_lim=[60, 1800], y_distance = 1.2, save=False, save_name=None):
        """
        Plot and compare multiple Raman entities on a single diagram.

        Parameters:
        - entities (list, or Dataframe, or Raman): A single entity or a list/tuple of entities.
        - labels (list, optional): Labels for each entity.
        - title (str, optional): Title for the plot.
        - x_label (str, optional): Label for the x-axis.
        - y_label (str, optional): Label for the y-axis.
        - peaks (list, optional): List of x-values where vertical lines will be drawn.
        - x_lim (list, optional): x-axis limits. Default [60, 1800]
        - save (bool, optional): Save image if True
        - save_name (str, optional): filename of the saved image

        Raises:
        - ValueError: If the number of entities and labels don't match.
        """
        # take the input argument to be a list if it passes one single entity
        if len(entities)<2:
            entities = tuple(entities[0])
            if len(entities) < 2:
                raise ValueError("At least two entities are required for comparison.")

        if labels is None:
            labels = [f"Curve {i + 1}" for i in range(len(entities))]
        elif len(entities) != len(labels):
            raise ValueError("The number of entities and labels must match.")
        
        # Plot the curve on one diagram
        for i, (entity, label) in enumerate(zip(entities, labels)):
            x = entity['Wavenumber']
            y = entity['Intensity']

            # Shift and normalize the curve
            y_normalized = y - i * y_distance

            plt.plot(x, y_normalized, label=label)

        if peaks is not None:
            for peak in peaks:
                plt.axvline(x=peak, color='grey', linestyle='--', linewidth=0.5, 
                            label=f'x={peak:.1f}')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(x_lim)
        plt.yticks([])  # Hide numeric values on the y-axis
        if labels != None:
            plt.legend(labels=labels, loc='upper left', bbox_to_anchor=(1, 1))  # Show labels on the legend    
        # OPTIONAl: save the image
        if save:
            if save_name == None:
                save_name = title + '.png'
            plt.savefig(save_name)  # use savefig() before show()  
        plt.show()

