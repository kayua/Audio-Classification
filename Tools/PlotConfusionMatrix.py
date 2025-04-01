#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import sys
    import numpy

    import seaborn
    import logging

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt


except ImportError as error:
    print(error)
    sys.exit(-1)


class ConfusionMatrixPlotter:
    """
    A class to plot and save confusion matrices with highly customizable visualizations.

    This class provides a method to plot confusion matrices from a list of dictionaries,
    each containing a confusion matrix and its associated class names. The class supports
    multiple customization options like figure size, colormap, font sizes, axis tick label rotation,
    color bar options, annotation format, and the option to display or save the plot.

    Arguments:
    ---------
        @confusion_matrix_fig_size : tuple, optional
            Size of the figure (default is (5, 5)).
        @confusion_matrix_cmap : str, optional
            Colormap for the confusion matrix (default is 'Blues').
        @confusion_matrix_annot_font_size : int, optional
            Font size for the annotations (default is 10).
        @confusion_matrix_label_font_size : int, optional
            Font size for the axis labels (default is 12).
        @confusion_matrix_title_font_size : int, optional
            Font size for the plot title (default is 14).
        @confusion_matrix_show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @confusion_matrix_colorbar : bool, optional
            Whether to display the color bar (default is True).
        @confusion_matrix_annot_kws : dict, optional
            Additional keyword arguments for annotations (default is None).
        @confusion_matrix_fmt : str, optional
            Format for annotations (default is 'd', for integers).
        @confusion_matrix_rotation : int, optional
            Rotation angle for x and y axis labels (default is 45).

    Example:
    -------
        >>> # python3
        ...     cm_plotter = ConfusionMatrixPlotter()
        ...     cm_list = [
        ...     {
        ...     'confusion_matrix': [[50, 10], [5, 35]],
        ...     'class_names': ['Class A', 'Class B'],
        ...     'title': 'Model 1'
        ...     },
        ...     {
        ...     'confusion_matrix': [[30, 20], [10, 40]],
        ...     'class_names': ['Class A', 'Class B'],
        ...     'title': 'Model 2'
        ...     }
        ...     ]
        >>>     cm_plotter.plot_confusion_matrices(cm_list, file_name_path='confusion_matrices/', show_plot=True)
    """

    def __init__(self, confusion_matrix_figure_size:tuple, confusion_matrix_cmap: str,
                 confusion_matrix_annot_font_size: int, confusion_matrix_label_font_size: int,
                 confusion_matrix_title_font_size: int, confusion_matrix_show_plot: bool,
                 confusion_matrix_colorbar: bool, confusion_matrix_annot_kws: None,
                 confusion_matrix_fmt: str, confusion_matrix_rotation: int):
        """
        Initialize the ConfusionMatrixPlotter with customizable plotting options.

        Parameters
        ----------
        @confusion_matrix_fig_size : tuple, optional
            Size of the figure (default is (5, 5)).
        @confusion_matrix_cmap : str, optional
            Colormap for the confusion matrix (default is 'Blues').
        @confusion_matrix_annot_font_size : int, optional
            Font size for the annotations (default is 10).
        @confusion_matrix_label_font_size : int, optional
            Font size for the axis labels (default is 12).
        @confusion_matrix_title_font_size : int, optional
            Font size for the plot title (default is 14).
        @confusion_matrix_show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @confusion_matrix_colorbar : bool, optional
            Whether to display the color bar (default is True).
        @confusion_matrix_annot_kws : dict, optional
            Additional keyword arguments for annotations (default is None).
        @confusion_matrix_fmt : str, optional
            Format for annotations (default is 'd', for integers).
        @confusion_matrix_rotation : int, optional
            Rotation angle for x and y axis labels (default is 45).
        """

        # Validate figure size
        if not isinstance(confusion_matrix_figure_size, tuple) or len(confusion_matrix_figure_size) != 2:
            raise ValueError("Figure size must be a tuple of two values (width, height).")

        if not all(isinstance(i, (int, float)) and i > 0 for i in confusion_matrix_figure_size):
            raise ValueError("Both width and height in figure size must be positive numbers.")

        # Validate colormap
        if not isinstance(confusion_matrix_cmap, str):
            raise ValueError("Colormap must be a string.")

        try:
            cm.get_cmap(confusion_matrix_cmap)  # Check if the colormap is valid

        except ValueError:
            raise ValueError(f"Invalid colormap name: {confusion_matrix_cmap}. Please provide a valid colormap.")

        # Validate font sizes
        if not isinstance(confusion_matrix_annot_font_size, int) or confusion_matrix_annot_font_size <= 0:
            raise ValueError("Annotation font size must be a positive integer.")

        if not isinstance(confusion_matrix_label_font_size, int) or confusion_matrix_label_font_size <= 0:
            raise ValueError("Label font size must be a positive integer.")

        if not isinstance(confusion_matrix_title_font_size, int) or confusion_matrix_title_font_size <= 0:
            raise ValueError("Title font size must be a positive integer.")

        # Validate whether the plot should be shown
        if not isinstance(confusion_matrix_show_plot, bool):
            raise ValueError("Show plot must be a boolean value.")

        # Validate colorbar option
        if not isinstance(confusion_matrix_colorbar, bool):
            raise ValueError("Colorbar option must be a boolean value.")

        # Validate annotation keyword arguments
        if confusion_matrix_annot_kws is not None and not isinstance(confusion_matrix_annot_kws, dict):
            raise ValueError("Annotation keyword arguments must be a dictionary.")

        # Validate annotation format
        if not isinstance(confusion_matrix_fmt, str):
            raise ValueError("Annotation format must be a string.")

        # Validate rotation angle for axis labels
        if not isinstance(confusion_matrix_rotation, int) or not (0 <= confusion_matrix_rotation <= 90):
            raise ValueError("Rotation angle must be an integer between 0 and 90.")

        # Initialize instance variables with provided or default values
        self._confusion_matrix_figure_size = confusion_matrix_figure_size  # Size of the figure to be plotted
        self._confusion_matrix_cmap = confusion_matrix_cmap  # Color map for the heatmap
        self._confusion_matrix_annot_font_size = confusion_matrix_annot_font_size  # Font size for annotations
        self._confusion_matrix_label_font_size = confusion_matrix_label_font_size  # Font size for axis labels
        self._confusion_matrix_title_font_size = confusion_matrix_title_font_size  # Font size for plot title
        self._confusion_matrix_show_plot = confusion_matrix_show_plot  # Whether to show the plot interactively
        self._confusion_matrix_colorbar = confusion_matrix_colorbar  # Whether to display the color bar
        self._confusion_matrix_annot_kws = confusion_matrix_annot_kws\
            if confusion_matrix_annot_kws is not None else {"size": self._confusion_matrix_annot_font_size}  # Annotation keyword

        self._confusion_matrix_fmt = confusion_matrix_fmt  # Format string for annotations (e.g., integers or floating point)
        self._confusion_matrix_rotation = confusion_matrix_rotation  # Rotation for axis labels (default is 45 degrees)

    @staticmethod
    def _process_confusion_matrix(confusion_matrix_dict):
        """
        Process the confusion matrix dictionary and return necessary data.

        Parameters
        ----------
        confusion_matrix_dict : dict
            A dictionary containing the confusion matrix, class names, and optional title.

        Returns
        -------
        numpy.ndarray
            The confusion matrix.
        list
            The class names.
        str
            The title for the confusion matrix plot.
        """
        # Convert confusion matrix to a NumPy array for easy manipulation
        confusion_matrix = numpy.array(confusion_matrix_dict["confusion_matrix"])
        # Get class names
        class_names = confusion_matrix_dict["class_names"]
        # Get plot title (if not provided, default to "Confusion Matrix")
        title = confusion_matrix_dict.get("title", "Confusion Matrix")
        return confusion_matrix, class_names, title

    def _plot_single_confusion_matrix(self, confusion_matrix, class_names, title, index, file_name_path):
        """
        Plot and save or display a single confusion matrix.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The confusion matrix to be plotted.
        class_names : list
            The class labels for the confusion matrix.
        title : str
            The title of the confusion matrix plot.
        index : int
            The index of the current confusion matrix in the list (for naming purposes).
        file_name_path : str
            The path where the confusion matrix image will be saved.
        """
        # Create a new figure with the specified size
        plt.figure(figsize=self._confusion_matrix_figure_size)

        # Create the heatmap using seaborn's heatmap function
        ax = seaborn.heatmap(
            confusion_matrix,
            annot=True,  # Annotate each cell with its value
            fmt=self._confusion_matrix_fmt,  # Format of the annotation (integer or float)
            cmap=self._confusion_matrix_cmap,  # Color map for the heatmap
            xticklabels=class_names,  # Class names for x-axis
            yticklabels=class_names,  # Class names for y-axis
            annot_kws=self._confusion_matrix_annot_kws,  # Additional annotation options (like font size)
            cbar=self._confusion_matrix_colorbar  # Show or hide the color bar
        )

        # Customize the axis labels and title
        ax.set_xlabel('Predicted Labels', fontsize=self._confusion_matrix_label_font_size)
        ax.set_ylabel('True Labels', fontsize=self._confusion_matrix_label_font_size)
        ax.set_title(title, fontsize=self._confusion_matrix_title_font_size)

        # Rotate the axis labels to avoid overlap
        plt.xticks(rotation=self._confusion_matrix_rotation)
        plt.yticks(rotation=self._confusion_matrix_rotation)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Define the file path to save the plot
        file_path = f"{file_name_path}matrix_{index + 1}.pdf"

        # If show_plot is True, display the plot; otherwise, save the plot as a file
        if self._confusion_matrix_show_plot:
            plt.show()  # Display the plot interactively
            logging.debug(f"Confusion matrix {index + 1} displayed on screen.")
        else:
            plt.savefig(file_path)  # Save the plot to a file
            logging.debug(f"Confusion matrix {index + 1} saved to {file_path}.")

        # Close the plot to free up memory
        plt.close()

    def plot_confusion_matrices(self, confusion_matrix_list, file_name_path):
        """
        Plot confusion matrices for a list of models and save them as images.

        Parameters
        ----------
        confusion_matrix_list : list of dict
            Each dictionary contains:
            - 'confusion_matrix': The confusion matrix as a 2D array.
            - 'class_names': The class labels for the matrix.
            - 'title' (optional): Title for the matrix plot.
        file_name_path : str
            Path to save the confusion matrix images.
        """
        logging.info("Starting confusion matrix plotting process.")

        # Iterate over each confusion matrix in the list
        for index, confusion_matrix_dict in enumerate(confusion_matrix_list):
            try:
                # Process the confusion matrix and extract data (matrix, class names, title)
                confusion_matrix, class_names, title = self._process_confusion_matrix(confusion_matrix_dict)

                logging.info(f"Plotting confusion matrix {index + 1}: {title}")

                # Call the function to plot and save/display the confusion matrix
                self._plot_single_confusion_matrix(confusion_matrix, class_names, title, index, file_name_path)

            except KeyError as e:
                # Handle missing keys in the input dictionary
                logging.error(f"Missing key in confusion matrix dictionary: {e}")
            except Exception as e:
                # Handle any other errors that occur during the plotting process
                logging.error(f"Error occurred while plotting confusion matrix {index + 1}: {e}")
                raise

        logging.info("Confusion matrix plotting process completed.")