#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{1}'
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
    import json
    import logging
    from pathlib import Path

    import numpy
    import seaborn
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

    Features:
        - Increased font sizes by 50% for better readability
        - Save plot data to JSON for reproducibility
        - Load and plot from JSON files
        - Comprehensive data persistence including confusion matrices and metrics

    Arguments:
    ---------
        @confusion_matrix_fig_size : tuple, optional
            Size of the figure (default is (5, 5)).
        @confusion_matrix_cmap : str, optional
            Colormap for the confusion matrix (default is 'Blues').
        @confusion_matrix_annot_font_size : int, optional
            Font size for the annotations. Automatically increased by 50%.
        @confusion_matrix_label_font_size : int, optional
            Font size for the axis labels. Automatically increased by 50%.
        @confusion_matrix_title_font_size : int, optional
            Font size for the plot title. Automatically increased by 50%.
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
        ...     cm_plotter = ConfusionMatrixPlotter(
        ...         confusion_matrix_figure_size=(8, 8),
        ...         confusion_matrix_cmap='Blues',
        ...         confusion_matrix_annot_font_size=10,
        ...         confusion_matrix_label_font_size=12,
        ...         confusion_matrix_title_font_size=14,
        ...         confusion_matrix_show_plot=False,
        ...         confusion_matrix_colorbar=True,
        ...         confusion_matrix_annot_kws=None,
        ...         confusion_matrix_fmt='d',
        ...         confusion_matrix_rotation=45
        ...     )
        ...     cm_list = [
        ...         {
        ...             'confusion_matrix': [[50, 10], [5, 35]],
        ...             'class_names': ['Class A', 'Class B'],
        ...             'title': 'Model 1'
        ...         }
        ...     ]
        ...     cm_plotter.plot_confusion_matrices(cm_list, file_name_path='confusion_matrices/')
        ...
        ...     # Later, reload and plot from JSON
        ...     ConfusionMatrixPlotter.plot_from_json_static('confusion_matrices/matrix_1_data.json', 'new_cm/')
        >>>
    """

    def __init__(self, confusion_matrix_figure_size: tuple, confusion_matrix_cmap: str,
                 confusion_matrix_annot_font_size: int, confusion_matrix_label_font_size: int,
                 confusion_matrix_title_font_size: int, confusion_matrix_show_plot: bool,
                 confusion_matrix_colorbar: bool, confusion_matrix_annot_kws: None,
                 confusion_matrix_fmt: str, confusion_matrix_rotation: int):
        """
        Initialize the ConfusionMatrixPlotter with customizable plotting options.

        NOTE: Font sizes are automatically increased by 50% for better readability.

        Parameters
        ----------
        @confusion_matrix_figure_size : tuple
            Size of the figure.
        @confusion_matrix_cmap : str
            Colormap for the confusion matrix.
        @confusion_matrix_annot_font_size : int
            Font size for the annotations (will be increased by 50%).
        @confusion_matrix_label_font_size : int
            Font size for the axis labels (will be increased by 50%).
        @confusion_matrix_title_font_size : int
            Font size for the plot title (will be increased by 50%).
        @confusion_matrix_show_plot : bool
            Whether to display the plot interactively.
        @confusion_matrix_colorbar : bool
            Whether to display the color bar.
        @confusion_matrix_annot_kws : dict
            Additional keyword arguments for annotations.
        @confusion_matrix_fmt : str
            Format for annotations.
        @confusion_matrix_rotation : int
            Rotation angle for x and y axis labels.
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

        # Store original font sizes for JSON serialization
        self._original_annot_font_size = confusion_matrix_annot_font_size
        self._original_label_font_size = confusion_matrix_label_font_size
        self._original_title_font_size = confusion_matrix_title_font_size

        # Initialize instance variables with provided or default values
        self._confusion_matrix_figure_size = confusion_matrix_figure_size
        self._confusion_matrix_cmap = confusion_matrix_cmap

        # Apply 50% increase to all font sizes
        self._confusion_matrix_annot_font_size = int(confusion_matrix_annot_font_size * 1.5)
        self._confusion_matrix_label_font_size = int(confusion_matrix_label_font_size * 1.5)
        self._confusion_matrix_title_font_size = int(confusion_matrix_title_font_size * 1.5)

        self._confusion_matrix_show_plot = confusion_matrix_show_plot
        self._confusion_matrix_colorbar = confusion_matrix_colorbar
        self._confusion_matrix_annot_kws = confusion_matrix_annot_kws \
            if confusion_matrix_annot_kws is not None else {"size": self._confusion_matrix_annot_font_size}
        self._confusion_matrix_fmt = confusion_matrix_fmt
        self._confusion_matrix_rotation = confusion_matrix_rotation

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
        confusion_matrix = numpy.array(confusion_matrix_dict["confusion_matrix"])
        class_names = confusion_matrix_dict["class_names"]
        title = confusion_matrix_dict.get("title", "Confusion Matrix")
        return confusion_matrix, class_names, title

    @staticmethod
    def _calculate_metrics(confusion_matrix):
        """
        Calculate common classification metrics from confusion matrix.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The confusion matrix.

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, and f1-score per class.
        """
        metrics = {}

        # Overall accuracy
        total = numpy.sum(confusion_matrix)
        correct = numpy.trace(confusion_matrix)
        metrics['accuracy'] = float(correct / total) if total > 0 else 0.0

        # Per-class metrics
        n_classes = confusion_matrix.shape[0]
        metrics['per_class'] = {}

        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = numpy.sum(confusion_matrix[:, i]) - tp
            fn = numpy.sum(confusion_matrix[i, :]) - tp
            tn = total - tp - fp - fn

            # Precision
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            # Recall
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            # F1-score
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            metrics['per_class'][i] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(numpy.sum(confusion_matrix[i, :]))
            }

        # Macro averages
        precisions = [metrics['per_class'][i]['precision'] for i in range(n_classes)]
        recalls = [metrics['per_class'][i]['recall'] for i in range(n_classes)]
        f1_scores = [metrics['per_class'][i]['f1_score'] for i in range(n_classes)]

        metrics['macro_avg'] = {
            'precision': float(numpy.mean(precisions)),
            'recall': float(numpy.mean(recalls)),
            'f1_score': float(numpy.mean(f1_scores))
        }

        return metrics

    def _save_confusion_matrix_to_json(self, confusion_matrix, class_names, title, index, file_name_path):
        """
        Saves confusion matrix data and configuration to a JSON file for reproducibility.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The confusion matrix
        class_names : list
            The class labels
        title : str
            The title of the confusion matrix
        index : int
            Index of the confusion matrix
        file_name_path : str
            The output directory path
        """
        # Calculate metrics
        metrics = self._calculate_metrics(confusion_matrix)

        cm_data = {
            "title": title,
            "matrix_index": index + 1,
            "confusion_matrix": confusion_matrix.tolist(),
            "class_names": class_names,
            "matrix_shape": list(confusion_matrix.shape),
            "metrics": metrics,
            "plot_configuration": {
                "figure_size": list(self._confusion_matrix_figure_size),
                "colormap": self._confusion_matrix_cmap,
                "annot_font_size_original": self._original_annot_font_size,
                "label_font_size_original": self._original_label_font_size,
                "title_font_size_original": self._original_title_font_size,
                "annot_font_size_actual": self._confusion_matrix_annot_font_size,
                "label_font_size_actual": self._confusion_matrix_label_font_size,
                "title_font_size_actual": self._confusion_matrix_title_font_size,
                "show_plot": self._confusion_matrix_show_plot,
                "colorbar": self._confusion_matrix_colorbar,
                "annotation_format": self._confusion_matrix_fmt,
                "rotation": self._confusion_matrix_rotation
            },
            "metadata": {
                "font_increase_applied": "50%",
                "total_predictions": int(numpy.sum(confusion_matrix)),
                "number_of_classes": len(class_names)
            }
        }

        json_file_path = f'{file_name_path}matrix_{index + 1}_data.json'

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(cm_data, json_file, indent=4, ensure_ascii=False)

        logging.info(f"Confusion matrix data saved to JSON: {json_file_path}")

    @classmethod
    def load_from_json(cls, json_file_path):
        """
        Creates a ConfusionMatrixPlotter instance from a JSON file containing plot configuration.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file with plot configuration

        Returns
        -------
        ConfusionMatrixPlotter
            A new instance configured from the JSON file

        Example:
            >>> plotter = ConfusionMatrixPlotter.load_from_json("matrix_1_data.json")
            >>> plotter.plot_confusion_matrices(cm_list, "./output/")
        """
        logging.info(f"Loading confusion matrix configuration from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            cm_data = json.load(json_file)

        config = cm_data['plot_configuration']

        # Create instance using original font sizes (will be increased by 50% automatically)
        instance = cls(
            confusion_matrix_figure_size=tuple(config['figure_size']),
            confusion_matrix_cmap=config['colormap'],
            confusion_matrix_annot_font_size=config['annot_font_size_original'],
            confusion_matrix_label_font_size=config['label_font_size_original'],
            confusion_matrix_title_font_size=config['title_font_size_original'],
            confusion_matrix_show_plot=config['show_plot'],
            confusion_matrix_colorbar=config['colorbar'],
            confusion_matrix_annot_kws=None,
            confusion_matrix_fmt=config['annotation_format'],
            confusion_matrix_rotation=config['rotation']
        )

        logging.info("ConfusionMatrixPlotter instance created from JSON configuration")
        return instance

    def plot_from_json(self, json_file_path, output_path):
        """
        Loads confusion matrix data from a JSON file and generates the plot.

        This is a convenience method that loads both the configuration and data from JSON,
        then generates the plot. Useful for reproducing plots later.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file containing plot data
        output_path : str
            Directory where the plot should be saved

        Example:
            >>> plotter = ConfusionMatrixPlotter(...configuration...)
            >>> plotter.plot_from_json("matrix_1_data.json", "./output/")
        """
        logging.info(f"Loading and plotting confusion matrix from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            cm_data = json.load(json_file)

        # Ensure output path exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Reconstruct the confusion matrix list
        cm_list = [{
            'confusion_matrix': cm_data['confusion_matrix'],
            'class_names': cm_data['class_names'],
            'title': cm_data['title']
        }]

        # Plot using the loaded data (don't save JSON again)
        self.plot_confusion_matrices(cm_list, output_path, save_json=False)

        logging.info(f"Confusion matrix recreated from JSON: {cm_data['title']}")

    @staticmethod
    def plot_from_json_static(json_file_path, output_path):
        """
        Static method to load configuration and data from JSON and plot in one step.

        This method creates a ConfusionMatrixPlotter instance from the JSON configuration,
        then immediately plots the data. Most convenient for one-off plotting from saved data.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file containing plot data and configuration
        output_path : str
            Directory where the plot should be saved

        Example:
            >>> ConfusionMatrixPlotter.plot_from_json_static("matrix_1_data.json", "./output/")
        """
        logging.info(f"Static confusion matrix plotting from JSON: {json_file_path}")

        # Load the instance from JSON
        plotter = ConfusionMatrixPlotter.load_from_json(json_file_path)

        # Plot using the loaded instance
        plotter.plot_from_json(json_file_path, output_path)

        logging.info("Static confusion matrix plot from JSON completed")

    def _plot_single_confusion_matrix(self, confusion_matrix, class_names, title, index, file_name_path):
        """
        Plot and save or display a single confusion matrix with increased font sizes.

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
            annot=True,
            fmt=self._confusion_matrix_fmt,
            cmap=self._confusion_matrix_cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws=self._confusion_matrix_annot_kws,
            cbar=self._confusion_matrix_colorbar
        )

        # Customize the axis labels and title with increased font sizes
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
            plt.show()
            logging.debug(f"Confusion matrix {index + 1} displayed on screen.")
        else:
            plt.savefig(file_path, bbox_inches='tight')
            logging.debug(f"Confusion matrix {index + 1} saved to {file_path}.")

        # Close the plot to free up memory
        plt.close()

    def plot_confusion_matrices(self, confusion_matrix_list, file_name_path, save_json=True):
        """
        Plot confusion matrices for a list of models and save them as images.

        Additionally, all plot data is saved to JSON for reproducibility.

        Parameters
        ----------
        confusion_matrix_list : list of dict
            Each dictionary contains:
            - 'confusion_matrix': The confusion matrix as a 2D array.
            - 'class_names': The class labels for the matrix.
            - 'title' (optional): Title for the matrix plot.
        file_name_path : str
            Path to save the confusion matrix images.
        save_json : bool, optional
            If True, saves plot data to JSON files. Default is True.
        """
        logging.info("Starting confusion matrix plotting process.")

        # Ensure output path exists
        Path(file_name_path).mkdir(parents=True, exist_ok=True)

        # Iterate over each confusion matrix in the list
        for index, confusion_matrix_dict in enumerate(confusion_matrix_list):
            try:
                # Process the confusion matrix and extract data
                confusion_matrix, class_names, title = self._process_confusion_matrix(confusion_matrix_dict)

                logging.info(f"Plotting confusion matrix {index + 1}: {title}")

                # Save to JSON if enabled
                if save_json:
                    self._save_confusion_matrix_to_json(confusion_matrix, class_names, title, index, file_name_path)

                # Call the function to plot and save/display the confusion matrix
                self._plot_single_confusion_matrix(confusion_matrix, class_names, title, index, file_name_path)

            except KeyError as e:
                logging.error(f"Missing key in confusion matrix dictionary: {e}")
            except Exception as e:
                logging.error(f"Error occurred while plotting confusion matrix {index + 1}: {e}")
                raise

        logging.info("Confusion matrix plotting process completed.")
