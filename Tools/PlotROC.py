#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
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
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

except ImportError as error:
    print(error)
    sys.exit(-1)


class ROCPlotter:
    """
    A class for plotting ROC curves and calculating AUC for each class based on predicted probabilities.

    This class provides a method to plot the Receiver Operating Characteristic (ROC) curve and calculate
    the Area Under the Curve (AUC) for each class. The method supports plotting for multi-class classification
    by binarizing the ground truth labels.

    Features:
        - Increased font sizes by 50% for better readability
        - Save plot data to JSON for reproducibility
        - Load and plot from JSON files
        - Comprehensive data persistence including ROC curves and AUC values
        - **ENFORCED SEABORN STYLE** for consistent purple/lilac background appearance

    Parameters:
    ----------
        @roc_curve_figure_size : tuple, optional
            Size of the figure (default is (8, 6)).
        @roc_curve_line_width : int, optional
            Line width of the ROC curve (default is 2).
        @roc_curve_marker_style : str, optional
            Style of markers on the ROC curve (default is 'o').
        @roc_curve_cmap : str, optional
            Color map for the ROC curve (default is 'Blues').
        @roc_curve_show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @roc_curve_title_font_size : int, optional
            Font size for the plot title. Automatically increased by 50%.
        @roc_curve_axis_label_font_size : int, optional
            Font size for the axis labels. Automatically increased by 50%.
        @roc_curve_legend_font_size : int, optional
            Font size for the legend. Automatically increased by 50%.
        @roc_curve_grid : bool, optional
            Whether to display the grid (default is True).
        @roc_curve_diagonal_line : bool, optional
            Whether to plot the diagonal line (default is True).

    Example:
    -------
        >>> # python3
        ...     roc_plotter = ROCPlotter(
        ...         roc_curve_figure_size=(10, 8),
        ...         roc_curve_line_width=2,
        ...         roc_curve_marker_style='o',
        ...         roc_curve_cmap='Blues',
        ...         roc_curve_show_plot=False,
        ...         roc_curve_title_font_size=14,
        ...         roc_curve_axis_label_font_size=12,
        ...         roc_curve_legend_font_size=10,
        ...         roc_curve_grid=True,
        ...         roc_curve_diagonal_line=True
        ...     )
        ...     probabilities_predicted = {
        ...         'model_name': 'MyModel',
        ...         'predicted': numpy.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
        ...         'ground_truth': numpy.array([1, 0, 1])
        ...     }
        ...     roc_plotter.plot_roc_curve(probabilities_predicted, file_name_path='roc_curve/')
        ...
        ...     # Later, reload and plot from JSON
        ...     ROCPlotter.plot_from_json_static('roc_curve/ROC_MyModel_data.json', 'new_roc/')
        >>>
    """

    def __init__(self, roc_curve_figure_size: tuple, roc_curve_line_width: int, roc_curve_marker_style: str,
                 roc_curve_cmap: str, roc_curve_show_plot: bool, roc_curve_title_font_size: int,
                 roc_curve_axis_label_font_size: int, roc_curve_legend_font_size: int, roc_curve_grid: bool,
                 roc_curve_diagonal_line: bool):
        """
        Initialize the ROCPlotter with customizable plotting options.

        NOTE: Font sizes are automatically increased by 50% for better readability.
        NOTE: Seaborn darkgrid style is automatically enforced for consistent appearance.

        Parameters
        ----------
        @roc_curve_figure_size : tuple
            Size of the figure.
        @roc_curve_line_width : int
            Line width of the ROC curve.
        @roc_curve_marker_style : str
            Style of markers on the ROC curve.
        @roc_curve_cmap : str
            Color map for the ROC curve.
        @roc_curve_show_plot : bool
            Whether to display the plot interactively.
        @roc_curve_title_font_size : int
            Font size for the plot title (will be increased by 50%).
        @roc_curve_axis_label_font_size : int
            Font size for the axis labels (will be increased by 50%).
        @roc_curve_legend_font_size : int
            Font size for the legend (will be increased by 50%).
        @roc_curve_grid : bool
            Whether to display the grid.
        @roc_curve_diagonal_line : bool
            Whether to plot the diagonal line.
        """

        # Validate figure size
        if not isinstance(roc_curve_figure_size, tuple) or len(roc_curve_figure_size) != 2:
            raise ValueError("roc_curve_figure_size must be a tuple of two values (width, height).")

        if not all(isinstance(i, (int, float)) and i > 0 for i in roc_curve_figure_size):
            raise ValueError("Both width and height in roc_curve_figure_size must be positive numbers.")

        # Validate line width
        if not isinstance(roc_curve_line_width, int) or roc_curve_line_width <= 0:
            raise ValueError("roc_curve_line_width must be a positive integer.")

        # Validate colormap
        if not isinstance(roc_curve_cmap, str) or roc_curve_cmap not in plt.colormaps():
            raise ValueError(f"Invalid colormap. Available colormaps are: {', '.join(plt.colormaps())}.")

        # Validate show plot
        if not isinstance(roc_curve_show_plot, bool):
            raise ValueError("roc_curve_show_plot must be a boolean value.")

        # Validate title font size
        if not isinstance(roc_curve_title_font_size, int) or roc_curve_title_font_size <= 0:
            raise ValueError("roc_curve_title_font_size must be a positive integer.")

        # Validate axis label font size
        if not isinstance(roc_curve_axis_label_font_size, int) or roc_curve_axis_label_font_size <= 0:
            raise ValueError("roc_curve_axis_label_font_size must be a positive integer.")

        # Validate legend font size
        if not isinstance(roc_curve_legend_font_size, int) or roc_curve_legend_font_size <= 0:
            raise ValueError("roc_curve_legend_font_size must be a positive integer.")

        # Validate grid option
        if not isinstance(roc_curve_grid, bool):
            raise ValueError("roc_curve_grid must be a boolean value.")

        # Validate diagonal line option
        if not isinstance(roc_curve_diagonal_line, bool):
            raise ValueError("roc_curve_diagonal_line must be a boolean value.")

        # Store original font sizes for JSON serialization
        self._original_title_font_size = roc_curve_title_font_size
        self._original_axis_label_font_size = roc_curve_axis_label_font_size
        self._original_legend_font_size = roc_curve_legend_font_size

        # Initialize instance variables with provided values
        self.roc_curve_figure_size = roc_curve_figure_size
        self.roc_curve_line_width = roc_curve_line_width
        self.roc_curve_marker_style = roc_curve_marker_style
        self.roc_curve_cmap = roc_curve_cmap
        self.roc_curve_show_plot = roc_curve_show_plot

        # Apply 50% increase to all font sizes
        self.roc_curve_title_font_size = int(roc_curve_title_font_size * 1.5)
        self.roc_curve_axis_label_font_size = int(roc_curve_axis_label_font_size * 1.5)
        self.roc_curve_legend_font_size = int(roc_curve_legend_font_size * 1.5)

        self.roc_curve_grid = roc_curve_grid
        self.roc_curve_diagonal_line = roc_curve_diagonal_line

    @staticmethod
    def _roc_curve_process_data(probabilities_predicted):
        """
        Process the predicted probabilities and ground truth labels.

        Parameters
        ----------
            probabilities_predicted : dict
                Dictionary containing the predicted probabilities and ground truth labels.

        Returns
        -------
        tuple
            A tuple containing the model name, predicted probabilities, and ground truth labels.
        """
        model_name = probabilities_predicted['model_name']
        y_score = probabilities_predicted['predicted']
        y_true = probabilities_predicted['ground_truth']

        return model_name, y_score, y_true

    @staticmethod
    def _roc_curve_binarize_labels(y_true, number_classes):
        """
        Binarize the ground truth labels for multi-class ROC calculation.

        Parameters
        ----------
            y_true : array
                Array of true class labels.
            number_classes : int
                The number of classes for multi-class classification.

        Returns
        -------
        array
            The binarized ground truth labels.
        """
        y_true_bin = label_binarize(y_true, classes=numpy.arange(number_classes))
        return y_true_bin

    @staticmethod
    def _roc_curve_calculate_roc_auc(y_true_bin, y_score):
        """
        Calculate the ROC curve and AUC for each class.

        Parameters
        ----------
            y_true_bin : array
                Binarized ground truth labels.
            y_score : array
                Predicted probabilities for each class.

        Returns
        -------
        dict, dict, dict
            Dictionaries containing the false positive rates, true positive rates, and AUC for each class.
        """
        false_positive_r = {}
        true_positive_r = {}
        roc_auc = {}

        for index in range(y_score.shape[1]):  # Iterate over each class
            false_positive_r[index], true_positive_r[index], _ = roc_curve(y_true_bin[:, index], y_score[:, index])
            roc_auc[index] = auc(false_positive_r[index], true_positive_r[index])

        return false_positive_r, true_positive_r, roc_auc

    def _save_roc_data_to_json(self, model_name, y_score, y_true, false_positive_r, true_positive_r,
                               roc_auc, file_name_path):
        """
        Saves all ROC plot data and configuration to a JSON file for reproducibility.

        Parameters
        ----------
            model_name : str
                The name of the model
            y_score : array
                Predicted probabilities for each class
            y_true : array
                Ground truth labels
            false_positive_r : dict
                False positive rates for each class
            true_positive_r : dict
                True positive rates for each class
            roc_auc : dict
                AUC values for each class
            file_name_path : str
                The output directory path
        """
        # Convert numpy arrays to lists for JSON serialization
        roc_data = {
            "model_name": model_name,
            "data": {
                "predicted_probabilities": y_score.tolist(),
                "ground_truth": y_true.tolist(),
                "number_of_classes": y_score.shape[1],
                "number_of_samples": y_score.shape[0]
            },
            "roc_curves": {
                str(i): {
                    "false_positive_rate": false_positive_r[i].tolist(),
                    "true_positive_rate": true_positive_r[i].tolist(),
                    "auc": float(roc_auc[i])
                }
                for i in range(y_score.shape[1])
            },
            "plot_configuration": {
                "figure_size": list(self.roc_curve_figure_size),
                "line_width": self.roc_curve_line_width,
                "marker_style": self.roc_curve_marker_style,
                "colormap": self.roc_curve_cmap,
                "show_plot": self.roc_curve_show_plot,
                "title_font_size_original": self._original_title_font_size,
                "axis_label_font_size_original": self._original_axis_label_font_size,
                "legend_font_size_original": self._original_legend_font_size,
                "title_font_size_actual": self.roc_curve_title_font_size,
                "axis_label_font_size_actual": self.roc_curve_axis_label_font_size,
                "legend_font_size_actual": self.roc_curve_legend_font_size,
                "grid": self.roc_curve_grid,
                "diagonal_line": self.roc_curve_diagonal_line
            },
            "metadata": {
                "font_increase_applied": "50%",
                "style_applied": "seaborn-v0_8-darkgrid",
                "average_auc": float(numpy.mean([roc_auc[i] for i in range(y_score.shape[1])])),
                "min_auc": float(min([roc_auc[i] for i in range(y_score.shape[1])])),
                "max_auc": float(max([roc_auc[i] for i in range(y_score.shape[1])]))
            }
        }

        json_file_path = f'{file_name_path}ROC_{model_name}_data.json'

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(roc_data, json_file, indent=4, ensure_ascii=False)

        logging.info(f"ROC data saved to JSON: {json_file_path}")

    @classmethod
    def load_from_json(cls, json_file_path):
        """
        Creates a ROCPlotter instance from a JSON file containing plot configuration.

        Parameters
        ----------
            json_file_path : str
                Path to the JSON file with plot configuration

        Returns
        -------
            ROCPlotter
                A new instance configured from the JSON file

        Example:
            >>> plotter = ROCPlotter.load_from_json("ROC_MyModel_data.json")
            >>> plotter.plot_roc_curve(probabilities_predicted, "./output/")
        """
        logging.info(f"Loading ROC plot configuration from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            roc_data = json.load(json_file)

        config = roc_data['plot_configuration']

        # Create instance using original font sizes (will be increased by 50% automatically)
        instance = cls(
            roc_curve_figure_size=tuple(config['figure_size']),
            roc_curve_line_width=config['line_width'],
            roc_curve_marker_style=config['marker_style'],
            roc_curve_cmap=config['colormap'],
            roc_curve_show_plot=config['show_plot'],
            roc_curve_title_font_size=config['title_font_size_original'],
            roc_curve_axis_label_font_size=config['axis_label_font_size_original'],
            roc_curve_legend_font_size=config['legend_font_size_original'],
            roc_curve_grid=config['grid'],
            roc_curve_diagonal_line=config['diagonal_line']
        )

        logging.info("ROCPlotter instance created from JSON configuration")
        return instance

    def plot_from_json(self, json_file_path, output_path):
        """
        Loads ROC plot data from a JSON file and generates the plot.

        This is a convenience method that loads both the configuration and data from JSON,
        then generates the plot. Useful for reproducing plots later.

        Parameters
        ----------
            json_file_path : str
                Path to the JSON file containing plot data
            output_path : str
                Directory where the plot should be saved

        Example:
            >>> plotter = ROCPlotter(...configuration...)
            >>> plotter.plot_from_json("ROC_MyModel_data.json", "./output/")
        """
        logging.info(f"Loading and plotting ROC from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            roc_data = json.load(json_file)

        model_name = roc_data['model_name']
        data = roc_data['data']

        # Ensure output path exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Reconstruct the probabilities_predicted dictionary
        probabilities_predicted = {
            'model_name': model_name,
            'predicted': numpy.array(data['predicted_probabilities']),
            'ground_truth': numpy.array(data['ground_truth'])
        }

        # Plot using the loaded data (don't save JSON again)
        self.plot_roc_curve(probabilities_predicted, output_path, save_json=False)

        logging.info(f"ROC plot recreated from JSON for model '{model_name}'")

    @staticmethod
    def plot_from_json_static(json_file_path, output_path):
        """
        Static method to load configuration and data from JSON and plot in one step.

        This method creates a ROCPlotter instance from the JSON configuration,
        then immediately plots the data. Most convenient for one-off plotting from saved data.

        Parameters
        ----------
            json_file_path : str
                Path to the JSON file containing plot data and configuration
            output_path : str
                Directory where the plot should be saved

        Example:
            >>> ROCPlotter.plot_from_json_static("ROC_MyModel_data.json", "./output/")
        """
        logging.info(f"Static ROC plotting from JSON: {json_file_path}")

        # Load the instance from JSON
        plotter = ROCPlotter.load_from_json(json_file_path)

        # Plot using the loaded instance
        plotter.plot_from_json(json_file_path, output_path)

        logging.info("Static ROC plot from JSON completed")

    def _roc_curve_plot_single(self, fpr, tpr, auc_score, class_index):
        """
        Plot the ROC curve for a single class.

        Parameters
        ----------
        fpr : array
            False positive rate for the class.
        tpr : array
            True positive rate for the class.
        auc_score : float
            AUC for the class.
        class_index : int
            Index of the current class.
        """
        plt.plot(fpr, tpr, label=f'Class {class_index} (AUC = {auc_score:.2f})',
                 linestyle='-', marker=self.roc_curve_marker_style, linewidth=self.roc_curve_line_width)

    def _roc_curve_add_plot_details(self, model_name):
        """
        Add title, labels, legend, and grid to the plot with increased font sizes.

        Parameters
        ----------
        model_name : str
            The name of the model to display in the plot title.
        """
        plt.xlabel('False Positive Rate', fontsize=self.roc_curve_axis_label_font_size)
        plt.ylabel('True Positive Rate', fontsize=self.roc_curve_axis_label_font_size)
        plt.title(f'ROC Curve for {model_name}', fontsize=self.roc_curve_title_font_size)

        if self.roc_curve_grid:
            plt.grid(True)

        plt.legend(loc='lower right', fontsize=self.roc_curve_legend_font_size)

    def _roc_curve_plot_diagonal_line(self):
        """
        Plot the diagonal line for a random classifier (FPR = TPR).
        """
        if self.roc_curve_diagonal_line:
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    def _roc_curve_save_or_show_plot(self, model_name, file_name_path):
        """
        Save or show the plot based on the user's preference.

        Parameters
        ----------
        model_name : str
            The name of the model to use in the saved file name.
        file_name_path : str
            Path where the plot will be saved.
        """
        file_path = f"{file_name_path}ROC_{model_name}.pdf"
        if self.roc_curve_show_plot:
            plt.show()
            logging.debug(f"ROC curve displayed for {model_name}.")
        else:
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            logging.info(f"ROC curve saved to {file_path}.")

    def plot_roc_curve(self, probabilities_predicted, file_name_path, save_json=True):
        """
        Plots the ROC curve and calculates AUC for each class based on the predicted probabilities.

        This method generates and saves the ROC plots. Additionally, all plot data is saved
        to JSON for reproducibility.

        **IMPORTANT**: This method enforces Seaborn darkgrid style for consistent appearance.

        Parameters
        ----------
        probabilities_predicted : dict
            A dictionary containing:
            - 'model_name': Name of the model.
            - 'predicted': An array of predicted probabilities for each class.
            - 'ground_truth': The ground truth labels.
        file_name_path : str
            Path to save the ROC curve plot.
        save_json : bool, optional
            If True, saves plot data to JSON files. Default is True.
        """
        logging.info("Starting to plot ROC curve.")

        # ============================================================
        # ENFORCE SEABORN STYLE FOR CONSISTENT PURPLE/LILAC BACKGROUND
        # ============================================================
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            logging.info("Applied seaborn-v0_8-darkgrid style")
        except:
            # Fallback for older matplotlib versions
            try:
                plt.style.use('seaborn-darkgrid')
                logging.info("Applied seaborn-darkgrid style (fallback)")
            except:
                logging.warning("Could not apply seaborn style. Using default style.")
        # ============================================================

        # Ensure output path exists
        Path(file_name_path).mkdir(parents=True, exist_ok=True)

        try:
            # Process the input data
            model_name, y_score, y_true = self._roc_curve_process_data(probabilities_predicted)

            logging.info(f"Model: {model_name}, Number of classes: {y_score.shape[1]}")

            # Binarize the ground truth labels
            y_true_bin = self._roc_curve_binarize_labels(y_true, y_score.shape[1])

            # Calculate ROC and AUC for each class
            false_positive_r, true_positive_r, roc_auc = self._roc_curve_calculate_roc_auc(y_true_bin, y_score)

            # Save plot data to JSON if enabled
            if save_json:
                self._save_roc_data_to_json(model_name, y_score, y_true, false_positive_r,
                                            true_positive_r, roc_auc, file_name_path)

            # ============ CORREÇÃO: Configure tick sizes BEFORE creating figure ============
            plt.rcParams['xtick.labelsize'] = 18  # Tamanho dos números do eixo X
            plt.rcParams['ytick.labelsize'] = 18  # Tamanho dos números do eixo Y
            # ===============================================================================

            # Plot the ROC curves for each class
            plt.figure(figsize=self.roc_curve_figure_size)
            for i in range(y_score.shape[1]):
                self._roc_curve_plot_single(false_positive_r[i], true_positive_r[i], roc_auc[i], i)

            # Add details like title, labels, legend, and grid to the plot
            self._roc_curve_add_plot_details(model_name)

            # Optionally, add the diagonal line representing a random classifier
            self._roc_curve_plot_diagonal_line()

            # Save or display the plot based on the user's preference
            self._roc_curve_save_or_show_plot(model_name, file_name_path)

        except Exception as e:
            logging.error(f"Error occurred while plotting ROC curve: {e}")
            raise