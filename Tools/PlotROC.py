#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import sys

    import numpy
    import logging

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

    Parameters:
    ----------
        @figure_size : tuple, optional
            Size of the figure (default is (8, 6)).
        @line_width : int, optional
            Line width of the ROC curve (default is 2).
        @marker_style : str, optional
            Style of markers on the ROC curve (default is 'o').
        @cmap : str, optional
            Color map for the ROC curve (default is 'Blues').
        @show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @title_font_size : int, optional
            Font size for the plot title (default is 14).
        @axis_label_font_size : int, optional
            Font size for the axis labels (default is 12).
        @legend_font_size : int, optional
            Font size for the legend (default is 10).
        @grid : bool, optional
            Whether to display the grid (default is True).
        @diagonal_line : bool, optional
            Whether to plot the diagonal line (default is True).

    Example:
    -------
        >>> # python3
        ...     roc_plotter = ROCPlotter()
        ...     probabilities_predicted = {
        ...     'model_name': 'MyModel',
        ...     'predicted': np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
        ...     'ground_truth': np.array([1, 0, 1])
        ...     }
        ...     roc_plotter.plot_roc_curve(probabilities_predicted, file_name_path='roc_curve/', show_plot=True)
        >>>
    """

    def __init__(self, figure_size=(8, 6), line_width=2, marker_style='o', cmap='Blues', show_plot=False,
                 title_font_size=14, axis_label_font_size=12, legend_font_size=10, grid=True, diagonal_line=True):
        """
        Initialize the ROCPlotter with customizable plotting options.

        Parameters
        ----------
        @figure_size : tuple, optional
            Size of the figure (default is (8, 6)).
        @line_width : int, optional
            Line width of the ROC curve (default is 2).
        @marker_style : str, optional
            Style of markers on the ROC curve (default is 'o').
        @cmap : str, optional
            Color map for the ROC curve (default is 'Blues').
        @show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @title_font_size : int, optional
            Font size for the plot title (default is 14).
        @axis_label_font_size : int, optional
            Font size for the axis labels (default is 12).
        @legend_font_size : int, optional
            Font size for the legend (default is 10).
        @grid : bool, optional
            Whether to display the grid (default is True).
        @diagonal_line : bool, optional
            Whether to plot the diagonal line (default is True).
        """
        # Initialize instance variables with provided or default values
        self.figure_size = figure_size
        self.line_width = line_width
        self.marker_style = marker_style
        self.cmap = cmap
        self.show_plot = show_plot
        self.title_font_size = title_font_size
        self.axis_label_font_size = axis_label_font_size
        self.legend_font_size = legend_font_size
        self.grid = grid
        self.diagonal_line = diagonal_line

    @staticmethod
    def _process_data(probabilities_predicted):
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
    def _binarize_labels(y_true, number_classes):
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
    def _calculate_roc_auc(y_true_bin, y_score):
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
        dict, dict
            Dictionaries containing the false positive rates, true positive rates, and AUC for each class.
        """
        false_positive_r = {}
        true_positive_r = {}
        roc_auc = {}

        for index in range(y_score.shape[1]):  # Iterate over each class
            false_positive_r[index], true_positive_r[index], _ = roc_curve(y_true_bin[:, index], y_score[:, index])
            roc_auc[index] = auc(false_positive_r[index], true_positive_r[index])

        return false_positive_r, true_positive_r, roc_auc

    def _plot_single_class_roc(self, fpr, tpr, auc_score, class_index):
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
                 linestyle='-', marker=self.marker_style, linewidth=self.line_width)

    def _add_plot_details(self, model_name):
        """
        Add title, labels, legend, and grid to the plot.

        Parameters
        ----------
        model_name : str
            The name of the model to display in the plot title.
        """
        plt.xlabel('False Positive Rate', fontsize=self.axis_label_font_size)
        plt.ylabel('True Positive Rate', fontsize=self.axis_label_font_size)
        plt.title(f'ROC Curve for {model_name}', fontsize=self.title_font_size)

        if self.grid:
            plt.grid(True)

        plt.legend(loc='lower right', fontsize=self.legend_font_size)

    def _plot_diagonal_line(self):
        """
        Plot the diagonal line for a random classifier (FPR = TPR).
        """
        if self.diagonal_line:
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    def _save_or_show_plot(self, model_name, file_name_path):
        """
        Save or show the plot based on the user's preference.

        Parameters
        ----------
        model_name : str
            The name of the model to use in the saved file name.
        file_name_path : str
            Path where the plot will be saved.
        """
        file_path = f"{file_name_path}ROC_{model_name}.png"
        if self.show_plot:
            plt.show()
            logging.debug(f"ROC curve displayed for {model_name}.")

        else:
            plt.savefig(file_path)
            plt.close()
            logging.info(f"ROC curve saved to {file_path}.")

    def plot_roc_curve(self, probabilities_predicted, file_name_path):
        """
        Plots the ROC curve and calculates AUC for each class based on the predicted probabilities.

        Parameters
        ----------
        probabilities_predicted : dict
            A dictionary containing:
            - 'model_name': Name of the model.
            - 'predicted': An array of predicted probabilities for each class.
            - 'ground_truth': The ground truth labels.
        file_name_path : str
            Path to save the ROC curve plot.
        """
        logging.info("Starting to plot ROC curve.")

        try:
            # Process the input data
            model_name, y_score, y_true = self._process_data(probabilities_predicted)

            logging.info(f"Model: {model_name}, Number of classes: {y_score.shape[1]}")

            # Binarize the ground truth labels
            y_true_bin = self._binarize_labels(y_true, y_score.shape[1])

            # Calculate ROC and AUC for each class
            false_positive_r, true_positive_r, roc_auc = self._calculate_roc_auc(y_true_bin, y_score)

            # Plot the ROC curves for each class
            plt.figure(figsize=self.figure_size)
            for i in range(y_score.shape[1]):
                self._plot_single_class_roc(false_positive_r[i], true_positive_r[i], roc_auc[i], i)

            # Add details like title, labels, legend, and grid to the plot
            self._add_plot_details(model_name)

            # Optionally, add the diagonal line representing a random classifier
            self._plot_diagonal_line()

            # Save or display the plot based on the user's preference
            self._save_or_show_plot(model_name, file_name_path)

        except Exception as e:
            logging.error(f"Error occurred while plotting ROC curve: {e}")
            raise
