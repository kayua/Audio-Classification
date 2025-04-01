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
# Copyright (c) 2025 Synthetic Ocean AI
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

    from Tools.PlotROC import ROCPlotter
    from Tools.PlotLoss import PlotLossCurve

    from Tools.PlotConfusionMatrix import ConfusionMatrixPlotter
    from Tools.PlotComparativeMetrics import ComparativeMetricsPlotter

except ImportError as error:
    print(error)
    sys.exit(-1)


class PlotterTools(ComparativeMetricsPlotter, ConfusionMatrixPlotter, PlotLossCurve, ROCPlotter):
    """
    A class that combines the functionalities of ComparativeMetricsPlotter, ConfusionMatrixPlotter,
    PlotLossCurve, and ROCPlotter. This class provides tools for visualizing various evaluation
    metrics and results in machine learning models, including comparative metrics plots, confusion matrices,
    loss curves, and ROC curves.

    It inherits from the following classes:
        - ComparativeMetricsPlotter: For plotting and visualizing comparative metrics.
        - ConfusionMatrixPlotter: For plotting confusion matrices.
        - PlotLossCurve: For plotting training and validation loss curves.
        - ROCPlotter: For plotting ROC curves and calculating AUC.

    Parameters:
    ----------
        @comparative_metrics_figure_width : int The width of the figure for comparative metrics plots.
        @comparative_metrics_figure_height : int The height of the figure for comparative metrics plots.
        @comparative_metrics_bar_width : float The width of the bars in the comparative metrics plots.
        @comparative_metrics_caption_size : int The font size for captions in comparative metrics plots.
        @comparative_metrics_show_plot : bool Whether to display the comparative metrics plot interactively.
        @confusion_matrix_figure_size : tuple The size of the figure for the confusion matrix plot (width, height).
        @confusion_matrix_cmap : str The colormap to use for the confusion matrix plot.
        @confusion_matrix_annot_font_size : int The font size for annotations in the confusion matrix plot.
        @confusion_matrix_label_font_size : int The font size for the axis labels in the confusion matrix plot.
        @confusion_matrix_title_font_size : int The font size for the title in the confusion matrix plot.
        @confusion_matrix_show_plot : bool Whether to display the confusion matrix plot interactively.
        @confusion_matrix_colorbar : bool Whether to display the colorbar in the confusion matrix plot.
        @confusion_matrix_annot_kws : dict or None Additional keyword arguments for annotation in the confusion matrix plot.
        @confusion_matrix_fmt : str The format string for displaying the confusion matrix values.
        @confusion_matrix_rotation : int The angle to rotate the axis labels in the confusion matrix plot.

        @loss_curve_history_dict_list : list A list of dictionaries containing model names and loss history for plotting loss curves.
        @loss_curve_path_output : str The directory path to save the loss curve plot.
        @loss_curve_figure_size : tuple The size of the figure for the loss curve plot (width, height).
        @loss_curve_training_loss_color : str The color of the training loss curve.
        @loss_curve_validation_loss_color : str The color of the validation loss curve.
        @loss_curve_title_font_size : int The font size for the title of the loss curve plot.
        @loss_curve_axis_font_size : int The font size for the axis labels of the loss curve plot.
        @loss_curve_legend_font_size : int The font size for the legend of the loss curve plot.
        @loss_curve_x_label : str The label for the x-axis of the loss curve plot.
        @loss_curve_y_label : str The label for the y-axis of the loss curve plot.
        @loss_curve_title : str The title of the loss curve plot.
        @loss_curve_grid : bool Whether to display grid lines on the loss curve plot.
        @loss_curve_line_style_training : str The line style for the training loss curve.
        @loss_curve_line_style_validation : str The line style for the validation loss curve.
        @loss_curve_line_width : int The line width for both the training and validation loss curves.

        @roc_curve_figure_size : tuple The size of the figure for the ROC curve plot (width, height).
        @roc_curve_line_width : int The line width for the ROC curve plot.
        @roc_curve_marker_style : str The marker style for the ROC curve plot.
        @roc_curve_cmap : str The colormap for the ROC curve plot.
        @roc_curve_show_plot : bool Whether to display the ROC curve plot interactively.
        @roc_curve_title_font_size : int The font size for the title of the ROC curve plot.
        @roc_curve_axis_label_font_size : int The font size for the axis labels of the ROC curve plot.
        @roc_curve_legend_font_size : int The font size for the legend of the ROC curve plot.
        @roc_curve_grid : bool Whether to display grid lines on the ROC curve plot.
        @roc_curve_diagonal_line : bool Whether to include the diagonal line (representing random guessing) on the ROC curve plot.

    Example:
    -------
    >>> # python3
    >>> plotter = PlotterTools(arguments)
    ...     plotter.plot_comparative_metrics()
    ...     plotter.plot_confusion_matrix()
    ...     plotter.plot_loss()
    >>>     plotter.plot_roc_curve()
    """
    def __init__(self, arguments):


        ComparativeMetricsPlotter.__init__(self,
                                           arguments.comparative_metrics_figure_width,
                                           arguments.comparative_metrics_figure_height,
                                           arguments.comparative_metrics_bar_width,
                                           arguments.comparative_metrics_caption_size,
                                           arguments.comparative_metrics_show_plot)

        ConfusionMatrixPlotter.__init__(self,
                                        arguments.confusion_matrix_figure_size,
                                        arguments.confusion_matrix_color_map,
                                        arguments.confusion_matrix_annot_font_size,
                                        arguments.confusion_matrix_label_font_size,
                                        arguments.confusion_matrix_title_font_size,
                                        arguments.confusion_matrix_show_plot,
                                        arguments.confusion_matrix_color_bar,
                                        arguments.confusion_matrix_annot_kws,
                                        arguments.confusion_matrix_fmt,
                                        arguments.confusion_matrix_rotation)

        PlotLossCurve.__init__(self,
                               arguments.loss_curve_figure_size,
                               arguments.loss_curve_training_loss_color,
                               arguments.loss_curve_validation_loss_color,
                               arguments.loss_curve_title_font_size,
                               arguments.loss_curve_axis_font_size,
                               arguments.loss_curve_legend_font_size,
                               arguments.loss_curve_x_label,
                               arguments.loss_curve_y_label,
                               arguments.loss_curve_title,
                               arguments.loss_curve_grid,
                               arguments.loss_curve_line_style_training,
                               arguments.loss_curve_line_style_validation,
                               arguments.loss_curve_line_width)

        ROCPlotter.__init__(self,
                            arguments.roc_curve_figure_size,
                            arguments.roc_curve_line_width,
                            arguments.roc_curve_marker_style,
                            arguments.roc_curve_color_map,
                            arguments.roc_curve_show_plot,
                            arguments.roc_curve_title_font_size,
                            arguments.roc_curve_axis_label_font_size,
                            arguments.roc_curve_legend_font_size,
                            arguments.roc_curve_grid,
                            arguments.roc_curve_diagonal_line)


