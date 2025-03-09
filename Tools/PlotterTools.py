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
    >>> plotter = PlotterTools(
    ...     comparative_metrics_figure_width=10, comparative_metrics_figure_height=8,
    ...     comparative_metrics_bar_width=0.3, comparative_metrics_caption_size=12,
    ...     comparative_metrics_show_plot=True, confusion_matrix_figure_size=(8, 6),
    ...     confusion_matrix_cmap="Blues", confusion_matrix_annot_font_size=10,
    ...     confusion_matrix_label_font_size=12, confusion_matrix_title_font_size=14,
    ...     confusion_matrix_show_plot=True, confusion_matrix_colorbar=True,
    ...     confusion_matrix_annot_kws=None, confusion_matrix_fmt='d',
    ...     confusion_matrix_rotation=45, loss_curve_history_dict_list=[...],
    ...     loss_curve_path_output='/path/to/save/plots', loss_curve_figure_size=(10, 6),
    ...     loss_curve_training_loss_color='blue', loss_curve_validation_loss_color='orange',
    ...     loss_curve_title_font_size=16, loss_curve_axis_font_size=12,
    ...     loss_curve_legend_font_size=10, loss_curve_x_label="Epochs", loss_curve_y_label="Loss",
    ...     loss_curve_title="Loss Curves", loss_curve_grid=True, loss_curve_line_style_training='-',
    ...     loss_curve_line_style_validation='--', loss_curve_line_width=2,
    ...     roc_curve_figure_size=(10, 6), roc_curve_line_width=2, roc_curve_marker_style='o',
    ...     roc_curve_cmap='Blues', roc_curve_show_plot=True, roc_curve_title_font_size=14,
    ...     roc_curve_axis_label_font_size=12, roc_curve_legend_font_size=10,
    ...     roc_curve_grid=True, roc_curve_diagonal_line=True
    ...     )
    ...     plotter.plot_comparative_metrics()
    ...     plotter.plot_confusion_matrix()
    ...     plotter.plot_loss()
    >>>     plotter.plot_roc_curve()
    """
    def __init__(self, comparative_metrics_figure_width: int, comparative_metrics_figure_height: int,
                 comparative_metrics_bar_width: float, comparative_metrics_caption_size: int,
                 comparative_metrics_show_plot: bool, confusion_matrix_figure_size:tuple, confusion_matrix_cmap: str,
                 confusion_matrix_annot_font_size: int, confusion_matrix_label_font_size: int,
                 confusion_matrix_title_font_size: int, confusion_matrix_show_plot: bool,
                 confusion_matrix_colorbar: bool, confusion_matrix_annot_kws: None, confusion_matrix_fmt: str,
                 confusion_matrix_rotation: int, loss_curve_history_dict_list, loss_curve_path_output,
                 loss_curve_figure_size: tuple, loss_curve_training_loss_color: str,
                 loss_curve_validation_loss_color: str, loss_curve_title_font_size: int, loss_curve_axis_font_size: int,
                 loss_curve_legend_font_size: int, loss_curve_x_label: str, loss_curve_y_label: str,
                 loss_curve_title: str, loss_curve_grid: bool, loss_curve_line_style_training: str,
                 loss_curve_line_style_validation: str, loss_curve_line_width: int, roc_curve_figure_size: tuple,
                 roc_curve_line_width: int, roc_curve_marker_style: str, roc_curve_cmap: str, roc_curve_show_plot: bool,
                 roc_curve_title_font_size: int, roc_curve_axis_label_font_size: int, roc_curve_legend_font_size: int,
                 roc_curve_grid: bool, roc_curve_diagonal_line: bool):


        ComparativeMetricsPlotter.__init__(self, comparative_metrics_figure_width, comparative_metrics_figure_height,
                                           comparative_metrics_bar_width, comparative_metrics_caption_size,
                                           comparative_metrics_show_plot)

        ConfusionMatrixPlotter.__init__(self, confusion_matrix_figure_size, confusion_matrix_cmap,
                                        confusion_matrix_annot_font_size, confusion_matrix_label_font_size,
                                        confusion_matrix_title_font_size, confusion_matrix_show_plot,
                                        confusion_matrix_colorbar, confusion_matrix_annot_kws, confusion_matrix_fmt,
                                        confusion_matrix_rotation)

        PlotLossCurve.__init__(self, loss_curve_history_dict_list, loss_curve_path_output, loss_curve_figure_size,
                               loss_curve_training_loss_color, loss_curve_validation_loss_color,
                               loss_curve_title_font_size, loss_curve_axis_font_size, loss_curve_legend_font_size,
                               loss_curve_x_label, loss_curve_y_label, loss_curve_title, loss_curve_grid,
                               loss_curve_line_style_training, loss_curve_line_style_validation, loss_curve_line_width)

        ROCPlotter.__init__(self, roc_curve_figure_size, roc_curve_line_width, roc_curve_marker_style, roc_curve_cmap,
                            roc_curve_show_plot, roc_curve_title_font_size, roc_curve_axis_label_font_size,
                            roc_curve_legend_font_size, roc_curve_grid, roc_curve_diagonal_line)


