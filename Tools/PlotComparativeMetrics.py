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

except ImportError as error:
    print(error)
    sys.exit(-1)



class ComparativeMetricsPlotter:
    """
    A class to plot comparative metrics for multiple models.

    This class allows for plotting a comparative bar chart of multiple metrics
    (e.g., Accuracy, Precision, Recall, F1) for different models, including
    standard deviation error bars.

    Parameters:
    ----------
        @figure_width : float, optional Width of the figure (default is 12).
        @figure_height : float, optional Height of the figure (default is 8).
        @bar_width : float, optional Width of the bars in the plot (default is 0.20).
        @caption_size : int, optional Size of the caption font (default is 10).
        @show_plot : bool, optional Whether to show the plot interactively (default is False).


    Example:
    -------
        >>> # python
        ...     metrics_data = [
        ...     {
        ...     'model_name': 'Model A',
        ...     'Acc.': {'value': 0.85, 'std': 0.02},
        ...     'Prec.': {'value': 0.80, 'std': 0.03},
        ...     'Rec.': {'value': 0.75, 'std': 0.05},
        ...     'F1.': {'value': 0.77, 'std': 0.04}
        ...     },
        ...     {
        ...     'model_name': 'Model B',
        ...     'Acc.': {'value': 0.88, 'std': 0.01},
        ...     'Prec.': {'value': 0.82, 'std': 0.02},
        ...     'Rec.': {'value': 0.78, 'std': 0.03},
        ...     'F1.': {'value': 0.80, 'std': 0.02}
        ...     }
        ...     ]
        ...
        ...     plotter = ComparativeMetricsPlotter()
        ...     plotter.plot_comparative_metrics(metrics_data,
        ...     'comparison_', figure_width=14, figure_height=10, show_plot=True)
        >>>
    """

    def __init__(self, figure_width: int, figure_height: int, bar_width: float, caption_size: int, show_plot: bool):
        """
        Initialize the ComparativeMetricsPlotter with customizable plot options.

        Parameters
        ----------
        figure_width : float, optional
            Width of the figure (default is 12).
        figure_height : float, optional
            Height of the figure (default is 8).
        bar_width : float, optional
            Width of the bars in the plot (default is 0.20).
        caption_size : int, optional
            Size of the caption font (default is 10).
        show_plot : bool, optional
            Whether to show the plot interactively (default is False).
        """
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.bar_width = bar_width
        self.caption_size = caption_size
        self.show_plot = show_plot

        self.list_metrics = ['Acc.', 'Prec.', 'Rec.', 'F1.']
        self.list_color_bases = {
            'Acc.': 'Blues',
            'Prec.': 'Greens',
            'Rec.': 'Reds',
            'F1.': 'Purples'
        }

    def _extract_metric_data(self, dictionary_metrics_list):
        """
        Extracts metric values and standard deviations from the provided data.

        Parameters
        ----------
        dictionary_metrics_list : list of dict
            List of dictionaries containing metric values and std for each model.

        Returns
        -------
        tuple
            A tuple containing:
            - list of metric names (e.g., ['Acc.', 'Prec.', 'Rec.', 'F1.'])
            - number of models
            - number of metrics
            - metric values for each model
        """
        number_models = len(dictionary_metrics_list)
        number_metrics = len(self.list_metrics)

        return self.list_metrics, number_models, number_metrics

    def _create_figure(self):
        """
        Creates the figure and axis for the plot.

        Returns
        -------
        figure, axis
            The created matplotlib figure and axis.
        """
        figure_plot, axis_plot = plt.subplots(figsize=(self.figure_width, self.figure_height))
        return figure_plot, axis_plot

    def _plot_bars_for_metric(self, axis_plot, positions, metric_id, model_data, metric_name, metric_color_base,
                              number_models):
        """
        Plots bars for a specific metric across all models.

        Parameters
        ----------
        axis_plot : matplotlib axis
            The axis on which to plot the bars.
        positions : array
            The x-positions for the bars.
        metric_id : int
            Index of the metric in the list (e.g., 'Acc.', 'Prec.', etc.).
        model_data : dict
            Data for the current model, including metric values and standard deviation.
        metric_name : str
            Name of the metric ('Acc.', 'Prec.', etc.).
        metric_color_base : str
            The color map base for the current metric.
        number_models : int
            Total number of models being compared.
        """
        for metric_dictionary_id, model in enumerate(model_data):
            metric_values = model[metric_name]['value']
            metric_stander_deviation = model[metric_name]['std']
            metric_color_bar = plt.get_cmap(metric_color_base)(metric_dictionary_id / (number_models - 1))
            metric_label = f"{metric_name} {model['model_name']}"

            bar_definitions = axis_plot.bar(
                positions[metric_id] + metric_dictionary_id * self.bar_width,
                metric_values,
                yerr=metric_stander_deviation,
                color=metric_color_bar,
                width=self.bar_width,
                edgecolor='grey',
                capsize=self.caption_size,
                label=metric_label
            )

            for shape_bar in bar_definitions:
                bar_height = shape_bar.get_height()
                axis_plot.annotate(f'{bar_height:.2f}',
                                   xy=(shape_bar.get_x() + shape_bar.get_width() / 2, bar_height),
                                   xytext=(0, 10),
                                   textcoords="offset points",
                                   ha='center',
                                   va='bottom')

    def _set_plot_labels(self, axis_plot, positions, number_models):
        """
        Set the labels, title, and legend for the plot.

        Parameters
        ----------
        axis_plot : matplotlib axis
            The axis on which to set the labels and legend.
        positions : array
            The x-positions for the bars.
        number_models : int
            Total number of models being compared.
        """
        axis_plot.set_xlabel('Metric', fontweight='bold')
        axis_plot.set_xticks([r + self.bar_width * (number_models - 1) / 2 for r in positions])
        axis_plot.set_xticklabels(self.list_metrics)
        axis_plot.set_ylabel('Score', fontweight='bold')
        axis_plot.set_title('Comparative Metrics', fontweight='bold')
        axis_plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=number_models)

    def _save_or_show_plot(self, file_name):
        """
        Saves or shows the plot based on user preference.

        Parameters
        ----------
        file_name : str
            The base name for the saved plot file.
        """
        output_path = f'{file_name}metrics.png'
        if self.show_plot:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(output_path)
        logging.debug(f"Comparative metrics plot saved to {output_path}")
        plt.close()

    def plot_comparative_metrics(self, dictionary_metrics_list, file_name):
        """
        Plots comparative metrics for different models, including standard deviation bars.

        Parameters
        ----------
        dictionary_metrics_list : list of dict
            List of dictionaries containing model names and metrics (e.g., Accuracy, Precision).
        file_name : str
            The path where the plot will be saved.
        """
        logging.info("Starting to plot comparative metrics.")

        try:
            # Extract data
            list_metrics, number_models, number_metrics = self._extract_metric_data(dictionary_metrics_list)

            # Create figure
            figure_plot, axis_plot = self._create_figure()
            positions = numpy.arange(number_metrics)

            # Plot bars for each metric
            for metric_id, metric_name in enumerate(list_metrics):
                metric_color_base = self.list_color_bases[metric_name]
                self._plot_bars_for_metric(axis_plot, positions, metric_id, dictionary_metrics_list, metric_name,
                                           metric_color_base, number_models)

            # Set plot labels, title, and legend
            self._set_plot_labels(axis_plot, positions, number_models)

            # Save or display the plot
            self._save_or_show_plot(file_name)

        except Exception as e:
            logging.error(f"An error occurred while plotting comparative metrics: {e}")
            raise