#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
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
        @comparative_metrics_figure_width : float, optional Width of the figure (default is 12).
        @comparative_metrics_figure_height : float, optional Height of the figure (default is 8).
        @comparative_metrics_bar_width : float, optional Width of the bars in the plot (default is 0.20).
        @comparative_metrics_caption_size : int, optional Size of the caption font (default is 10).
        @comparative_metrics_show_plot : bool, optional Whether to show the plot interactively (default is False).


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

    def __init__(self, comparative_metrics_figure_width: int, comparative_metrics_figure_height: int,
                 comparative_metrics_bar_width: float, comparative_metrics_caption_size: int,
                 comparative_metrics_show_plot: bool):
        """
        Initialize the ComparativeMetricsPlotter with customizable plot options.

        Parameters
        ----------
        @comparative_metrics_figure_width : float, optional
            Width of the figure (default is 12).
        @comparative_metrics_figure_height : float, optional
            Height of the figure (default is 8).
        @comparative_metrics_bar_width : float, optional
            Width of the bars in the plot (default is 0.20).
        @comparative_metrics_caption_size : int, optional
            Size of the caption font (default is 10).
        @comparative_metrics_show_plot : bool, optional
            Whether to show the plot interactively (default is False).
        """

        # Validate inputs
        if not isinstance(comparative_metrics_figure_width, (int, float)) or comparative_metrics_figure_width <= 0:
            raise ValueError("Width of the figure must be a positive number.")

        if not isinstance(comparative_metrics_figure_height, (int, float)) or comparative_metrics_figure_height <= 0:
            raise ValueError("Height of the figure must be a positive number.")

        if not isinstance(comparative_metrics_bar_width, (int, float)) or not (0 < comparative_metrics_bar_width <= 1):
            raise ValueError("Bar width must be a float between 0 and 1.")

        if not isinstance(comparative_metrics_caption_size, int) or comparative_metrics_caption_size <= 0:
            raise ValueError("Caption size must be a positive integer.")

        if not isinstance(comparative_metrics_show_plot, bool):
            raise ValueError("Show plot must be a boolean value.")

        self._comparative_metrics_figure_width = comparative_metrics_figure_width
        self._comparative_metrics_figure_height = comparative_metrics_figure_height
        self._comparative_metrics_bar_width = comparative_metrics_bar_width
        self._comparative_metrics_caption_size = comparative_metrics_caption_size
        self._comparative_metrics_show_plot = comparative_metrics_show_plot

        self._comparative_metrics_list_metrics = ['Acc.', 'Prec.', 'Rec.', 'F1.']
        # Paleta de cores baseada na imagem fornecida
        # Gradientes do claro ao escuro para cada métrica
        self._comparative_metrics_list_color_bases = {
            'Acc.': ['#87CEEB', '#5DADE2', '#2E86C1', '#1A5276'],  # Blues: claro → escuro
            'Prec.': ['#A3D977', '#52BE80', '#27AE60', '#186A3B'],  # Greens: claro → escuro
            'Rec.': ['#F8B88B', '#EC7063', '#CB4335', '#943126'],  # Oranges/Reds: claro → escuro
            'F1.': ['#BB8FCE', '#9B59B6', '#7D3C98', '#5B2C6F']  # Purples: claro → escuro
        }

    def _comparative_metrics_extract_metric_data(self, dictionary_metrics_list):
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
        number_metrics = len(self._comparative_metrics_list_metrics)

        return self._comparative_metrics_list_metrics, number_models, number_metrics

    def _comparative_metrics_create_figure(self):
        """
        Creates the figure and axis for the plot.

        Returns
        -------
        figure, axis
            The created matplotlib figure and axis.
        """
        # Configurar estilo profissional para publicação científica
        # Baseado em padrões de Nature, Science, Springer e Elsevier
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['legend.framealpha'] = 1
        plt.rcParams['legend.edgecolor'] = '0.3'
        plt.rcParams['legend.fancybox'] = False
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
        plt.rcParams['ps.fonttype'] = 42

        figure_plot, axis_plot = plt.subplots(figsize=(self._comparative_metrics_figure_width,
                                                       self._comparative_metrics_figure_height))

        # Grid horizontal muito sutil e suave
        axis_plot.yaxis.grid(True, linestyle='-', alpha=0.12, color='0.75', linewidth=0.5, zorder=0)
        axis_plot.set_axisbelow(True)

        # Configurar fundo branco suave
        axis_plot.set_facecolor('#FAFAFA')
        figure_plot.patch.set_facecolor('white')

        return figure_plot, axis_plot

    def _comparative_metrics_plot_bars_for_metric(self, axis_plot, positions, metric_id, model_data,
                                                  metric_name, metric_color_base, number_models):
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
        metric_color_base : list
            Lista de cores para a métrica atual.
        number_models : int
            Total number of models being compared.
        """
        for metric_dictionary_id, model in enumerate(model_data):

            metric_values = model[metric_name]['value']
            metric_stander_deviation = model[metric_name]['std']
            # Usar cores da lista
            metric_color_bar = metric_color_base[metric_dictionary_id % len(metric_color_base)]
            metric_label = f"{model['model_name']} ({metric_name})"

            bar_definitions = axis_plot.bar(
                positions[metric_id] + metric_dictionary_id * self._comparative_metrics_bar_width,
                metric_values,
                yerr=metric_stander_deviation,
                color=metric_color_bar,
                width=self._comparative_metrics_bar_width,
                edgecolor='0.4',
                linewidth=0.7,
                capsize=5,
                error_kw={'elinewidth': 0.9, 'capthick': 0.9, 'ecolor': '0.4'},
                label=metric_label,
                alpha=0.95,
                zorder=3
            )

            # Anotações mais discretas e suaves
            for shape_bar in bar_definitions:
                bar_height = shape_bar.get_height()
                axis_plot.annotate(f'{bar_height:.3f}',
                                   xy=(shape_bar.get_x() + shape_bar.get_width() / 2, bar_height),
                                   xytext=(0, 5),
                                   textcoords="offset points",
                                   ha='center',
                                   va='bottom',
                                   fontsize=9,
                                   color='0.4')

    def _comparative_metrics_set_plot_labels(self, axis_plot, positions, number_models):
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
        # Labels com fonte adequada para publicação - cores suaves
        axis_plot.set_xlabel('Performance Metric', fontsize=13, color='0.3')
        axis_plot.set_xticks([r + self._comparative_metrics_bar_width * (number_models - 1) / 2 for r in positions])
        axis_plot.set_xticklabels(self._comparative_metrics_list_metrics, fontsize=12)
        axis_plot.set_ylabel('Score', fontsize=13, color='0.3')

        # Título mais discreto ou pode ser removido (muitos journals preferem sem título)
        # axis_plot.set_title('Model Performance Comparison', fontsize=14, pad=15, color='0.3')

        # Legenda profissional com bordas suaves
        legend = axis_plot.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(number_models, 4),  # Máximo 4 colunas para melhor legibilidade
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor='0.5',
            fontsize=10,
            framealpha=1,
            columnspacing=1.5,
            handlelength=2,
            handleheight=1.5
        )
        legend.get_frame().set_linewidth(0.6)

        # Melhorar aparência dos eixos (estilo minimalista científico) - cores suaves
        axis_plot.spines['top'].set_visible(False)
        axis_plot.spines['right'].set_visible(False)
        axis_plot.spines['left'].set_linewidth(0.7)
        axis_plot.spines['bottom'].set_linewidth(0.7)
        axis_plot.spines['left'].set_color('0.5')
        axis_plot.spines['bottom'].set_color('0.5')

        # Ajustar cor dos ticks - mais suaves
        axis_plot.tick_params(axis='both', colors='0.4', which='major')

    def _comparative_metrics_save_or_show_plot(self, file_name):
        """
        Saves or shows the plot based on user preference.

        Parameters
        ----------
        file_name : str
            The base name for the saved plot file.
        """
        output_path = f'{file_name}metrics.pdf'
        if self._comparative_metrics_show_plot:
            plt.show()
        else:
            plt.tight_layout()
            # Salvar em alta qualidade para publicação
            plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf',
                        facecolor='white', edgecolor='none', transparent=False,
                        metadata={'Creator': 'Matplotlib', 'Author': __author__})
            # Salvar também em PNG de alta resolução
            output_path_png = f'{file_name}metrics.png'
            plt.savefig(output_path_png, dpi=600, bbox_inches='tight', format='png',
                        facecolor='white', edgecolor='none', transparent=False)
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
            list_metrics, number_models, number_metrics \
                = self._comparative_metrics_extract_metric_data(dictionary_metrics_list)

            # Create figure
            figure_plot, axis_plot = self._comparative_metrics_create_figure()
            positions = numpy.arange(number_metrics)

            # Plot bars for each metric
            for metric_id, metric_name in enumerate(list_metrics):
                metric_color_base = self._comparative_metrics_list_color_bases[metric_name]
                self._comparative_metrics_plot_bars_for_metric(axis_plot, positions, metric_id,
                                                               dictionary_metrics_list, metric_name,
                                                               metric_color_base, number_models)

            # Set plot labels, title, and legend
            self._comparative_metrics_set_plot_labels(axis_plot, positions, number_models)

            # Save or display the plot
            self._comparative_metrics_save_or_show_plot(file_name)

        except Exception as e:
            logging.error(f"An error occurred while plotting comparative metrics: {e}")
            raise