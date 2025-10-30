#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{1}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/30'
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
    import matplotlib.colors as mcolors

except ImportError as error:
    print(error)
    sys.exit(-1)


class ComparativeMetricsPlotter:
    """
    A class to plot comparative metrics for multiple models.

    This class allows for plotting a comparative bar chart of multiple metrics
    (e.g., Accuracy, Precision, Recall, F1) for different models, including
    standard deviation error bars. MELHORADO para lidar com muitos modelos.

    Parameters:
    ----------
        @comparative_metrics_figure_width : float, optional
            Width of the figure (default is 12). Auto-ajustado se necessário.
        @comparative_metrics_figure_height : float, optional
            Height of the figure (default is 8).
        @comparative_metrics_bar_width : float, optional
            Width of the bars in the plot (default is 0.20). Auto-ajustado se necessário.
        @comparative_metrics_caption_size : int, optional
            Size of the caption font (default is 10).
        @comparative_metrics_show_plot : bool, optional
            Whether to show the plot interactively (default is False).


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
        ...     plotter = ComparativeMetricsPlotter(
        ...         comparative_metrics_figure_width=14,
        ...         comparative_metrics_figure_height=10,
        ...         comparative_metrics_show_plot=True
        ...     )
        ...     plotter.plot_comparative_metrics(metrics_data, 'comparison_')
        >>>
    """

    def __init__(self, comparative_metrics_figure_width: int = 12,
                 comparative_metrics_figure_height: int = 8,
                 comparative_metrics_bar_width: float = 0.20,
                 comparative_metrics_caption_size: int = 10,
                 comparative_metrics_show_plot: bool = False):
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

        # Paleta de cores expandida para suportar muitos modelos
        self._comparative_metrics_list_color_bases = {
            'Acc.': ['#87CEEB', '#5DADE2', '#2E86C1', '#1A5276', '#154360', '#0E2F44', '#0B2133', '#081622'],
            'Prec.': ['#A3D977', '#52BE80', '#27AE60', '#186A3B', '#145A32', '#0E4B26', '#0A3D1E', '#072E16'],
            'Rec.': ['#F8B88B', '#EC7063', '#CB4335', '#943126', '#78281F', '#5F1F18', '#4A1812', '#35110D'],
            'F1.': ['#BB8FCE', '#9B59B6', '#7D3C98', '#5B2C6F', '#4A235A', '#3B1A48', '#2D1436', '#1F0E24']
        }

    def _generate_color_palette(self, base_colors, num_colors):
        """
        Gera uma paleta de cores com gradiente suave para qualquer número de modelos.

        Parameters
        ----------
        base_colors : list
            Lista de cores base em formato hexadecimal.
        num_colors : int
            Número de cores necessárias.

        Returns
        -------
        list
            Lista de strings de cores em formato hexadecimal.
        """
        if num_colors <= len(base_colors):
            return base_colors[:num_colors]

        # Converter cores base para RGB
        colors_rgb = [mcolors.hex2color(c) for c in base_colors]

        # Criar gradiente interpolado
        generated_colors = []
        segments = num_colors - 1
        colors_per_segment = segments / (len(base_colors) - 1)

        for i in range(len(base_colors) - 1):
            start_color = numpy.array(colors_rgb[i])
            end_color = numpy.array(colors_rgb[i + 1])

            # Número de cores neste segmento
            segment_size = int(numpy.ceil(colors_per_segment))

            for j in range(segment_size):
                if len(generated_colors) >= num_colors:
                    break

                # Interpolação linear entre as cores
                alpha = j / segment_size
                interpolated_color = start_color * (1 - alpha) + end_color * alpha
                generated_colors.append(mcolors.rgb2hex(interpolated_color))

        # Adicionar última cor se necessário
        if len(generated_colors) < num_colors:
            generated_colors.append(base_colors[-1])

        return generated_colors[:num_colors]

    def _calculate_optimal_layout(self, number_models):
        """
        Calcula o layout ótimo baseado no número de modelos.

        Parameters
        ----------
        number_models : int
            Número de modelos a serem plotados.

        Returns
        -------
        dict
            Dicionário com parâmetros otimizados.
        """
        layout = {
            'bar_width': self._comparative_metrics_bar_width,
            'figure_width': self._comparative_metrics_figure_width,
            'figure_height': self._comparative_metrics_figure_height,
            'show_annotations': True,
            'annotation_fontsize': 9,
            'annotation_rotation': 0,
            'legend_ncol': min(number_models, 4),
            'legend_fontsize': 10,
            'group_spacing': 1.5
        }

        # Ajustes para diferentes quantidades de modelos
        if number_models <= 4:
            # Configuração padrão para poucos modelos
            pass
        elif number_models <= 6:
            layout['bar_width'] = 0.18
            layout['figure_width'] = max(self._comparative_metrics_figure_width, 14)
        elif number_models <= 8:
            layout['bar_width'] = 0.15
            layout['figure_width'] = max(self._comparative_metrics_figure_width, 16)
            layout['annotation_fontsize'] = 8
            layout['legend_ncol'] = 4
        elif number_models <= 12:
            layout['bar_width'] = 0.12
            layout['figure_width'] = max(self._comparative_metrics_figure_width, 18)
            layout['annotation_fontsize'] = 7
            layout['annotation_rotation'] = 45
            layout['legend_ncol'] = 6
            layout['legend_fontsize'] = 9
            layout['group_spacing'] = 1.8
        elif number_models <= 16:
            layout['bar_width'] = 0.10
            layout['figure_width'] = max(self._comparative_metrics_figure_width, 22)
            layout['annotation_fontsize'] = 6
            layout['annotation_rotation'] = 60
            layout['legend_ncol'] = 8
            layout['legend_fontsize'] = 8
            layout['group_spacing'] = 2.0
        else:
            # Muitos modelos - desabilitar anotações
            layout['bar_width'] = 0.08
            layout['figure_width'] = max(self._comparative_metrics_figure_width, 24 + (number_models - 16) * 0.5)
            layout['show_annotations'] = False
            layout['legend_ncol'] = min(10, number_models)
            layout['legend_fontsize'] = 8
            layout['group_spacing'] = 2.2

        return layout

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
        """
        number_models = len(dictionary_metrics_list)
        number_metrics = len(self._comparative_metrics_list_metrics)

        return self._comparative_metrics_list_metrics, number_models, number_metrics

    def _comparative_metrics_create_figure(self, layout):
        """
        Creates the figure and axis for the plot.

        Parameters
        ----------
        layout : dict
            Layout parameters calculated based on number of models.

        Returns
        -------
        figure, axis
            The created matplotlib figure and axis.
        """
        # Configurar estilo profissional para publicação científica
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
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        figure_plot, axis_plot = plt.subplots(
            figsize=(layout['figure_width'], layout['figure_height'])
        )

        # Grid horizontal sutil
        axis_plot.yaxis.grid(True, linestyle='-', alpha=0.12, color='0.75', linewidth=0.5, zorder=0)
        axis_plot.set_axisbelow(True)

        # Configurar fundo
        axis_plot.set_facecolor('#FAFAFA')
        figure_plot.patch.set_facecolor('white')

        return figure_plot, axis_plot

    def _comparative_metrics_plot_bars_for_metric(self, axis_plot, positions, metric_id, model_data,
                                                  metric_name, metric_colors, number_models, layout):
        """
        Plots bars for a specific metric across all models.

        Parameters
        ----------
        axis_plot : matplotlib axis
            The axis on which to plot the bars.
        positions : array
            The x-positions for the bars.
        metric_id : int
            Index of the metric in the list.
        model_data : list of dict
            Data for all models.
        metric_name : str
            Name of the metric ('Acc.', 'Prec.', etc.).
        metric_colors : list
            List of colors for this metric.
        number_models : int
            Total number of models.
        layout : dict
            Layout parameters.
        """
        bar_width = layout['bar_width']
        group_spacing = layout['group_spacing']

        for metric_dictionary_id, model in enumerate(model_data):
            metric_values = model[metric_name]['value']
            metric_stander_deviation = model[metric_name]['std']

            # Usar cor da paleta gerada
            metric_color_bar = metric_colors[metric_dictionary_id]
            metric_label = f"{model['model_name']} ({metric_name})"

            # Calcular posição da barra
            bar_position = positions[metric_id] * group_spacing + metric_dictionary_id * bar_width

            bar_definitions = axis_plot.bar(
                bar_position,
                metric_values,
                yerr=metric_stander_deviation,
                color=metric_color_bar,
                width=bar_width,
                edgecolor='0.4',
                linewidth=0.7,
                capsize=4,
                error_kw={'elinewidth': 0.9, 'capthick': 0.9, 'ecolor': '0.4'},
                label=metric_label,
                alpha=0.95,
                zorder=3
            )

            # Adicionar anotações apenas se configurado
            if layout['show_annotations']:
                for shape_bar in bar_definitions:
                    bar_height = shape_bar.get_height()
                    axis_plot.annotate(
                        f'{bar_height:.3f}',
                        xy=(shape_bar.get_x() + shape_bar.get_width() / 2, bar_height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=layout['annotation_fontsize'],
                        color='0.4',
                        rotation=layout['annotation_rotation']
                    )

    def _comparative_metrics_set_plot_labels(self, axis_plot, positions, number_models, layout):
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
        layout : dict
            Layout parameters.
        """
        bar_width = layout['bar_width']
        group_spacing = layout['group_spacing']

        # Labels dos eixos
        axis_plot.set_xlabel('Performance Metric', fontsize=13, color='0.3')
        axis_plot.set_ylabel('Score', fontsize=13, color='0.3')

        # Calcular posições centralizadas para os labels das métricas
        tick_positions = [
            positions[i] * group_spacing + bar_width * (number_models - 1) / 2
            for i in range(len(positions))
        ]
        axis_plot.set_xticks(tick_positions)
        axis_plot.set_xticklabels(self._comparative_metrics_list_metrics, fontsize=12)

        # Configurar legenda
        legend = axis_plot.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=layout['legend_ncol'],
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor='0.5',
            fontsize=layout['legend_fontsize'],
            framealpha=1,
            columnspacing=1.2,
            handlelength=1.5,
            handleheight=1.2
        )
        legend.get_frame().set_linewidth(0.6)

        # Estilo minimalista científico
        axis_plot.spines['top'].set_visible(False)
        axis_plot.spines['right'].set_visible(False)
        axis_plot.spines['left'].set_linewidth(0.7)
        axis_plot.spines['bottom'].set_linewidth(0.7)
        axis_plot.spines['left'].set_color('0.5')
        axis_plot.spines['bottom'].set_color('0.5')

        # Ajustar cor dos ticks
        axis_plot.tick_params(axis='both', colors='0.4', which='major')

    def _comparative_metrics_save_or_show_plot(self, file_name):
        """
        Saves or shows the plot based on user preference.

        Parameters
        ----------
        file_name : str
            The base name for the saved plot file.
        """
        if self._comparative_metrics_show_plot:
            plt.show()
        else:
            plt.tight_layout()

            # Salvar em PDF de alta qualidade
            output_path_pdf = f'{file_name}metrics.pdf'
            plt.savefig(
                output_path_pdf,
                dpi=600,
                bbox_inches='tight',
                format='pdf',
                facecolor='white',
                edgecolor='none',
                transparent=False,
                metadata={'Creator': 'Matplotlib', 'Author': __author__}
            )

            # Salvar em PNG de alta resolução
            output_path_png = f'{file_name}metrics.png'
            plt.savefig(
                output_path_png,
                dpi=600,
                bbox_inches='tight',
                format='png',
                facecolor='white',
                edgecolor='none',
                transparent=False
            )

            logging.info(f"Gráfico salvo: {output_path_pdf} e {output_path_png}")

        plt.close()

    def plot_comparative_metrics(self, dictionary_metrics_list, file_name):
        """
        Plots comparative metrics for different models, including standard deviation bars.
        MELHORADO para lidar eficientemente com muitos modelos.

        Parameters
        ----------
        dictionary_metrics_list : list of dict
            List of dictionaries containing model names and metrics.
        file_name : str
            The path where the plot will be saved.
        """
        logging.info(f"Iniciando plot com {len(dictionary_metrics_list)} modelos.")

        try:
            # Extrair dados
            list_metrics, number_models, number_metrics = \
                self._comparative_metrics_extract_metric_data(dictionary_metrics_list)

            # Calcular layout ótimo
            layout = self._calculate_optimal_layout(number_models)

            logging.info(f"Layout calculado: largura={layout['figure_width']}, "
                         f"bar_width={layout['bar_width']}, "
                         f"anotações={'sim' if layout['show_annotations'] else 'não'}")

            # Criar figura
            figure_plot, axis_plot = self._comparative_metrics_create_figure(layout)
            positions = numpy.arange(number_metrics)

            # Plotar barras para cada métrica
            for metric_id, metric_name in enumerate(list_metrics):
                # Gerar paleta de cores para todos os modelos
                base_colors = self._comparative_metrics_list_color_bases[metric_name]
                metric_colors = self._generate_color_palette(base_colors, number_models)

                self._comparative_metrics_plot_bars_for_metric(
                    axis_plot, positions, metric_id, dictionary_metrics_list,
                    metric_name, metric_colors, number_models, layout
                )

            # Configurar labels e legenda
            self._comparative_metrics_set_plot_labels(axis_plot, positions, number_models, layout)

            # Salvar ou exibir
            self._comparative_metrics_save_or_show_plot(file_name)

            logging.info("Plot concluído com sucesso!")

        except Exception as e:
            logging.error(f"Erro ao plotar métricas: {e}")
            raise