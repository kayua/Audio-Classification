#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


DEFAULT_COMPARATIVE_METRICS_FIGURE_WIDTH = 12
DEFAULT_COMPARATIVE_METRICS_FIGURE_HEIGHT = 8
DEFAULT_COMPARATIVE_METRICS_BAR_WIDTH = 0.20
DEFAULT_COMPARATIVE_METRICS_CAPTION_SIZE = 10
DEFAULT_COMPARATIVE_METRICS_SHOW_PLOT = False


def add_comparative_metrics_plotter_arguments(parser):

    parser.add_argument('--comparative_metrics_plotter_figure_width', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_FIGURE_WIDTH, help='Width of the figure in inches'
                        )

    parser.add_argument('--comparative_metrics_plotter_figure_height', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_FIGURE_HEIGHT, help='Height of the figure in inches'
                        )

    parser.add_argument('--comparative_metrics_plotter_bar_width', type=float,
                        default=DEFAULT_COMPARATIVE_METRICS_BAR_WIDTH, help='Width of the bars in a bar plot'
                        )

    parser.add_argument('--comparative_metrics_plotter_caption_size', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_CAPTION_SIZE, help='Font size for captions in the plot'
                        )

    parser.add_argument('--comparative_metrics_plotter_show_plot', type=bool,
                        default=DEFAULT_COMPARATIVE_METRICS_SHOW_PLOT, help='Whether to display the plot after creation'
                        )

    return parser