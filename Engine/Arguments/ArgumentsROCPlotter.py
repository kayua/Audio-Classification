#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


DEFAULT_ROC_PLOTTER_FIGURE_SIZE = (8, 6)
DEFAULT_ROC_PLOTTER_LINE_WIDTH = 2
DEFAULT_ROC_PLOTTER_MARKER_STYLE = 'o'
DEFAULT_ROC_PLOTTER_COLOR_MAP = 'Blue'
DEFAULT_ROC_PLOTTER_SHOW_PLOT = True
DEFAULT_ROC_PLOTTER_TITLE_FONT_SIZE = 14
DEFAULT_ROC_PLOTTER_AXIS_LABEL_FONT_SIZE = 12
DEFAULT_ROC_PLOTTER_LEGEND_FONT_SIZE = 10
DEFAULT_ROC_PLOTTER_GRID = True
DEFAULT_ROC_PLOTTER_DIAGONAL_LINE = True



def add_roc_plotter_arguments(parser):

    parser.add_argument('--roc_plotter_figure_size', type=tuple, default=DEFAULT_ROC_PLOTTER_FIGURE_SIZE,
                        help='Size of the figure (width, height) in inches'
                        )

    parser.add_argument('--roc_plotter_line_width', type=int, default=DEFAULT_ROC_PLOTTER_LINE_WIDTH,
                        help='Width of the lines in the plot'
                        )

    parser.add_argument('--roc_plotter_marker_style', type=str, default=DEFAULT_ROC_PLOTTER_MARKER_STYLE,
                        help='Marker style for data points (e.g., "o", "x")'
                        )

    parser.add_argument('--roc_plotter_color_map', type=str, default=DEFAULT_ROC_PLOTTER_COLOR_MAP,
                        help='Color map to use for the plot'
                        )

    parser.add_argument('--roc_plotter_show_plot', type=bool, default=DEFAULT_ROC_PLOTTER_SHOW_PLOT,
                        help='Whether to display the plot after creation'
                        )

    parser.add_argument('--roc_plotter_title_font_size', type=int, default=DEFAULT_ROC_PLOTTER_TITLE_FONT_SIZE,
                        help='Font size for the plot title'
                        )

    parser.add_argument('--roc_plotter_axis_label_font_size', type=int,
                        default=DEFAULT_ROC_PLOTTER_AXIS_LABEL_FONT_SIZE,
                        help='Font size for the axis labels'
                        )

    parser.add_argument('--roc_plotter_legend_font_size', type=int, default=DEFAULT_ROC_PLOTTER_LEGEND_FONT_SIZE,
                        help='Font size for the plot legend'
                        )

    parser.add_argument('--roc_plotter_grid', type=bool, default=DEFAULT_ROC_PLOTTER_GRID,
                        help='Whether to display the grid on the plot'
                        )

    parser.add_argument('--roc_plotter_diagonal_line', type=bool, default=DEFAULT_ROC_PLOTTER_DIAGONAL_LINE,
                        help='Whether to include a diagonal line in the plot (e.g., for scatter plots)'
                        )

    return parser
