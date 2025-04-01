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


DEFAULT_ROC_PLOTTER_FIGURE_SIZE = (8, 6)
DEFAULT_ROC_PLOTTER_LINE_WIDTH = 2
DEFAULT_ROC_PLOTTER_MARKER_STYLE = ''
DEFAULT_ROC_PLOTTER_COLOR_MAP = 'magma'
DEFAULT_ROC_PLOTTER_SHOW_PLOT = False
DEFAULT_ROC_PLOTTER_TITLE_FONT_SIZE = 14
DEFAULT_ROC_PLOTTER_AXIS_LABEL_FONT_SIZE = 12
DEFAULT_ROC_PLOTTER_LEGEND_FONT_SIZE = 10
DEFAULT_ROC_PLOTTER_GRID = True
DEFAULT_ROC_PLOTTER_DIAGONAL_LINE = True



def add_roc_plotter_arguments(parser):

    parser.add_argument('--roc_curve_figure_size', type=tuple, default=DEFAULT_ROC_PLOTTER_FIGURE_SIZE,
                        help='Size of the figure (width, height) in inches'
                        )

    parser.add_argument('--roc_curve_line_width', type=int, default=DEFAULT_ROC_PLOTTER_LINE_WIDTH,
                        help='Width of the lines in the plot'
                        )

    parser.add_argument('--roc_curve_marker_style', type=str, default=DEFAULT_ROC_PLOTTER_MARKER_STYLE,
                        help='Marker style for data points (e.g., "o", "x")'
                        )

    parser.add_argument('--roc_curve_color_map', type=str, default=DEFAULT_ROC_PLOTTER_COLOR_MAP,
                        help='Color map to use for the plot'
                        )

    parser.add_argument('--roc_curve_show_plot', type=bool, default=DEFAULT_ROC_PLOTTER_SHOW_PLOT,
                        help='Whether to display the plot after creation'
                        )

    parser.add_argument('--roc_curve_title_font_size', type=int, default=DEFAULT_ROC_PLOTTER_TITLE_FONT_SIZE,
                        help='Font size for the plot title'
                        )

    parser.add_argument('--roc_curve_axis_label_font_size', type=int,
                        default=DEFAULT_ROC_PLOTTER_AXIS_LABEL_FONT_SIZE,
                        help='Font size for the axis labels'
                        )

    parser.add_argument('--roc_curve_legend_font_size', type=int, default=DEFAULT_ROC_PLOTTER_LEGEND_FONT_SIZE,
                        help='Font size for the plot legend'
                        )

    parser.add_argument('--roc_curve_grid', type=bool, default=DEFAULT_ROC_PLOTTER_GRID,
                        help='Whether to display the grid on the plot'
                        )

    parser.add_argument('--roc_curve_diagonal_line', type=bool, default=DEFAULT_ROC_PLOTTER_DIAGONAL_LINE,
                        help='Whether to include a diagonal line in the plot (e.g., for scatter plots)'
                        )

    return parser
