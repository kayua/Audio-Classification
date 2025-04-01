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


DEFAULT_LOSS_PLOTTER_FIGURE_SIZE = (10, 6)
DEFAULT_LOSS_PLOTTER_TRAINING_LOSS_COLOR = "blue"
DEFAULT_LOSS_PLOTTER_VALIDATION_LOSS_COLOR = "orange"
DEFAULT_LOSS_PLOTTER_TITLE_FONT_SIZE = 16
DEFAULT_LOSS_PLOTTER_AXIS_FONT_SIZE = 12
DEFAULT_LOSS_PLOTTER_LEGEND_FONT_SIZE = 10
DEFAULT_LOSS_PLOTTER_X_LABEL = 'Epochs'
DEFAULT_LOSS_PLOTTER_Y_LABEL = 'Loss'
DEFAULT_LOSS_PLOTTER_TITLE = 'Loss Graph'
DEFAULT_LOSS_PLOTTER_GRID = True
DEFAULT_LOSS_PLOTTER_LINE_STYLE_TRAINING = '-'
DEFAULT_LOSS_PLOTTER_LINE_STYLE_VALIDATION = '--'
DEFAULT_LOSS_PLOTTER_LINE_WIDTH = 2


def add_loss_plotter_arguments(parser):

    parser.add_argument('--loss_curve_figure_size', type=tuple, default=DEFAULT_LOSS_PLOTTER_FIGURE_SIZE,
                        help='Size of the figure (width, height) in inches'
                        )

    parser.add_argument('--loss_curve_training_loss_color', type=str,
                        default=DEFAULT_LOSS_PLOTTER_TRAINING_LOSS_COLOR,
                        help='Color of the training loss curve'
                        )

    parser.add_argument('--loss_curve_validation_loss_color', type=str,
                        default=DEFAULT_LOSS_PLOTTER_VALIDATION_LOSS_COLOR,
                        help='Color of the validation loss curve'
                        )

    parser.add_argument('--loss_curve_title_font_size', type=int, default=DEFAULT_LOSS_PLOTTER_TITLE_FONT_SIZE,
                        help='Font size for the plot title'
                        )

    parser.add_argument('--loss_curve_axis_font_size', type=int, default=DEFAULT_LOSS_PLOTTER_AXIS_FONT_SIZE,
                        help='Font size for the axis labels'
                        )

    parser.add_argument('--loss_curve_legend_font_size', type=int, default=DEFAULT_LOSS_PLOTTER_LEGEND_FONT_SIZE,
                        help='Font size for the legend'
                        )

    parser.add_argument('--loss_curve_x_label', type=str, default=DEFAULT_LOSS_PLOTTER_X_LABEL,
                        help='Label for the x-axis'
                        )

    parser.add_argument('--loss_curve_y_label', type=str, default=DEFAULT_LOSS_PLOTTER_Y_LABEL,
                        help='Label for the y-axis'
                        )

    parser.add_argument('--loss_curve_title', type=str, default=DEFAULT_LOSS_PLOTTER_TITLE,
                        help='Title of the plot'
                        )

    parser.add_argument('--loss_curve_grid', type=bool, default=DEFAULT_LOSS_PLOTTER_GRID,
                        help='Whether to display the grid on the plot'
                        )

    parser.add_argument('--loss_curve_line_style_training', type=str,
                        default=DEFAULT_LOSS_PLOTTER_LINE_STYLE_TRAINING,
                        help='Line style for the training loss curve'
                        )

    parser.add_argument('--loss_curve_line_style_validation', type=str,
                        default=DEFAULT_LOSS_PLOTTER_LINE_STYLE_VALIDATION,
                        help='Line style for the validation loss curve'
                        )

    parser.add_argument('--loss_curve_line_width', type=int, default=DEFAULT_LOSS_PLOTTER_LINE_WIDTH,
                        help='Line width for the curves'
                        )

    return parser