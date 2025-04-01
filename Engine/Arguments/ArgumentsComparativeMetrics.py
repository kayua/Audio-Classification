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


DEFAULT_COMPARATIVE_METRICS_FIGURE_WIDTH = 12
DEFAULT_COMPARATIVE_METRICS_FIGURE_HEIGHT = 8
DEFAULT_COMPARATIVE_METRICS_BAR_WIDTH = 0.20
DEFAULT_COMPARATIVE_METRICS_CAPTION_SIZE = 10
DEFAULT_COMPARATIVE_METRICS_SHOW_PLOT = False


def add_comparative_metrics_plotter_arguments(parser):

    parser.add_argument('--comparative_metrics_figure_width', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_FIGURE_WIDTH, help='Width of the figure in inches'
                        )

    parser.add_argument('--comparative_metrics_figure_height', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_FIGURE_HEIGHT, help='Height of the figure in inches'
                        )

    parser.add_argument('--comparative_metrics_bar_width', type=float,
                        default=DEFAULT_COMPARATIVE_METRICS_BAR_WIDTH, help='Width of the bars in a bar plot'
                        )

    parser.add_argument('--comparative_metrics_caption_size', type=int,
                        default=DEFAULT_COMPARATIVE_METRICS_CAPTION_SIZE, help='Font size for captions in the plot'
                        )

    parser.add_argument('--comparative_metrics_show_plot', type=bool,
                        default=DEFAULT_COMPARATIVE_METRICS_SHOW_PLOT, help='Whether to display the plot after creation'
                        )

    return parser