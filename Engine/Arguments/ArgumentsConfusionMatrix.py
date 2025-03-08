#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


DEFAULT_CONFUSION_MATRIX_FIGURE_SIZE = (5, 5)
DEFAULT_CONFUSION_MATRIX_COLOR_MAP = 'Blues'
DEFAULT_CONFUSION_MATRIX_ANNOT_FONT_SIZE = 10
DEFAULT_CONFUSION_MATRIX_LABEL_FONT_SIZE = 12
DEFAULT_CONFUSION_MATRIX_TITLE_FONT_SIZE = 14
DEFAULT_CONFUSION_MATRIX_SHOW_PLOT = False
DEFAULT_CONFUSION_MATRIX_COLOR_BAR = True
DEFAULT_CONFUSION_MATRIX_ANNOT_KWS = None
DEFAULT_CONFUSION_MATRIX_FMT = 'd'
DEFAULT_CONFUSION_MATRIX_ROTATION = 45


def add_confusion_matrix_arguments(parser):

    parser.add_argument('--confusion_matrix_figure_size', type=tuple, default=DEFAULT_CONFUSION_MATRIX_FIGURE_SIZE,
                        help='Size of the figure (width, height) in inches'
                        )

    parser.add_argument('--confusion_matrix_color_map', type=str, default=DEFAULT_CONFUSION_MATRIX_COLOR_MAP,
                        help='Color map to use for the plot (e.g., "Blues", "viridis")'
                        )

    parser.add_argument('--confusion_matrix_annot_font_size', type=int, default=DEFAULT_CONFUSION_MATRIX_ANNOT_FONT_SIZE,
                        help='Font size for annotations on the plot'
                        )

    parser.add_argument('--confusion_matrix_label_font_size', type=int, default=DEFAULT_CONFUSION_MATRIX_LABEL_FONT_SIZE,
                        help='Font size for labels on the plot'
                        )

    parser.add_argument('--confusion_matrix_title_font_size', type=int, default=DEFAULT_CONFUSION_MATRIX_TITLE_FONT_SIZE,
                        help='Font size for the plot title'
                        )

    parser.add_argument('--confusion_matrix_show_plot', type=bool, default=DEFAULT_CONFUSION_MATRIX_SHOW_PLOT,
                        help='Whether to display the plot after creation'
                        )

    parser.add_argument('--confusion_matrix_color_bar', type=bool, default=DEFAULT_CONFUSION_MATRIX_COLOR_BAR,
                        help='Whether to display a color bar'
                        )

    parser.add_argument('--confusion_matrix_annot_kws', type=dict, default=DEFAULT_CONFUSION_MATRIX_ANNOT_KWS,
                        help='Additional keyword arguments for annotations'
                        )

    parser.add_argument('--confusion_matrix_fmt', type=str, default=DEFAULT_CONFUSION_MATRIX_FMT,
                        help='Format string for annotations (e.g., "d", ".2f")'
                        )

    parser.add_argument('--confusion_matrix_rotation', type=int, default=DEFAULT_CONFUSION_MATRIX_ROTATION,
                        help='Rotation angle for the labels'
                        )

    return parser