#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

try:
    import os
    import sys
    import logging

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_OVERLAP = 2
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SIZE_KERNEL = 3
DEFAULT_SIZE_BATCH = 32
DEFAULT_HOP_LENGTH = 256
DEFAULT_NUMBER_HEADS = 4
DEFAULT_MAX_LENGTH = 100

DEFAULT_NUMBER_SPLITS = 5
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_DROPOUT_DECAY = 0.2

DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_EMBEDDING_DIMENSION = 64
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_INPUT_DIMENSION = (80, 40)
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

DEFAULT_NUMBER_CONFORMER_BLOCKS = 4
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 80
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


def add_conformer_arguments(parser):

    parser.add_argument('--conformer_number_conformer_blocks', type=int, default=DEFAULT_NUMBER_CONFORMER_BLOCKS,
                        help='Number of conformer blocks'
                        )

    parser.add_argument('--conformer_embedding_dimension', type=int, default=DEFAULT_EMBEDDING_DIMENSION,
                        help='Dimension of embedding layer'
                        )

    parser.add_argument('--conformer_number_heads', type=int, default=DEFAULT_NUMBER_HEADS,
                        help='Number of heads in multi-head attention'
                        )

    parser.add_argument('--conformer_size_kernel', type=int, default=DEFAULT_SIZE_KERNEL,
                        help='Size of convolution kernel'
                        )

    parser.add_argument('--conformer_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--conformer_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--conformer_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--conformer_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window'
                        )

    parser.add_argument('--conformer_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--conformer_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--conformer_number_filters_spectrogram', type=int, default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                        help='Number of filters in the spectrogram'
                        )

    return parser