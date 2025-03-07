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
    import glob
    import logging

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


DEFAULT_OVERLAP = 2

DEFAULT_SIZE_BATCH = 32
DEFAULT_HOP_LENGTH = 256

DEFAULT_NUMBER_SPLITS = 5
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SAMPLE_RATE = 8000

DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = '*.wav'
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_INPUT_DIMENSION = (40, 256)

DEFAULT_LIST_DENSE_NEURONS = [128, 129]
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'relu'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


def add_mlp_arguments(parser):

    parser.add_argument('--mlp_list_dense_neurons', default=DEFAULT_LIST_DENSE_NEURONS,
                        help='List of LSTM cell sizes for each layer'
                        )

    parser.add_argument('--mlp_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--mlp_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--mlp_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--mlp_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window'
                        )

    parser.add_argument('--mlp_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--mlp_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--mlp_last_layer_activation', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer'
                        )

    parser.add_argument('--mlp_intermediary_layer_activation', type=str, default=DEFAULT_INTERMEDIARY_LAYER_ACTIVATION,
                        help='Activation function for intermediary layers'
                        )

    return parser