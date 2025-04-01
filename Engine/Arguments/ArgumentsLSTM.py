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


try:
    import os
    import sys

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_OVERLAP = 1
DEFAULT_SIZE_BATCH = 32
DEFAULT_HOP_LENGTH = 256
DEFAULT_NUMBER_SPLITS = 5

DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10

DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_INPUT_DIMENSION = (40, 256)
DEFAULT_LIST_LSTM_CELLS = [128, 129]

DEFAULT_RECURRENT_ACTIVATION = 'sigmoid'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'tanh'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


def add_lstm_arguments(parser):

    parser.add_argument('--lstm_list_lstm_cells', default=DEFAULT_LIST_LSTM_CELLS,
                        help='List of LSTM cell sizes for each layer'
                        )

    parser.add_argument('--lstm_loss_function', type=str, default='sparse_categorical_crossentropy',
                                 help='Loss function to use during training')

    parser.add_argument('--lstm_optimizer_function', type=str, default="adam",
                                 help='Optimizer to use during training'
                                 )

    parser.add_argument('--lstm_input_dimension', default=DEFAULT_INPUT_DIMENSION,
                        help='Dimensions of the input data (height, width)'
                        )

    parser.add_argument('--lstm_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--lstm_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--lstm_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--lstm_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window'
                        )

    parser.add_argument('--lstm_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--lstm_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--lstm_last_layer_activation', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer'
                        )

    parser.add_argument('--lstm_recurrent_activation', type=str, default=DEFAULT_RECURRENT_ACTIVATION,
                        help='Activation function for LSTM recurrent step'
                        )

    parser.add_argument('--lstm_intermediary_layer_activation', type=str, default=DEFAULT_INTERMEDIARY_LAYER_ACTIVATION,
                        help='Activation function for intermediary layers'
                        )


    return parser