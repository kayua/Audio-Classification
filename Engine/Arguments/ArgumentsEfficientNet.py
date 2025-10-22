#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/10/22'
__last_update__ = '2025/10/22'
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
    import glob

except ImportError as error:
    print(error)
    sys.exit(-1)


DEFAULT_OVERLAP = 1

DEFAULT_SIZE_BATCH = 32
DEFAULT_HOP_LENGTH = 256

DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_SAMPLE_RATE = 8000

DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

DEFAULT_INPUT_DIMENSION = (513, 40, 1)
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_EFFICIENTNET_VERSION = 'B0'  # Can be B0, B1, B2, B3, B4, B5, B6, B7
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


def add_efficientnet_arguments(parser):

    parser.add_argument('--efficientnet_optimizer_function', type=str, default=DEFAULT_OPTIMIZER_FUNCTION,
                        help='Optimizer to use during training (default: adam)'
                        )

    parser.add_argument('--efficientnet_loss_function', type=str, default=DEFAULT_LOSS_FUNCTION,
                        help='Loss function to use during training (default: sparse_categorical_crossentropy)'
                        )

    parser.add_argument('--efficientnet_input_dimension', default=DEFAULT_INPUT_DIMENSION,
                        help='Dimensions of the input data (height, width, channels)'
                        )

    parser.add_argument('--efficientnet_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT (default: 256)'
                        )

    parser.add_argument('--efficientnet_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size (default: 40)'
                        )

    parser.add_argument('--efficientnet_number_filters_spectrogram', type=int, default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                        help='Number of filters for spectrogram generation (default: 512)'
                        )

    parser.add_argument('--efficientnet_version', type=str, default=DEFAULT_EFFICIENTNET_VERSION,
                        help='EfficientNet version to use: B0, B1, B2, B3, B4, B5, B6, B7 (default: B0)'
                        )

    parser.add_argument('--efficientnet_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network (default: 0.2)'
                        )

    parser.add_argument('--efficientnet_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram (default: 1)'
                        )

    parser.add_argument('--efficientnet_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels (default: 80)'
                        )

    parser.add_argument('--efficientnet_last_layer_activation', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer (default: softmax)'
                        )

    parser.add_argument('--efficientnet_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window (default: 1024)'
                        )

    parser.add_argument('--efficientnet_use_pretrained', type=bool, default=False,
                        help='Use pretrained ImageNet weights (default: False)'
                        )

    parser.add_argument('--efficientnet_fine_tune', type=bool, default=False,
                        help='Fine-tune the pretrained model (default: False)'
                        )

    parser.add_argument('--efficientnet_freeze_layers', type=int, default=0,
                        help='Number of layers to freeze from the beginning (default: 0)'
                        )

    return parser