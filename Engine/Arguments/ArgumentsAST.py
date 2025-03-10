#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import sys

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_OVERLAP = 2

DEFAULT_SIZE_BATCH = 8
DEFAULT_SIZE_FFT = 1024
DEFAULT_HEAD_SIZE = 64
DEFAULT_NUMBER_HEADS = 2
DEFAULT_HOP_LENGTH = 512

DEFAULT_NUMBER_BLOCKS = 2
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_AUDIO_DURATION = 10
DEFAULT_NUMBER_FILTERS = 64

DEFAULT_SIZE_PATCH = (16, 16)
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = '*.wav'
DEFAULT_PROJECTION_DIMENSION = 16
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_NORMALIZATION_EPSILON = 1e-6
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


def add_ast_arguments(parser):

    parser.add_argument('--ast_projection_dimension', type=int, default=DEFAULT_PROJECTION_DIMENSION,
                        help='Dimension for projection layer'
                        )
    parser.add_argument('--ast_optimizer_function', type=str, default="adam",
                                 help='Optimizer to use during training'
                                 )
    parser.add_argument('--ast_loss_function', type=str, default='sparse_categorical_crossentropy',
                                 help='Loss function to use during training')

    parser.add_argument('--ast_head_size', type=int, default=DEFAULT_HEAD_SIZE,
                        help='Size of each head in multi-head attention'
                        )

    parser.add_argument('--ast_number_heads', type=int, default=DEFAULT_NUMBER_HEADS,
                        help='Number of heads in multi-head attention'
                        )

    parser.add_argument('--ast_number_blocks', type=int, default=DEFAULT_NUMBER_BLOCKS,
                        help='Number of transformer blocks'
                        )

    parser.add_argument('--ast_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--ast_size_fft', type=int, default=DEFAULT_SIZE_FFT,
                        help='Size of FFT window'
                        )

    parser.add_argument('--ast_patch_size', type=tuple, default=DEFAULT_SIZE_PATCH,
                        help='Size of the patches in the spectrogram'
                        )

    parser.add_argument('--ast_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--ast_dropout', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--ast_intermediary_activation', type=str, default=DEFAULT_INTERMEDIARY_ACTIVATION,
                        help='Activation function for intermediary layers'
                        )

    parser.add_argument('--ast_last_activation_layer', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer'
                        )

    parser.add_argument('--ast_normalization_epsilon', type=float, default=DEFAULT_NORMALIZATION_EPSILON,
                        help='Epsilon value for normalization layers'
                        )

    parser.add_argument('--ast_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--ast_window_size_fft', type=int, default=DEFAULT_SIZE_FFT,
                        help='Size of the FFT window for spectral analysis'
                        )

    parser.add_argument('--ast_window_size_factor', type=float, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--ast_number_filters_spectrogram', type=int, default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                        help='Number of filters in the spectrogram'
                        )

    return parser