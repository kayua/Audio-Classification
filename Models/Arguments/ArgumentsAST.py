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
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default constants for the Audio Classification Model
DEFAULT_WINDOW_SIZE_FACTOR = 40 # Window expansion factor
DEFAULT_PROJECTION_DIMENSION = 64  # Dimension of the linear projection
DEFAULT_HEAD_SIZE = 256  # Size of each attention head
DEFAULT_NUMBER_HEADS = 2  # Number of attention heads
DEFAULT_NUMBER_BLOCKS = 2  # Number of transformer encoder blocks
DEFAULT_NUMBER_CLASSES = 4  # Number of output classes for classification
DEFAULT_SAMPLE_RATE = 8000  # Sample rate for loading audio
DEFAULT_NUMBER_FILTERS = 128  # Number of filters for the Mel spectrogram
DEFAULT_HOP_LENGTH = 512  # Hop length for the Mel spectrogram
DEFAULT_SIZE_FFT = 1024  # FFT size for the Mel spectrogram
DEFAULT_SIZE_PATCH = (16, 16)  # Size of the patches to be extracted from the spectrogram
DEFAULT_OVERLAP = 2  # Overlap ratio between patches
DEFAULT_DROPOUT_RATE = 0.2  # Dropout rate
DEFAULT_NUMBER_EPOCHS = 10  # Number of training epochs
DEFAULT_SIZE_BATCH = 32  # Batch size for training
DEFAULT_NUMBER_SPLITS = 5  # Number of splits for cross-validation
DEFAULT_NORMALIZATION_EPSILON = 1e-6  # Epsilon value for layer normalization
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'  # Activation function for intermediary layers
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'  # Activation function for the output layer
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'  # Loss function for model compilation
DEFAULT_OPTIMIZER_FUNCTION = 'adam'  # Optimizer function for model compilation
DEFAULT_FILE_EXTENSION = "*.wav"  # File format for sound files
DEFAULT_AUDIO_DURATION = 10  # Duration of audio to be considered
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512


def get_audio_ast_arguments(parser):

    parser.add_argument('--ast_projection_dimension', type=int,
                        default=DEFAULT_PROJECTION_DIMENSION, help='Dimension for projection layer')

    parser.add_argument('--ast_head_size', type=int,
                        default=DEFAULT_HEAD_SIZE, help='Size of each head in multi-head attention')

    parser.add_argument('--ast_number_heads', type=int,
                        default=DEFAULT_NUMBER_HEADS, help='Number of heads in multi-head attention')

    parser.add_argument('--ast_number_blocks', type=int,
                        default=DEFAULT_NUMBER_BLOCKS, help='Number of transformer blocks')

    parser.add_argument('--ast_hop_length', type=int,
                        default=DEFAULT_HOP_LENGTH, help='Hop length for STFT')

    parser.add_argument('--ast_size_fft', type=int,
                        default=DEFAULT_SIZE_FFT, help='Size of FFT window')

    parser.add_argument('--ast_patch_size', type=tuple,
                        default=DEFAULT_SIZE_PATCH, help='Size of the patches in the spectrogram')

    parser.add_argument('--ast_overlap', type=int,
                        default=DEFAULT_OVERLAP, help='Overlap between patches in the spectrogram')

    parser.add_argument('--ast_dropout', type=float,
                        default=DEFAULT_DROPOUT_RATE, help='Dropout rate in the network')

    parser.add_argument('--ast_intermediary_activation', type=str,
                        default=DEFAULT_INTERMEDIARY_ACTIVATION, help='Activation function for intermediary layers')

    parser.add_argument('--ast_last_activation_layer', type=str,
                        default=DEFAULT_LAST_LAYER_ACTIVATION, help='Activation function for the last layer')

    parser.add_argument('--ast_normalization_epsilon', type=float,
                        default=DEFAULT_NORMALIZATION_EPSILON, help='Epsilon value for normalization layers')

    parser.add_argument('--ast_decibel_scale_factor', type=float,
                        default=DEFAULT_DECIBEL_SCALE_FACTOR, help='Scale factor for converting to decibels')

    parser.add_argument('--ast_window_size_fft', type=int,
                        default=DEFAULT_SIZE_FFT, help='Size of the FFT window for spectral analysis')

    parser.add_argument('--ast_window_size_factor', type=float,
                        default=DEFAULT_WINDOW_SIZE_FACTOR, help='Factor applied to FFT window size')

    parser.add_argument('--ast_number_filters_spectrogram', type=int,
                        default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM, help='Number of filters in the spectrogram')

    return parser