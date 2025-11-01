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

DEFAULT_ALPHA = 1.0
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUMBER_LAYERS = 4
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_SAMPLE_RATE = 8000

DEFAULT_EXPANSION_FACTOR = 6
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

DEFAULT_CONVOLUTIONAL_PADDING = 'same'
DEFAULT_INPUT_DIMENSION = (513, 40, 1)
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu6'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_SIZE_CONVOLUTIONAL_FILTERS = (3, 3)
DEFAULT_FILTERS_PER_BLOCK = [8, 12, 16, 24, 32, 64, 128]
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'
DEFAULT_INVERTED_RESIDUAL_BLOCKS = [
    # (expansion_factor, output_channels, num_blocks, stride)
    (1, 12, 1, 1),
    (6, 16, 2, 2),
    (6, 24, 3, 2),
    (6, 32, 4, 2),
    (6, 64, 3, 1),
    (6, 128, 3, 2),
    (6, 128, 1, 1),
]


def add_mobilenetv2_arguments(parser):

    parser.add_argument('--mobilenetv2_optimizer_function', type=str, default="adam",
                        help='Optimizer to use during training'
                        )

    parser.add_argument('--mobilenetv2_loss_function', type=str, default='sparse_categorical_crossentropy',
                        help='Loss function to use during training'
                        )

    parser.add_argument('--mobilenetv2_input_dimension', default=DEFAULT_INPUT_DIMENSION,
                        help='Dimensions of the input data (height, width)'
                        )

    parser.add_argument('--mobilenetv2_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--mobilenetv2_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--mobilenetv2_number_filters_spectrogram', type=int, default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                        help='Number of filters for spectrogram generation'
                        )

    parser.add_argument('--mobilenetv2_filters_per_block', default=DEFAULT_FILTERS_PER_BLOCK,
                        help='Number of filters in each inverted residual block'
                        )

    parser.add_argument('--mobilenetv2_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--mobilenetv2_alpha', type=float, default=DEFAULT_ALPHA,
                        help='Width multiplier for MobileNetV2 (controls network width)'
                        )

    parser.add_argument('--mobilenetv2_expansion_factor', type=int, default=DEFAULT_EXPANSION_FACTOR,
                        help='Expansion factor for inverted residual blocks'
                        )

    parser.add_argument('--mobilenetv2_number_layers', type=int, default=DEFAULT_NUMBER_LAYERS,
                        help='Number of convolutional layers'
                        )

    parser.add_argument('--mobilenetv2_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--mobilenetv2_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--mobilenetv2_convolutional_padding', type=str, default=DEFAULT_CONVOLUTIONAL_PADDING,
                        help='Padding type for convolutional layers'
                        )

    parser.add_argument('--mobilenetv2_intermediary_activation', type=str, default=DEFAULT_INTERMEDIARY_ACTIVATION,
                        help='Activation function for intermediary layers (ReLU6 recommended for MobileNetV2)'
                        )

    parser.add_argument('--mobilenetv2_last_layer_activation', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer'
                        )

    parser.add_argument('--mobilenetv2_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window'
                        )

    parser.add_argument('--mobilenetv2_size_convolutional_filters', type=tuple, default=DEFAULT_SIZE_CONVOLUTIONAL_FILTERS,
                        help='Size of the convolutional filters'
                        )

    parser.add_argument('--mobilenetv2_inverted_residual_blocks', default=DEFAULT_INVERTED_RESIDUAL_BLOCKS,
                        help='Configuration for inverted residual blocks: (expansion, channels, blocks, stride)'
                        )

    return parser

