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
DEFAULT_INTERMEDIARY_ACTIVATION = 'hard_swish'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_SIZE_CONVOLUTIONAL_FILTERS = (3, 3)
DEFAULT_FILTERS_PER_BLOCK = [16, 24, 40, 80, 112, 160]
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'
DEFAULT_MOBILENETV3_VARIANT = 'large'  # 'large' or 'small'

# MobileNetV3-Large configuration
# (kernel, exp_size, out_channels, SE, NL, stride)
# SE: use Squeeze-and-Excitation, NL: activation (RE=ReLU, HS=hard-swish)
DEFAULT_BNECK_BLOCKS_LARGE = [
    # kernel, exp, out, SE, NL, stride
    (3, 16, 16, False, 'RE', 1),
    (3, 64, 24, False, 'RE', 2),
    (3, 72, 24, False, 'RE', 1),
    (5, 72, 40, True, 'RE', 2),
    (5, 120, 40, True, 'RE', 1),
    (5, 120, 40, True, 'RE', 1),
    (3, 240, 80, False, 'HS', 2),
    (3, 200, 80, False, 'HS', 1),
    (3, 184, 80, False, 'HS', 1),
    (3, 184, 80, False, 'HS', 1),
    (3, 480, 112, True, 'HS', 1),
    (3, 672, 112, True, 'HS', 1),
    (5, 672, 160, True, 'HS', 2),
    (5, 960, 160, True, 'HS', 1),
    (5, 960, 160, True, 'HS', 1),
]

# MobileNetV3-Small configuration
DEFAULT_BNECK_BLOCKS_SMALL = [
    # kernel, exp, out, SE, NL, stride
    (3, 16, 16, True, 'RE', 2),
    (3, 72, 24, False, 'RE', 2),
    (3, 88, 24, False, 'RE', 1),
    (5, 96, 40, True, 'HS', 2),
    (5, 240, 40, True, 'HS', 1),
    (5, 240, 40, True, 'HS', 1),
    (5, 120, 48, True, 'HS', 1),
    (5, 144, 48, True, 'HS', 1),
    (5, 288, 96, True, 'HS', 2),
    (5, 576, 96, True, 'HS', 1),
    (5, 576, 96, True, 'HS', 1),
]


def add_mobilenetv3_arguments(parser):

    parser.add_argument('--mobilenetv3_variant', type=str, default=DEFAULT_MOBILENETV3_VARIANT,
                        choices=['large', 'small'],
                        help='MobileNetV3 variant: large or small'
                        )

    parser.add_argument('--mobilenetv3_optimizer_function', type=str, default="adam",
                        help='Optimizer to use during training'
                        )

    parser.add_argument('--mobilenetv3_loss_function', type=str, default='sparse_categorical_crossentropy',
                        help='Loss function to use during training'
                        )

    parser.add_argument('--mobilenetv3_input_dimension', default=DEFAULT_INPUT_DIMENSION,
                        help='Dimensions of the input data (height, width)'
                        )

    parser.add_argument('--mobilenetv3_hop_length', type=int, default=DEFAULT_HOP_LENGTH,
                        help='Hop length for STFT'
                        )

    parser.add_argument('--mobilenetv3_window_size_factor', type=int, default=DEFAULT_WINDOW_SIZE_FACTOR,
                        help='Factor applied to FFT window size'
                        )

    parser.add_argument('--mobilenetv3_number_filters_spectrogram', type=int, default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                        help='Number of filters for spectrogram generation'
                        )

    parser.add_argument('--mobilenetv3_filters_per_block', default=DEFAULT_FILTERS_PER_BLOCK,
                        help='Number of filters in each bottleneck block'
                        )

    parser.add_argument('--mobilenetv3_dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE,
                        help='Dropout rate in the network'
                        )

    parser.add_argument('--mobilenetv3_alpha', type=float, default=DEFAULT_ALPHA,
                        help='Width multiplier for MobileNetV3 (controls network width)'
                        )

    parser.add_argument('--mobilenetv3_expansion_factor', type=int, default=DEFAULT_EXPANSION_FACTOR,
                        help='Expansion factor for bottleneck blocks'
                        )

    parser.add_argument('--mobilenetv3_number_layers', type=int, default=DEFAULT_NUMBER_LAYERS,
                        help='Number of convolutional layers'
                        )

    parser.add_argument('--mobilenetv3_overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap between patches in the spectrogram'
                        )

    parser.add_argument('--mobilenetv3_decibel_scale_factor', type=float, default=DEFAULT_DECIBEL_SCALE_FACTOR,
                        help='Scale factor for converting to decibels'
                        )

    parser.add_argument('--mobilenetv3_convolutional_padding', type=str, default=DEFAULT_CONVOLUTIONAL_PADDING,
                        help='Padding type for convolutional layers'
                        )

    parser.add_argument('--mobilenetv3_intermediary_activation', type=str, default=DEFAULT_INTERMEDIARY_ACTIVATION,
                        help='Activation function for intermediary layers (hard-swish recommended for MobileNetV3)'
                        )

    parser.add_argument('--mobilenetv3_last_layer_activation', type=str, default=DEFAULT_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer'
                        )

    parser.add_argument('--mobilenetv3_window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Size of the FFT window'
                        )

    parser.add_argument('--mobilenetv3_size_convolutional_filters', type=tuple, default=DEFAULT_SIZE_CONVOLUTIONAL_FILTERS,
                        help='Size of the convolutional filters'
                        )

    parser.add_argument('--mobilenetv3_bneck_blocks_large', default=DEFAULT_BNECK_BLOCKS_LARGE,
                        help='Configuration for MobileNetV3-Large bottleneck blocks'
                        )

    parser.add_argument('--mobilenetv3_bneck_blocks_small', default=DEFAULT_BNECK_BLOCKS_SMALL,
                        help='Configuration for MobileNetV3-Small bottleneck blocks'
                        )

    return parser