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
    import sys

    import logging
    import argparse

    from Engine.Arguments.ArgumentsAST import add_ast_arguments
    from Engine.Arguments.ArgumentsMLP import add_mlp_arguments

    from Engine.Arguments.ArgumentsLSTM import add_lstm_arguments

    from Engine.Arguments.ArgumentsResidual import add_residual_arguments

    from Engine.Arguments.ArgumentsConformer import add_conformer_arguments
    from Engine.Arguments.ArgumentsWav2Vec2 import add_wav_to_vec_arguments

    from Engine.Arguments.ArgumentsROCPlotter import add_roc_plotter_arguments

    from Engine.Arguments.ArgumentsLossPlotter import add_loss_plotter_arguments

    from Engine.Arguments.ArgumentsConfusionMatrix import add_confusion_matrix_arguments

    from Engine.Arguments.ArgumentsComparativeMetrics import add_comparative_metrics_plotter_arguments

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_MATRIX_FIGURE_SIZE = (5, 5)
DEFAULT_MATRIX_COLOR_MAP = 'Blues'
DEFAULT_MATRIX_ANNOTATION_FONT_SIZE = 10
DEFAULT_MATRIX_LABEL_FONT_SIZE = 12
DEFAULT_MATRIX_TITLE_FONT_SIZE = 14
DEFAULT_SHOW_PLOT = False
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_OVERLAP = 1
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_PLOT_WIDTH = 14
DEFAULT_PLOT_HEIGHT = 8
DEFAULT_PLOT_BAR_WIDTH = 0.15
DEFAULT_PLOT_CAP_SIZE = 10
DEFAULT_OUTPUT_DIRECTORY = "Results/"
DEFAULT_DATASET_DIRECTORY = "Dataset/"
DEFAULT_LOSS = 'sparse_categorical_crossentropy'

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}





class Arguments:
    """
    Arguments Class

    This class manages the configuration and command-line arguments for a machine learning training
    and evaluation pipeline. It collects arguments from multiple subsystems (AST, Conformer, LSTM,
    MLP, Residual, Wav2Vec) and consolidates them into a single namespace.

    The class also handles argument parsing, logging of all settings, and provides a static method
    to define the core arguments related to dataset handling, training parameters, plotting, and
    logging verbosity.

    Methods
    -------
    @__init__() :
        Initializes the argument parser, adds arguments from multiple components, parses the
        command-line arguments, and logs the final settings.

    @show_all_settings() :
        Logs the parsed arguments in a formatted manner for better visibility, including
        the command used to launch the script.

    @get_arguments() :
        Static method that defines the base arguments required for dataset handling, training,
        plotting, and logging.

    Notes
    -----
    This class relies on external functions to append arguments related to specific model types
    (AST, Conformer, LSTM, MLP, Residual, Wav2Vec). These functions are expected to extend the
    `ArgumentParser` instance with additional arguments specific to each model architecture.
    """

    def __init__(self):
        """
        Initializes the Arguments class.

        This constructor:
            1. Initializes a base ArgumentParser with common arguments.
            2. Sequentially adds arguments from different model components.
            3. Parses the command-line arguments.
            4. Logs all parsed arguments and the command used to run the script.

        External functions (add_ast_arguments, add_conformer_arguments, etc.) are assumed to
        be responsible for adding architecture-specific arguments to the parser.
        """
        super().__init__()

        # Initialize argument parser with common arguments
        self.input_arguments = self.get_arguments()

        # Append additional arguments related to specific architectures
        self.input_arguments = add_ast_arguments(self.input_arguments)
        self.input_arguments = add_conformer_arguments(self.input_arguments)
        self.input_arguments = add_lstm_arguments(self.input_arguments)
        self.input_arguments = add_mlp_arguments(self.input_arguments)
        self.input_arguments = add_residual_arguments(self.input_arguments)
        self.input_arguments = add_wav_to_vec_arguments(self.input_arguments)
        self.input_arguments = add_comparative_metrics_plotter_arguments(self.input_arguments)
        self.input_arguments = add_confusion_matrix_arguments(self.input_arguments)
        self.input_arguments = add_loss_plotter_arguments(self.input_arguments)
        self.input_arguments = add_roc_plotter_arguments(self.input_arguments)
        # Parse the combined set of arguments
        self.input_arguments = self.input_arguments.parse_args()
        self.show_all_settings()
        # Log all parsed settings

    def show_all_settings(self):
        """
        Logs all parsed arguments in a formatted and structured way.

        This method logs:
            - The full command used to run the script (useful for reproducibility).
            - All arguments and their values, properly formatted for readability.

        This function is particularly useful in machine learning experiments where tracking
        hyperparameters and configurations is critical.

        The output is printed to the logging framework, typically to console or log files.
        """
        logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))

        logging.info("Settings:")

        # Compute maximum argument name length for aligned logging
        lengths = [len(x) for x in vars(self.input_arguments).keys()]
        max_length = max(lengths)

        # Log each argument name and value pair
        for keys, values in sorted(vars(self.input_arguments).items()):
            settings_parser = "\t" + keys.ljust(max_length, " ") + " : {}".format(values)
            logging.info(settings_parser)

        # Extra spacing for readability
        logging.info("")

    @staticmethod
    def get_arguments():
        """
        Defines the base arguments required for the training and evaluation pipeline.

        These arguments are common across different models and cover:

            - Dataset paths and structure.
            - Training parameters such as epochs, batch size, and cross-validation splits.
            - Loss function selection.
            - Audio-specific settings such as sample rate and overlap.
            - Plotting configuration for visualizations.
            - Logging verbosity level.

        Returns
        -------
        argparse.ArgumentParser
            Configured ArgumentParser with all core arguments.
        """
        argument_parser = argparse.ArgumentParser(
            description="Model evaluation with metrics and confusion matrices."
        )

        argument_parser.add_argument(
            "--dataset_directory", type=str, default=DEFAULT_DATASET_DIRECTORY,
            help="Directory containing the dataset files (audio, labels, etc.)."
        )

        argument_parser.add_argument(
            "--number_epochs", type=int, default=DEFAULT_NUMBER_EPOCHS,
            help="Number of training epochs for model training."
        )

        argument_parser.add_argument(
            "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
            help="Size of the training batches."
        )

        argument_parser.add_argument(
            "--number_splits", type=int, default=DEFAULT_NUMBER_SPLITS,
            help="Number of splits for cross-validation."
        )


        argument_parser.add_argument(
            "--loss", type=str, default=DEFAULT_LOSS,
            help="Loss function to be used during training (e.g., cross-entropy, MSE)."
        )

        argument_parser.add_argument(
            "--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
            help="Sample rate for audio files (e.g., 16000 Hz)."
        )

        argument_parser.add_argument(
            "--overlap", type=int, default=DEFAULT_OVERLAP,
            help="Overlap size (in samples or frames) when splitting audio segments."
        )

        argument_parser.add_argument(
            "--number_classes", type=int, default=DEFAULT_NUMBER_CLASSES,
            help="Total number of classes in the dataset (for classification tasks)."
        )

        argument_parser.add_argument(
            "--output_directory", type=str, default=DEFAULT_OUTPUT_DIRECTORY,
            help="Directory where outputs (models, plots, logs) will be saved."
        )

        argument_parser.add_argument(
            "--plot_width", type=float, default=DEFAULT_PLOT_WIDTH,
            help="Width of plots (in inches) for visualizations."
        )

        argument_parser.add_argument(
            "--plot_height", type=float, default=DEFAULT_PLOT_HEIGHT,
            help="Height of plots (in inches) for visualizations."
        )

        argument_parser.add_argument(
            "--plot_bar_width", type=float, default=DEFAULT_PLOT_BAR_WIDTH,
            help="Width of individual bars in bar plots."
        )

        argument_parser.add_argument(
            "--plot_cap_size", type=float, default=DEFAULT_PLOT_CAP_SIZE,
            help="Size of error bar caps in plots."
        )

        argument_parser.add_argument(
            "--verbosity", type=int, default=DEFAULT_VERBOSITY,
            help="Logging verbosity level (higher values indicate more detailed logs)."
        )

        argument_parser.add_argument(
            "--file_extension", type=str, default='*.wav',
            help="Logging verbosity level (higher values indicate more detailed logs)."
        )

        return argument_parser

def auto_arguments(function):
    """
    Decorator to initialize an instance of the Arguments class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Arguments.
    """

    def wrapper(self, *args, **kwargs):
        arguments = Arguments()
        self.input_arguments = arguments.input_arguments
        return function(self, *args, **kwargs)

    return wrapper
