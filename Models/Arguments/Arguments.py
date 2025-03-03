#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Models.Arguments.ArgumentsAST import add_ast_arguments
from Models.Arguments.ArgumentsConformer import add_conformer_arguments
from Models.Arguments.ArgumentsLSTM import add_lstm_arguments
from Models.Arguments.ArgumentsMLP import add_MLP_arguments, add_mlp_arguments
from Models.Arguments.ArgumentsResidual import add_residual_arguments
from Models.Arguments.ArgumentsWav2Vec2 import add_wav_to_vec_arguments

try:
    import sys
    import logging

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}


def arguments(function):
    """
    Decorator to initialize an instance of the Arguments class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Arguments.
    """

    def wrapper(self, *args, **kwargs):
        # Initialize the Arguments class for the instance
        Arguments.__init__(self)
        # Call the wrapped function with the provided arguments
        return function(self, *args, **kwargs)

    return wrapper



class Arguments:
    """
    Class to manage and parse command-line arguments for various machine
    learning models and settings. It initializes the arguments using
    multiple specific argument addition functions.

    Attributes:
        arguments (Namespace): Parsed command-line arguments.
    """

    def __init__(self):
        """
        Initializes the Arguments class by adding various argument
        options for different machine learning models and settings.
        It also configures logging based on verbosity settings and
        prepares the output directory for storing logs.
        """
        # Initialize arguments from the framework
        super().__init__()
        self.arguments = add_ast_arguments()
        self.arguments = add_conformer_arguments(self.arguments)
        self.arguments = add_lstm_arguments(self.arguments)
        self.arguments = add_mlp_arguments(self.arguments)
        self.arguments = add_residual_arguments(self.arguments)
        self.arguments = add_wav_to_vec_arguments(self.arguments)

        self.arguments = self.arguments.parse_args()

        self.show_all_settings()

    def show_all_settings(self):
        """
        Logs all settings and command-line arguments after parsing.
        Displays the command used to run the script along with the
        corresponding values for each argument.
        """
        # Log the command used to execute the script
        logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
        logging.info("Settings:")
        # Calculate the maximum length of argument names for formatting
        lengths = [len(x) for x in vars(self.arguments).keys()]
        max_length = max(lengths)

        # Log each argument and its value
        for keys, values in sorted(vars(self.arguments).items()):
            settings_parser = "\t"  # Start with a tab for indentation
            # Left-justify the argument name for better readability
            settings_parser += keys.ljust(max_length, " ")
            # Append the value of the argument
            settings_parser += " : {}".format(values)
            # Log the formatted argument and value
            logging.info(settings_parser)

        logging.info("")  # Log a newline for spacing