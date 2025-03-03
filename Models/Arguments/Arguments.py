
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

LOGGING_FILE_NAME = "logging.log"


class LoggerSetup:
    def __init__(self, arguments):
        self.arguments = arguments
        self.setup_logger()

    def get_logs_path(self):
        return "./logs"

    def setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(self.arguments.verbosity)

        logging_format = '%(asctime)s\t***\t%(message)s'
        if self.arguments.verbosity == logging.DEBUG:
            logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

        formatter = logging.Formatter(logging_format)

        logging_filename = os.path.join(self.get_logs_path(), LOGGING_FILE_NAME)
        rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
        rotatingFileHandler.setLevel(self.arguments.verbosity)
        rotatingFileHandler.setFormatter(formatter)

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(self.arguments.verbosity)
        streamHandler.setFormatter(formatter)

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(rotatingFileHandler)
        logger.addHandler(streamHandler)


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
        self.arguments = add_argument_framework()
        # Add various model-specific argument parsers
        self.arguments = add_argument_adversarial(self.arguments)
        self.arguments = add_argument_data_load(self.arguments)
        self.arguments = add_argument_autoencoder(self.arguments)
        self.arguments = add_argument_diffusion(self.arguments)
        self.arguments = add_argument_variation_autoencoder(self.arguments)
        self.arguments = add_argument_wasserstein_gan(self.arguments)
        self.arguments = add_argument_decision_tree(self.arguments)
        self.arguments = add_argument_gaussian_process(self.arguments)
        self.arguments = add_argument_gradient_boosting(self.arguments)
        self.arguments = add_argument_k_means(self.arguments)
        self.arguments = add_argument_knn(self.arguments)
        self.arguments = add_argument_naive_bayes(self.arguments)
        self.arguments = add_argument_linear_regression(self.arguments)
        self.arguments = add_argument_spectral_clustering(self.arguments)
        self.arguments = add_argument_perceptron(self.arguments)
        self.arguments = add_argument_quadratic_discriminant_analysis(self.arguments)
        self.arguments = add_argument_random_forest(self.arguments)
        self.arguments = add_argument_stochastic_gradient_descent(self.arguments)
        self.arguments = add_argument_support_vector_machine(self.arguments)

        self.arguments = self.arguments.parse_args()

        LoggerSetup(self.arguments)
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