#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/26'
__credits__ = ['unknown']


try:

    import os
    import gc
    import sys

    import numpy
    import logging
    import argparse
    import tensorflow
    import subprocess

    import seaborn as sns
    from datetime import datetime
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix

    from logging.handlers import RotatingFileHandler
    from Engine.Models.AST import AudioAST, get_audio_ast_args
    from Engine.Models.LSTM import AudioLSTM, get_lstm_model_args
    from Engine.Models.MLP import DenseModel, get_MLP_model_args

    from Engine.Models.Conformer import Conformer, get_conformer_models_args
    from Engine.Models.Wav2Vec2 import AudioWav2Vec2, get_wav_to_vec_args
    from Engine.Models.ResidualModel import ResidualModel, get_residual_model_args

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt")
    sys.exit(-1)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tensorflow.get_logger().setLevel('ERROR')

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

DEFAULT_MATRIX_FIGURE_SIZE = (5, 5)
DEFAULT_MATRIX_COLOR_MAP = 'Blues'
DEFAULT_MATRIX_ANNOTATION_FONT_SIZE = 10
DEFAULT_MATRIX_LABEL_FONT_SIZE = 12
DEFAULT_MATRIX_TITLE_FONT_SIZE = 14
DEFAULT_SHOW_PLOT = False
DEFAULT_DATASET_DIRECTORY = "Dataset/"
DEFAULT_NUMBER_EPOCHS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUMBER_SPLITS = 2
DEFAULT_LOSS = 'sparse_categorical_crossentropy'
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_OVERLAP = 2
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_OUTPUT_DIRECTORY = "Results/"
DEFAULT_PLOT_WIDTH = 14
DEFAULT_PLOT_HEIGHT = 8
DEFAULT_PLOT_BAR_WIDTH = 0.15
DEFAULT_PLOT_CAP_SIZE = 10


class EvaluationModels:

    def __init__(self):
        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []
        self.list_roc_curve = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    @staticmethod
    def train_and_collect_metrics(model_class, dataset_directory, number_epochs, batch_size, number_splits, loss,
                                  sample_rate, overlap, number_classes, arguments):
        """
        Trains the model and collects relevant metrics, history, confusion matrices, and ROC curve data.

        Args:
            model_class (class): The class of the model to be instantiated and trained.
            dataset_directory (str): Directory where the dataset is located.
            number_epochs (int): Number of training epochs.
            batch_size (int): Size of the training batches.
            number_splits (int): Number of data splits for cross-validation.
            loss (str): The loss function used during training.
            sample_rate (int): Sample rate of the data.
            overlap (float): Overlap rate between splits.
            number_classes (int): Number of target classes.
            arguments (dict): Additional arguments for the training process.

        Returns:
            tuple: Contains metrics, history, confusion matrices, and ROC curve data.
        """
        logging.info(f"Starting training for model {model_class.__name__}.")

        try:
            # Instantiate the model
            instance = model_class()
            logging.info(f"Instantiated model class '{model_class.__name__}'.")

            # Train the model and collect results
            metrics, history, matrices, roc_list = instance.train(dataset_directory,
                                                                  number_epochs,
                                                                  batch_size,
                                                                  number_splits,
                                                                  loss,
                                                                  sample_rate,
                                                                  overlap,
                                                                  number_classes,
                                                                  arguments)
            logging.info(
                f"Training completed for model '{model_class.__name__}'. Collected metrics, history, matrices, and ROC data.")

        except Exception as e:
            logging.error(f"Error during training of model '{model_class.__name__}': {e}")
            raise

        finally:
            # Force garbage collection to free up memory
            gc.collect()
            logging.info(f"Garbage collection complete after training model '{model_class.__name__}'.")

        return metrics, history, matrices, roc_list


    def run(self, models, dataset_directory, number_epochs, batch_size, number_splits, loss, sample_rate, overlap,
            number_classes, output_directory, plot_width, plot_height, plot_bar_width, plot_cap_size, arguments):

        logging.info("Starting the training and evaluation process.")

        for i, model_class in enumerate(models):
            logging.debug(f"Training model {i + 1}/{len(models)}: {model_class.__name__}")

            try:
                metrics, history, matrices, roc_list = self.train_and_collect_metrics(model_class=model_class,
                                                                                      dataset_directory=dataset_directory,
                                                                                      number_epochs=number_epochs,
                                                                                      batch_size=batch_size,
                                                                                      number_splits=number_splits,
                                                                                      loss=loss,
                                                                                      sample_rate=sample_rate,
                                                                                      overlap=overlap,
                                                                                      number_classes=number_classes,
                                                                                      arguments=arguments)

                self.mean_metrics.append(metrics)
                self.mean_history.append(history)
                self.mean_matrices.append(matrices)

                logging.info(f"Model {model_class.__name__} training completed. Metrics collected.")

                self.plot_roc_curve(roc_list, "Results/")
                logging.info(f"ROC curve plotted for {model_class.__name__}.")

            except Exception as e:
                logging.error(f"Error during training of model {model_class.__name__}: {str(e)}")
                raise

        try:
            logging.info("Plotting comparative metrics.")
            self.plot_comparative_metrics(dictionary_metrics_list=self.mean_metrics,
                                          file_name=output_directory,
                                          figure_width=plot_width,
                                          figure_height=plot_height,
                                          bar_width=plot_bar_width,
                                          caption_size=plot_cap_size)

            logging.info("Plotting confusion matrices.")
            self.plot_confusion_matrices(confusion_matrix_list=self.mean_matrices,
                                         file_name_path=output_directory,
                                         fig_size=DEFAULT_MATRIX_FIGURE_SIZE,
                                         cmap=DEFAULT_MATRIX_COLOR_MAP,
                                         annot_font_size=DEFAULT_MATRIX_ANNOTATION_FONT_SIZE,
                                         label_font_size=DEFAULT_MATRIX_LABEL_FONT_SIZE,
                                         title_font_size=DEFAULT_MATRIX_TITLE_FONT_SIZE,
                                         show_plot=DEFAULT_SHOW_PLOT)

            logging.info("Plotting and saving loss.")
            self.plot_and_save_loss(history_dict_list=self.mean_history, path_output=output_directory)

        except Exception as e:
            logging.error(f"Error during plotting or saving results: {str(e)}")
            raise

        try:
            logging.info("Running final script to generate PDF.")
            self.run_python_script('--output', "Results.pdf")
            logging.info("PDF generation completed.")

        except Exception as e:
            logging.error(f"Error running the script for PDF generation: {str(e)}")
            raise

    @staticmethod
    def run_python_script(*args) -> int:
        """
        Executes a Python script with the specified arguments and logs the output.

        Args:
            *args: Additional arguments to pass to the script.

        Returns:
            int: The return code of the executed script.
        """
        logging.info("Starting the execution of 'GeneratePDF.py' with arguments: %s", args)

        try:
            # Prepare the command to run the script
            command = ['python3', 'GeneratePDF.py'] + list(args)
            logging.info("Command to be executed: %s", command)

            # Execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Log standard output
            logging.info("Standard Output:\n%s", result.stdout)

            # Log standard error if it exists
            if result.stderr:
                logging.warning("Standard Error:\n%s", result.stderr)

            return result.returncode

        except subprocess.CalledProcessError as e:

            logging.error("An error occurred while executing the script: %s", e)
            logging.error("Standard Output:\n%s", e.stdout)
            logging.error("Standard Error:\n%s", e.stderr)
            return e.returncode

        finally:
            logging.info("Execution of 'GeneratePDF.py' completed.")


def get_arguments():

    parser = argparse.ArgumentParser(description="Model evaluation with metrics and confusion matrices.")

    parser.add_argument("--dataset_directory", type=str,
                        default=DEFAULT_DATASET_DIRECTORY, help="Directory containing the dataset.")

    parser.add_argument("--number_epochs", type=int,
                        default=DEFAULT_NUMBER_EPOCHS, help="Number of training epochs.")

    parser.add_argument("--batch_size", type=int,
                        default=DEFAULT_BATCH_SIZE, help="Size of the batches for training.")

    parser.add_argument("--number_splits", type=int,
                        default=DEFAULT_NUMBER_SPLITS, help="Number of splits for cross-validation.")

    parser.add_argument("--loss", type=str,
                        default=DEFAULT_LOSS, help="Loss function to use during training.")

    parser.add_argument("--sample_rate", type=int,
                        default=DEFAULT_SAMPLE_RATE, help="Sample rate of the audio files.")

    parser.add_argument("--overlap", type=int,
                        default=DEFAULT_OVERLAP, help="Overlap for the audio segments.")

    parser.add_argument("--number_classes", type=int,
                        default=DEFAULT_NUMBER_CLASSES, help="Number of classes in the dataset.")

    parser.add_argument("--output_directory", type=str,
                        default=DEFAULT_OUTPUT_DIRECTORY, help="Directory to save output files.")

    parser.add_argument("--plot_width", type=float,
                        default=DEFAULT_PLOT_WIDTH, help="Width of the plots.")

    parser.add_argument("--plot_height", type=float,
                        default=DEFAULT_PLOT_HEIGHT, help="Height of the plots.")

    parser.add_argument("--plot_bar_width", type=float,
                        default=DEFAULT_PLOT_BAR_WIDTH, help="Width of the bars in the bar plots.")

    parser.add_argument("--plot_cap_size", type=float,
                        default=DEFAULT_PLOT_CAP_SIZE, help="Capsize of the error bars in the bar plots.")

    parser.add_argument("--verbosity", type=int,
                        help='Verbosity (Default {})'.format(DEFAULT_VERBOSITY), default=DEFAULT_VERBOSITY)

    parser =  get_audio_ast_args(parser)
    parser = get_conformer_models_args(parser)
    parser = get_lstm_model_args(parser)
    parser =  get_MLP_model_args(parser)
    parser = get_residual_model_args(parser)
    parser = get_wav_to_vec_args(parser)

    arguments = parser.parse_args()

    return arguments

def show_all_settings(arguments):
    """
    Logs all settings and command-line arguments after parsing.
    Displays the command used to run the script along with the
    corresponding values for each argument.
    """

    # Log the command used to execute the script
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    # Calculate the maximum length of argument names for formatting
    lengths = [len(x) for x in vars(arguments).keys()]
    max_length = max(lengths)

    # Log each argument and its value
    for keys, values in sorted(vars(arguments).items()):

        # Start with a tab for indentation
        settings_parser = "\t"

        # Left-justify the argument name for better readability
        settings_parser += keys.ljust(max_length, " ")

        # Append the value of the argument
        settings_parser += " : {}".format(values)

        # Log the formatted argument and value
        logging.info(settings_parser)

    logging.info("")  # Log a newline for spacing


# Main entry point of the program
if __name__ == "__main__":

    # Get input arguments from the user
    input_arguments = get_arguments()
    # Obtain a logger to log information
    logger = logging.getLogger()

    # Function that defines the path for the logs directory
    def get_logs_path():
        logs_dir = 'Logs'  # Name of the directory where logs will be stored
        os.makedirs(logs_dir, exist_ok=True)  # Create the directory if it doesn't exist
        return logs_dir  # Return the path to the logs directory

    # Default format for log messages
    logging_format = '%(asctime)s\t***\t%(message)s'

    # If verbosity is set to DEBUG, update the log format to include more information
    if input_arguments.verbosity == logging.DEBUG:
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    # Generate a log file name with the current date and time
    LOGGING_FILE_NAME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    logging_filename = os.path.join(get_logs_path(), LOGGING_FILE_NAME)  # Create the full path for the log file

    # Set the log level based on user input
    logger.setLevel(input_arguments.verbosity)

    # Create a rotating file handler for logs
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=1000000, backupCount=5)
    rotatingFileHandler.setLevel(input_arguments.verbosity)  # Set log level for the file handler
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))  # Set the format for log messages

    # Add the file handler to the logger
    logger.addHandler(rotatingFileHandler)

    # Create a console handler to display logs in the console
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(input_arguments.verbosity)  # Set log level for the console handler
    consoleHandler.setFormatter(logging.Formatter(logging_format))  # Set the format for log messages

    # Add the console handler to the logger
    logger.addHandler(consoleHandler)

    # If the logger already has handlers, clear them
    if logger.hasHandlers():
        logger.handlers.clear()

    # Re-add the handlers to the logger
    logger.addHandler(rotatingFileHandler)
    logger.addHandler(consoleHandler)

    # Display all current settings based on input arguments
    show_all_settings(input_arguments)

    # List of available machine learning models for evaluation
    available_models = [
        AudioAST,
        AudioLSTM,
        DenseModel,
        Conformer,
        AudioWav2Vec2,
        ResidualModel
    ]

    # Create an instance of the evaluation class
    evaluation = EvaluationModels()
    # Run the evaluation of the models with specified parameters
    evaluation.run(
        models=available_models,  # Models to be evaluated
        dataset_directory=input_arguments.dataset_directory,  # Directory of the dataset
        number_epochs=input_arguments.number_epochs,  # Number of epochs for training
        batch_size=input_arguments.batch_size,  # Batch size
        number_splits=input_arguments.number_splits,  # Number of splits for validation
        loss=input_arguments.loss,  # Loss function to be used
        sample_rate=input_arguments.sample_rate,  # Sample rate
        overlap=input_arguments.overlap,  # Overlap between samples
        number_classes=input_arguments.number_classes,  # Number of classes in the dataset
        output_directory=input_arguments.output_directory,  # Output directory for results
        plot_width=input_arguments.plot_width,  # Width of the plot
        plot_height=input_arguments.plot_height,  # Height of the plot
        plot_bar_width=input_arguments.plot_bar_width,  # Width of bars in the plot
        plot_cap_size=input_arguments.plot_cap_size,  # Size of caps in the plots
        arguments=input_arguments  # Additional arguments for the evaluation function
    )