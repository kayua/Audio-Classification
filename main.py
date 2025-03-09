#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/26'
__credits__ = ['unknown']


# try:

import gc
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import tensorflow

from Tools.RunScript import RunScript

# except ImportError as error:
#     print(error)
#     print("1. Install requirements:")
#     print("  pip3 install --upgrade pip")
#     print("  pip3 install -r requirements.txt")
#     sys.exit(-1)


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


class EvaluationModels(RunScript):

    def __init__(self):
        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []
        self.list_roc_curve = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    @staticmethod
    def train_and_collect_metrics(model_class, dataset_directory, number_epochs, batch_size, number_splits, loss,
                                  sample_rate, overlap, number_classes, arguments):

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