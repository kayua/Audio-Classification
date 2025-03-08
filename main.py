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
    def calculate_accuracy(label_true, label_predicted):
        try:
            return accuracy_score(label_true, label_predicted)
        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true, label_predicted):
        try:
            return precision_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true, label_predicted):
        try:
            return recall_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true, label_predicted):
        try:
            return f1_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_auc(label_true, label_predicted_probability):
        try:
            return roc_auc_score(label_true, label_predicted_probability, multi_class='ovr')
        except Exception as e:
            raise ValueError(f"Error calculating AUC: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true, label_predicted):
        try:
            return confusion_matrix(label_true, label_predicted).tolist()
        except Exception as e:
            raise ValueError(f"Error calculating confusion matrix: {e}")


    def calculate_metrics(self, label_true, label_predicted):
        metrics = {}
        logging.info("Starting to calculate metrics.")

        try:
            logging.info("Calculating accuracy.")
            metrics['accuracy'] = self.calculate_accuracy(label_true, label_predicted)

            logging.info("Calculating precision.")
            metrics['precision'] = self.calculate_precision(label_true, label_predicted)

            logging.info("Calculating recall.")
            metrics['recall'] = self.calculate_recall(label_true, label_predicted)

            logging.info("Calculating F1 score.")
            metrics['f1_score'] = self.calculate_f1_score(label_true, label_predicted)

            logging.info("Calculating confusion matrix.")
            confusion_matrix_result = self.calculate_confusion_matrix(label_true, label_predicted)

        except ValueError as e:
            logging.error(f"An error occurred while calculating metrics: {e}")
            return {}, None

        logging.info("Metric calculation completed successfully.")
        return metrics, confusion_matrix_result


    @staticmethod
    def plot_comparative_metrics(dictionary_metrics_list,
                                 file_name,
                                 figure_width=12,
                                 figure_height=8,
                                 bar_width=0.20,
                                 caption_size=10):

        logging.info("Starting to plot comparative metrics.")

        try:
            # Define metrics and corresponding color bases
            list_metrics = ['Acc.', 'Prec.', 'Rec.', 'F1.']
            list_color_bases = {'Acc.': 'Blues', 'Prec.': 'Greens', 'Rec.': 'Reds', 'F1.': 'Purples'}

            number_metrics = len(list_metrics)
            number_models = len(dictionary_metrics_list)

            logging.info(f"Number of models: {number_models}, Number of metrics: {number_metrics}")

            # Create the figure for the plot
            figure_plot, axis_plot = plt.subplots(figsize=(figure_width, figure_height))
            positions = numpy.arange(number_metrics)

            logging.info(f"Figure created with dimensions ({figure_width}, {figure_height}).")

            # Loop through metrics to create the bars
            for metric_id, key_metric in enumerate(list_metrics):
                logging.info(f"Processing metric: {key_metric}")

                for metric_dictionary_id, model_name in enumerate(dictionary_metrics_list):
                    logging.info(f"Processing model: {model_name['model_name']} for metric {key_metric}")

                    # Retrieve metric values and standard deviations
                    metric_values = model_name[key_metric]['value']
                    metric_stander_deviation = model_name[key_metric]['std']
                    metric_color_bar = plt.get_cmap(list_color_bases[key_metric])(
                        metric_dictionary_id / (number_models - 1))
                    metric_label = f"{key_metric} {model_name['model_name']}"

                    logging.info(
                        f"Values: {metric_values},"
                        f" Std deviation: {metric_stander_deviation},"
                        f" Color: {list_color_bases[key_metric]}")

                    # Plotting the bars
                    metric_bar_definitions = axis_plot.bar(positions[metric_id] + metric_dictionary_id * bar_width,
                                                           metric_values,
                                                           yerr=metric_stander_deviation,
                                                           color=metric_color_bar,
                                                           width=bar_width,
                                                           edgecolor='grey',
                                                           capsize=caption_size,
                                                           label=metric_label)

                    logging.debug(f"Bars plotted for {metric_label}")

                    # Annotating the bars with values
                    for shape_bar in metric_bar_definitions:
                        bar_height = shape_bar.get_height()
                        axis_plot.annotate(f'{bar_height:.2f}',
                                           xy=(shape_bar.get_x() + shape_bar.get_width() / 2, bar_height),
                                           xytext=(0, 10),
                                           textcoords="offset points",
                                           ha='center',
                                           va='bottom')
                        logging.debug(f"Annotated bar with height: {bar_height}")

            # Adding labels, titles, and legends
            axis_plot.set_xlabel('Metric', fontweight='bold')
            axis_plot.set_xticks([r + bar_width * (number_models - 1) / 2 for r in positions])
            axis_plot.set_xticklabels(list_metrics)
            axis_plot.set_ylabel('Score', fontweight='bold')
            axis_plot.set_title('Comparative Metrics', fontweight='bold')
            axis_plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=number_models)

            logging.info("Finalizing layout and saving plot.")

            # Save the plot as a file
            plt.tight_layout()
            output_path = f'{file_name}metrics.png'
            plt.savefig(output_path)
            logging.debug(f"Comparative metrics plot saved to {output_path}")

            # Closing the plot to free up memory
            plt.close()

        except Exception as e:
            logging.error(f"An error occurred while plotting comparative metrics: {e}")
            raise


    @staticmethod
    def plot_roc_curve(probabilities_predicted, file_name_path):
        """
        Plots the ROC curve and calculates AUC for each class based on the predicted probabilities.

        Parameters
        ----------
        probabilities_predicted : dict
            A dictionary containing:
            - 'model_name': Name of the model.
            - 'predicted': An array of predicted probabilities for each class.
            - 'ground_truth': The ground truth labels.
        file_name_path : str
            Path to save the ROC curve plot.
        """
        logging.info("Starting to plot ROC curve.")

        try:
            # Extract data from the dictionary
            model_name = probabilities_predicted['model_name']
            y_score = probabilities_predicted['predicted']
            y_true = probabilities_predicted['ground_truth']

            logging.info(f"Model: {model_name}, Number of classes: {y_score.shape[1]}")

            # Binarize the ground truth labels for ROC calculation
            y_true_bin = label_binarize(y_true, classes=numpy.arange(y_score.shape[1]))
            logging.debug(f"Ground truth labels binarized: {y_true_bin.shape}")

            # Initialize variables for ROC calculation
            false_positive_r = {}
            true_positive_r = {}
            roc_auc = {}

            # Calculate ROC curve and AUC for each class
            for i in range(y_score.shape[1]):
                false_positive_r[i], true_positive_r[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(false_positive_r[i], true_positive_r[i])
                logging.info(f"Class {i}: AUC = {roc_auc[i]:.2f}")

            # Plot the ROC curves
            plt.figure()
            for i in range(y_score.shape[1]):
                plt.plot(false_positive_r[i], true_positive_r[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
                logging.debug(f"ROC curve plotted for class {i}")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend(loc='lower right')

            # Save the plot to file
            file_path = f"{file_name_path}ROC_{model_name}.png"
            plt.savefig(file_path)
            plt.close()
            logging.info(f"ROC curve saved to {file_path}")

        except Exception as e:
            logging.error(f"Error occurred while plotting ROC curve: {e}")
            raise


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