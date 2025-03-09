#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/26'
__credits__ = ['unknown']

import gc
import logging
import os

import tensorflow

from Engine.Arguments.Arguments import auto_arguments
from Engine.Models.AST import AudioSpectrogramTransformer
from Engine.Models.Conformer import Conformer
from Engine.Models.LSTM import AudioLSTM
from Tools.Logger import auto_logger
from Tools.PlotterTools import PlotterTools
from Tools.RunScript import RunScript

# try:

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


class Main(RunScript, PlotterTools):

    def __init__(self):

        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []
        self.list_roc_curve = []

        self.__start__()
        PlotterTools.__init__(self, self.input_arguments)

    @auto_logger
    def __autologger__(self):
        """

            A Logger class designed to manage and configure logging for an application.
            It supports logging to both console and rotating log files. The log file name is
            dynamically generated based on the current date and time, and it creates backups of
            the log file to prevent the log file from growing too large.

            Attributes:
            ----------
                @_logger (logging.Logger): The main logger object for logging messages.
                @_logging_format (str): The format in which log messages are written.
                @_rotatingFileHandler (logging.Handler): The handler that writes logs to a rotating file.
                @_consoleHandler (logging.Handler): The handler that writes logs to the console.

        """
        pass

    @auto_arguments
    def __start__(self):
        self.__autologger__()
        """
            Arguments Class

            This class manages the configuration and command-line arguments for a machine learning training
            and evaluation pipeline. It collects arguments from multiple subsystems (AST, Conformer, LSTM,
            MLP, Residual, Wav2Vec) and consolidates them into a single namespace.
            
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

        """
        pass

    def train_and_collect_metrics(self, model_class):

        logging.info(f"Starting training for model {model_class.__name__}.")

        try:
            # Instantiate the model
            instance = model_class(arguments = self.input_arguments)
            logging.info(f"Instantiated model class '{model_class.__name__}'.")
            # Train the model and collect results
            instance.build_model()
            model_metrics, model_history, model_matrices, model_roc_list = instance.train()
            #logging.info( f"Training completed for model '{model_class.__name__}'."
            #              f" Collected metrics, history, matrices, and ROC data.")11

        except Exception as e:
            logging.error(f"Error during training of model '{model_class.__name__}': {e}")
            raise

        finally:
            # Force garbage collection to free up memory
            gc.collect()
            logging.info(f"Garbage collection complete after training model '{model_class.__name__}'.")

        return [], [], [], []

        # return model_metrics, model_history, model_matrices, model_roc_list


    def __exec__(self, models, output_directory):

        logging.info("Starting the training and evaluation process.")

        for i, model_class in enumerate(models):
            logging.debug(f"Training model {i + 1}/{len(models)}: {model_class.__name__}")

            try:
                metrics, history, matrices, roc_list = self.train_and_collect_metrics(model_class=model_class)

                # self.mean_metrics.append(metrics)
                # self.mean_history.append(history)
                # self.mean_matrices.append(matrices)
                #
                # logging.info(f"Model {model_class.__name__} training completed. Metrics collected.")
                #
                # self.plot_roc_curve(roc_list, "Results/")
                # logging.info(f"ROC curve plotted for {model_class.__name__}.")

            except Exception as e:
                logging.error(f"Error during training of model {model_class.__name__}: {str(e)}")
                raise

        try:
            logging.info("Plotting comparative metrics.")
            self.plot_comparative_metrics(dictionary_metrics_list=self.mean_metrics,
                                          file_name=output_directory)

            logging.info("Plotting confusion matrices.")
            self.plot_confusion_matrices(confusion_matrix_list=self.mean_matrices,
                                         file_name_path=output_directory)

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
    main = Main()
    main.__start__()

    available_models = [
       AudioLSTM,
#        Conformer,
#        DenseModel,
#        AudioSpectrogramTransformer,
#        AudioWav2Vec2,
#        ResidualModel
    ]

    main.__exec__(available_models, "Results")