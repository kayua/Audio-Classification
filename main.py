#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

from Engine.Models import MLP
from Engine.Models.MobileNet import MobileNetModel
from Engine.Models.Process.MLP_Process import MLPProcess

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


"""
Audio Classification using Multiple Neural Network Architectures

This script implements and evaluates multiple neural network architectures for audio classification,
including Wav2Vec2, MLP, Audio Spectrogram Transformer, Conformer, Residual, and LSTM.

It performs the following tasks:
    - Training and evaluation of the models
    - Generation of performance metrics: Accuracy, Precision, Recall, and F1-Score
    - Generation of Confusion Matrix
    - Plotting of the ROC Curve
    - Comparison of models' performance
    - Automatic generation of a final results report

Example Usage:
--------------
Run the script directly from the command line:
    $ python3 main.py

Requirements:
-------------
Ensure that all dependencies are installed using the provided requirements file:
    $ pip3 install --upgrade pip
    $ pip3 install -r requirements.txt

Author: Unknown
Email: unknown@unknown.com.br
Version: 1.0.0
Initial Data: 2024/07/17
Last Update: 2024/07/26
Credits: Unknown
"""

try:

    import gc
    import os
    import os

    import sys

    import logging
    import tensorflow

    from Tools.Logger import auto_logger
    from Tools.RunScript import RunScript

    from Engine.Models.LSTM import AudioLSTM
    from Engine.Models.MLP import DenseModel

    from Tools.PlotterTools import PlotterTools
    from Engine.Models.Conformer import Conformer
    from Engine.Models.Wav2Vec2 import AudioWav2Vec2

    from Engine.Arguments.Arguments import auto_arguments
    from Engine.Models.ResidualModel import ResidualModel

    from Engine.Models.AST import AudioSpectrogramTransformer

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


class Main(RunScript, PlotterTools):
    """
    Main Class for Training and Evaluating Audio Classification Models

    This class handles model instantiation, training, metric collection, visualization, and final report generation.
    """

    def __init__(self):
        """
        Initializes the Main class by setting up lists for metric storage and initializing necessary tools.
        """
        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []
        self.list_roc_curve = []

        self.__start__()
        PlotterTools.__init__(self, self.input_arguments)

    def train_and_collect_metrics(self, model_class):
        """
        Trains a given model and collects relevant performance metrics.

        Args:
            model_class (class): The neural network model class to be instantiated and trained.

        Returns:
            tuple: Model metrics, training history, confusion matrices, and ROC curve data.
        """
        logging.info(f"Starting training for model {model_class.__name__}.")

        try:
            # Instantiate the model
            instance = model_class(arguments = self.input_arguments)
            logging.info(f"Instantiated model class '{model_class.__name__}'.")
            # Train the model and collect results
            instance.build_model()
            model_metrics, model_history, model_matrices, model_roc_list = instance.train()
            logging.info( f"Training completed for model '{model_class.__name__}'."
                          f" Collected metrics, history, matrices, and ROC data.")

        except Exception as e:
            logging.error(f"Error during training of model '{model_class.__name__}': {e}")
            raise

        finally:
            # Force garbage collection to free up memory
            gc.collect()
            logging.info(f"Garbage collection complete after training model '{model_class.__name__}'.")


        return model_metrics, model_history, model_matrices, model_roc_list


    def __exec__(self, models, output_directory):
        """
        Executes training and evaluation for a list of models.

        Args:
            models (list): List of model classes to be trained and evaluated.
            output_directory (str): Directory where results will be saved.
        """
        logging.info("Starting the training and evaluation process.")

        for index, model_class in enumerate(models):
            logging.debug(f"Training model {index + 1}/{len(models)}: {model_class.__name__}")

            try:
                metrics, history, matrices, roc_list = self.train_and_collect_metrics(model_class=model_class)

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
                                          file_name=output_directory)

            logging.info("Plotting confusion matrices.")
            self.plot_confusion_matrices(confusion_matrix_list=self.mean_matrices,
                                         file_name_path=output_directory)

            logging.info("Plotting and saving loss.")
            self.plot_loss(loss_curve_history_dict_list=self.mean_history, loss_curve_path_output=output_directory)

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

    @auto_logger
    def __autologger__(self):
        """

            A Logger class designed to manage and configure logging for an application.

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

            Methods
            -------
                @__init__() :
                    Initializes the argument parser, adds arguments from multiple components
                @show_all_settings() :
                    Logs the parsed arguments in a formatted manner for better visibility
                @get_arguments() :
                    Static method that defines the base arguments required 

        """
        pass


# Main entry point of the program
if __name__ == "__main__":
    main = Main()
    main.__start__()

    available_models = [Conformer]

    main.__exec__(available_models, "Results")
