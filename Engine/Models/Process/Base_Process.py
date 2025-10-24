#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
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
    import os
    import sys

    import numpy
    import logging

except ImportError as error:
    print(error)
    sys.exit(-1)

class BaseProcess:
    """
    A base class for common utility functions used in the audio classification pipeline.

    This class contains methods for processing metrics, managing directories, normalizing signals,
    and extracting labels from file names. It is used as a utility class to assist in the
    workflow of audio classification tasks.

    Methods
    -------
    __cast_to_dic__(self, metrics_list, probabilities_list, real_labels_list, confusion_matriz_list, history_model)
        Converts the provided lists and the model's training history into a dictionary containing
        aggregated metrics, confusion matrix, and predicted probabilities.

    __create_dir__(sub_directories, list_class_path)
        Creates a list of class directories from the given path. It checks if the directories exist
        and adds them to the list.

    normalization_signal(signal_segments)
        Normalizes a list of signal segments to a range between 0 and 1.

    __get_label__(file_name)
        Extracts the label of a given file based on the directory structure of the dataset.

    Example Usage:
    --------------
    # Instantiate the BaseProcess class (typically as a superclass)
    base_process = BaseProcess()

    # Example usage of normalization
    normalized_signal = base_process.normalization_signal(signal_segments)

    # Example usage of label extraction
    label = base_process.__get_label__('path/to/audio_file/label_audio.wav')
    """

    def __cast_to_dic__(self, metrics_list,
                        probabilities_list, real_labels_list, confusion_matriz_list, history_model):
        """
        Converts the provided lists and model's history into a dictionary containing
        aggregated metrics, confusion matrix,  and predicted probabilities. The function
        calculates the mean and standard deviation of the metrics for accuracy, precision,
        recall, and F1 score across all folds in the cross-validation.

        Parameters
        ----------
            @metrics_list : list
                List of dictionaries containing metrics (accuracy, precision, recall, F1 score) for each fold.
            @probabilities_list : list
                List of model predicted probabilities for each fold.
            @real_labels_list : list
                List of real labels (ground truth) for each fold.
            @confusion_matriz_list : list
                List of confusion matrices for each fold.
            @history_model : object
                The history object of the model, which stores the training process.

        Returns
        -------
        tuple
            A tuple containing:
            - mean_metrics: A dictionary with the mean and standard deviation of the accuracy,
             precision, recall, and F1 score.
            - model history: A dictionary with model's training history.
            - mean_confusion_matrices: A dictionary with the average confusion matrix across all folds.
            - probabilities_predicted: A dictionary containing the predicted probabilities and real labels.

        """
        # Calculate mean metrics across all folds
        mean_metrics = {
            'model_name': self.model_name,
            'Acc.': {'value': numpy.mean([metric['accuracy'] for metric in metrics_list]),
                     'std': numpy.std([metric['accuracy'] for metric in metrics_list])},
            'Prec.': {'value': numpy.mean([metric['precision'] for metric in metrics_list]),
                      'std': numpy.std([metric['precision'] for metric in metrics_list])},
            'Rec.': {'value': numpy.mean([metric['recall'] for metric in metrics_list]),
                     'std': numpy.std([metric['recall'] for metric in metrics_list])},
            'F1.': {'value': numpy.mean([metric['f1_score'] for metric in metrics_list]),
                    'std': numpy.std([metric['f1_score'] for metric in metrics_list])},
        }

        probabilities_predicted = {
            'model_name': self.model_name,
            'predicted': numpy.concatenate(probabilities_list),
            'ground_truth': numpy.concatenate(real_labels_list)
        }

        confusion_matrix_array = numpy.array(confusion_matriz_list)
        mean_confusion_matrix = numpy.mean(confusion_matrix_array, axis=0)
        mean_confusion_matrix = numpy.round(mean_confusion_matrix).astype(numpy.int32).tolist()

        mean_confusion_matrices = {
            "confusion_matrix": mean_confusion_matrix,
            "class_names": ['Class {}'.format(i) for i in range(self.number_classes)],
            "title": self.model_name
        }

        return (mean_metrics, {"Name": self.model_name, "History": history_model.history}, mean_confusion_matrices,
                probabilities_predicted)

    @staticmethod
    def __create_dir__(sub_directories, list_class_path):
        """
        Creates a list of class directories from the given path. Checks if the directories exist
        and adds them to the provided list of class paths.

        Parameters
        ----------
        sub_directories : str
            Path to the directory containing subdirectories of audio files.
        list_class_path : list
            A list to which the subdirectories will be added if they exist.

        Returns
        -------
        list
            The list of subdirectories that exist.
        None
            If the provided path does not exist.

        Example Usage:
        --------------
        class_paths = base_process.__create_dir__('path/to/data', [])
        """
        # Check if the directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Collect all class directories
        logging.info(f"Reading subdirectories in '{sub_directories}'...")
        for class_dir in os.listdir(sub_directories):

            class_path = os.path.join(sub_directories, class_dir)

            if os.path.isdir(class_path):
                list_class_path.append(class_path)

        return list_class_path

    @staticmethod
    def normalization_signal(signal_segments):
        """
        Normalizes a list of signal segments to a range between 0 and 1.

        Parameters
        ----------
        signal_segments : list
            A list or array of audio signal segments to be normalized.

        Returns
        -------
        numpy.ndarray
            A normalized numpy array with values between 0 and 1.

        Example Usage:
        --------------
        normalized_signal = base_process.normalization_signal(signal_segments)
        """
        signal_segments = numpy.abs(numpy.array(signal_segments))

        # Normalize each segment
        signal_min = numpy.min(signal_segments)
        signal_max = numpy.max(signal_segments)

        if signal_max != signal_min:
            normalized_signal = (signal_segments - signal_min) / (signal_max - signal_min)
        else:
            normalized_signal = numpy.zeros_like(signal_segments)

        return normalized_signal


    @staticmethod
    def __get_label__(file_name):
        """
        Extracts the label from the file path based on the directory structure.

        The label is assumed to be the first part of the parent directory's name (separated by an underscore).

        Parameters
        ----------
            @file_name : str
                The file path from which the label will be extracted.

        Returns
        -------
            @str
                The label extracted from the file's parent directory.

        Example:
        --------------
            label = base_process.__get_label__('path/to/audio_file/label_audio.wav')
        """

        return file_name.split('/')[-2].split('_')[0]