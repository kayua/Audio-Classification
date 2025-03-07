#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

try:
    import sys
    import numpy
    import logging
    from Engine.Exception.MetricsException import AccuracyError

except ImportError as error:
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 -m venv ~/Python3venv/DeepOceanAI ")
    print("  source ~/Python3venv/DeepOceanAI/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class Accuracy:
    """
    A class for calculating accuracy and validating input labels.

    Methods:
        get_metric(predicted_labels, true_labels):
            Calculate accuracy given predicted and true labels.

        _check_input_labels(predicted_labels, true_labels):
            Check the validity and type of input labels.

    Example:
        # Create an instance of the Accuracy class
        accuracy_calculator = Accuracy()

        # Define predicted and true labels as numpy arrays
        predicted_labels = np.array([1, 0, 1, 0, 1])
        true_labels = np.array([1, 0, 0, 1, 1])

        # Calculate and print the accuracy
        print(f"Accuracy: {accuracy_calculator.get_metric(predicted_labels, true_labels)}")
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate accuracy given predicted and true labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels.
            true_labels (numpy.ndarray): Array of true labels.

        Returns:
            float: The accuracy as a floating-point number between 0 and 1.

        Raises:
            AccuracyError: Custom exception class for handling accuracy calculation errors.
        """
        # Log start of accuracy calculation
        logging.info("Calculating accuracy...")

        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:
            correct_prediction = numpy.sum(predicted_labels == true_labels)
            accuracy = correct_prediction / len(predicted_labels)

            # Log accuracy result
            logging.info(f"Accuracy calculated successfully: {accuracy}")

            return accuracy

        except AccuracyError as e:
            logging.error(f"An error occurred: {e}")
            raise e

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels.
            true_labels (numpy.ndarray): Array of true labels.

        Raises:
            AccuracyError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels or true_labels is None
        if predicted_labels is None or true_labels is None:
            logging.error("One of the label arrays is None.")
            raise AccuracyError("Error: predicted_labels and true_labels cannot be None")

        # Check if predicted_labels and true_labels are numpy arrays
        if not isinstance(predicted_labels, numpy.ndarray) or not isinstance(true_labels, numpy.ndarray):
            logging.error("The input labels must be numpy arrays.")
            raise AccuracyError("Error: predicted_labels and true_labels must be numpy arrays")

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            logging.error("The input arrays do not have matching dimensions.")
            raise AccuracyError("Error: predicted_labels and true_labels must have the same dimensions")

        # Log unique values found in the label arrays
        logging.info(f"Unique values in predicted_labels: {numpy.unique(predicted_labels)}")
        logging.info(f"Unique values in true_labels: {numpy.unique(true_labels)}")

        # Check if the labels are binary or multiclass
        unique_pred = numpy.unique(predicted_labels)
        unique_true = numpy.unique(true_labels)

        if len(unique_pred) > 2 or len(unique_true) > 2:
            logging.info("Detected multiclass labels.")
        else:
            # Check for binary (0 or 1) labels
            if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
                logging.error("Predicted labels are not binary.")
                raise AccuracyError("Error: predicted_labels must be binary (0 or 1) or multiclass")

            if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
                logging.error("True labels are not binary.")
                raise AccuracyError("Error: true_labels must be binary (0 or 1) or multiclass")