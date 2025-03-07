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
    import numpy as np
    from Engine.Exception.MetricsException import RecallError

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

class Recall:
    """
    A class for calculating recall and validating input labels.

    Attributes:
        None

    Methods:
        calculate_recall(predicted_labels, true_labels):
            Calculate recall given predicted and true labels.

        _check_input_labels(predicted_labels, true_labels):
            Check the validity and type of input labels.

    Exceptions:
        RecallError: Custom exception class for handling recall calculation errors.

    Example:
        # Create an instance of the Recall class
        recall_calculator = Recall()

        # Define predicted and true labels as numpy arrays
        predicted_labels = np.array([1, 0, 1, 1, 0])
        true_labels = np.array([1, 1, 0, 1, 0])

        # Calculate and print the recall
        print(f"Recall: {recall_calculator.calculate_recall(predicted_labels, true_labels)}")
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate recall given predicted and true labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Returns:
            float: The recall as a floating-point number between 0 and 1.

        Raises:
            RecallError: Custom exception class for handling recall calculation errors.
        """
        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:
            # Count true positives and false negatives
            true_positive = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 1 and true == 1)
            false_negative = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 0 and true == 1)

            if true_positive + false_negative == 0:
                return 0.0

            # Calculate recall as the ratio of true positives to the sum of true positives and false negatives
            recall = true_positive / (true_positive + false_negative)

            return recall

        except RecallError as e:
            return f"Recall Error: {e}"

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            RecallError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an PrecisionError with an error message
            raise RecallError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                   "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, np.ndarray):
            # Raise an PrecisionError with an error message
            raise RecallError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                   " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an PrecisionError with an error message
            raise RecallError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                   " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, np.ndarray):
            # Raise an PrecisionError with an error message
            raise RecallError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                   " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an RecallError with an error message
            raise RecallError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                   " have the same dimensions but are assigned different dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not np.all(np.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an RecallError with an error message
            raise RecallError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                   " array composed of values 0 and 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not np.all(np.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an RecallError with an error message
            raise RecallError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                   " composed of values 0 and 1, but given different values")
