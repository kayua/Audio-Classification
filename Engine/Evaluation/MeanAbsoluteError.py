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
    from Engine.Exception.MetricsException import MAEError

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

class MeanAbsoluteError:
    """
    A class for calculating the Mean Absolute Error (MAE) between predicted and true labels.

    Attributes:
        None

    Methods:
        get_mean_absolute_error(predicted_labels, true_labels):
            Calculate the Mean Absolute Error (MAE) between predicted labels and true labels.

    Example:
        # Define true labels and predicted labels
        true_labels = [2.0, 3.0, 4.0, 5.0]
        predicted_labels = [1.8, 2.9, 3.8, 4.9]

        # Calculate the MAE using the MeanAbsoluteError class
        mae_calculator = MeanAbsoluteError()
        mae = mae_calculator.get_mean_absolute_error(predicted_labels, true_labels)

        # Print the MAE value
        print(f"Mean Absolute Error (MAE): {mae}")
    """
    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate the Mean Absolute Error (MAE) between predicted labels and true labels.

        Args:
            predicted_labels: List of predicted labels.
            true_labels: List of true labels.

        Returns:
            float: The Mean Absolute Error (MAE) as a floating-point number.

        Raises:
            MAEError: Custom exception class for handling MAE calculation errors.
        """
        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)
        try:

            # Calculate absolute errors for each prediction
            absolute_errors = [abs(y - true) for y, true in zip(predicted_labels, true_labels)]

            # Calculate the Mean Absolute Error (MAE)
            mean_absolute_error = sum(absolute_errors) / len(predicted_labels)

            return mean_absolute_error

        except MAEError as e:
            # Handle the case where a MAEError is raised and return an error message
            return f"MAE Error: {e}"

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            AccuracyError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an MAEError with an error message
            raise MAEError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, np.ndarray):
            # Raise an MeanSquareEError with an error message
            raise MAEError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an MeanSquareEError with an error message
            raise MAEError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, np.ndarray):
            # Raise an MAEError with an error message
            raise MAEError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an MAEError with an error message
            raise MAEError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                "have the same dimensions but are assigned different "
                                                "dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not np.all(np.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an MAEError with an error message
            raise MAEError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                " array composed of values 0 and 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not np.all(np.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an MAEError with an error message
            raise MAEError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                " composed of values 0 and 1, but given different values")
