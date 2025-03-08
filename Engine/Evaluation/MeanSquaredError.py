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
    from Engine.Exception.MetricsException import MeanSquareEError


except ImportError as error:
    print(error)
    sys.exit(-1)

class MeanSquareError:
    """
    A class for calculating the Mean Squared Error (MSE) between predicted and true labels.

    Attributes:
        None

    Methods:
        calculate_mean_square_error(predicted_labels, true_labels):
            Calculate the Mean Squared Error (MSE) between predicted labels and true labels.

    Example:
        # Define true labels and predicted labels
        true_labels = [2.0, 3.0, 4.0, 5.0]
        predicted_labels = [1.8, 2.9, 3.8, 4.9]

        # Calculate the MSE using the MeanSquareError class
        mse_calculator = MeanSquareError()
        mse = mse_calculator.get_mean_square_error(predicted_labels, true_labels)

        # Print the MSE value
        print(f"Mean Squared Error (MSE): {mse}")
    """
    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate the Mean Squared Error (MSE) between predicted labels and true labels.

        Args:
            predicted_labels: List of predicted labels.
            true_labels: List of true labels.

        Returns:
            float: The Mean Squared Error (MSE) as a floating-point number.

        Raises:
            MeanSquareError: Custom exception class for handling MSE calculation errors.
        """
        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:

            # Calculate squared errors for each prediction
            squared_errors = [(y - true) ** 2 for y, true in zip(predicted_labels, true_labels)]

            # Calculate the Mean Squared Error (MSE)
            mean_square_error = sum(squared_errors) / len(predicted_labels)

            return float(mean_square_error)

        except MeanSquareEError as e:
            return f"MSE Error: {e}"

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
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                        "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, np.ndarray):
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                        " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                        " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, np.ndarray):
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                        " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                        "have the same dimensions but are assigned different "
                                                        "dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not np.all(np.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                        " array composed of values 0 and 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not np.all(np.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an MeanSquareEError with an error message
            raise MeanSquareEError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                        " composed of values 0 and 1, but given different values")
