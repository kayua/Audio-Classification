#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

try:
    import sys

    import logging

    from sklearn.metrics import f1_score

    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import confusion_matrix

except ImportError as error:
    print(error)
    sys.exit(-1)


class Metrics:
    """
    A class for calculating various performance metrics for classification models.

    This class provides static methods to calculate the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - AUC (Area Under the Curve)
        - Confusion Matrix

    Additionally, the class provides a method to calculate all the above metrics in one go along with
    the confusion matrix. Each method has exception handling to ensure that errors are caught and
    informative messages are logged.

    Methods:
    -------
        @calculate_accuracy(label_true, label_predicted): Calculate the accuracy of the model's predictions.
        @calculate_precision(label_true, label_predicted): Calculate the precision of the model's predictions.
        @calculate_recall(label_true, label_predicted): Calculate the recall of the model's predictions.
        @calculate_f1_score(label_true, label_predicted): Calculate the F1 score of the model's predictions.
        @calculate_auc(label_true, label_predicted_probability): Calculate the Area Under the Curve (AUC)
         for multi-class classification.
        @calculate_confusion_matrix(label_true, label_predicted):
        @Calculate the confusion matrix of the model's predictions.
        @calculate_metrics(label_true, label_predicted): Calculate all the above metrics and the confusion matrix in one go.
    """

    @staticmethod
    def calculate_accuracy(label_true, label_predicted):
        """
        Calculate the accuracy score of the model's predictions.

        Accuracy is the ratio of the number of correct predictions to the total number of predictions.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        float
            The accuracy score, which is a value between 0 and 1, where 1 indicates perfect accuracy.

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of accuracy.
        """
        try:
            return accuracy_score(label_true, label_predicted)

        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true, label_predicted):
        """
        Calculate the precision score of the model's predictions.

        Precision is the ratio of true positive predictions to the total predicted positives.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        float
            The precision score, where higher values indicate better precision.

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of precision.
        """
        try:
            return precision_score(label_true, label_predicted, average='weighted')

        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true, label_predicted):
        """
        Calculate the recall score of the model's predictions.

        Recall is the ratio of true positive predictions to the total actual positives.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        float
            The recall score, where higher values indicate better recall.

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of recall.
        """

        try:
            return recall_score(label_true, label_predicted, average='weighted')

        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true, label_predicted):
        """
        Calculate the F1 score of the model's predictions.

        F1 Score is the harmonic mean of precision and recall, providing a balance between the two.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        float
            The F1 score, where higher values indicate better balance between precision and recall.

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of F1 score.
        """

        try:
            return f1_score(label_true, label_predicted, average='weighted')

        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_auc(label_true, label_predicted_probability):
        """
        Calculate the Area Under the Curve (AUC) for the ROC curve.

        AUC is a measure of how well the model distinguishes between classes. It is particularly useful
        in binary and multi-class classification problems.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted_probability : array-like of shape (n_samples, n_classes)
            Predicted class probabilities by the model.

        Returns:
        -------
        float
            The AUC score, where a higher value indicates better classification performance.

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of AUC.
        """

        try:
            return roc_auc_score(label_true, label_predicted_probability, multi_class='ovr')

        except Exception as e:
            raise ValueError(f"Error calculating AUC: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true, label_predicted):
        """
        Calculate the confusion matrix for the model's predictions.

        The confusion matrix shows the counts of actual vs predicted values, providing insight
        into how well the model distinguishes between classes.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        list of lists
            The confusion matrix as a 2D list (rows represent actual classes, columns represent predicted classes).

        Raises:
        ------
        ValueError
            If an error occurs during the calculation of the confusion matrix.
        """
        try:
            return confusion_matrix(label_true, label_predicted).tolist()

        except Exception as e:
            raise ValueError(f"Error calculating confusion matrix: {e}")

    def calculate_metrics(self, label_true, label_predicted):
        """
        Calculate multiple evaluation metrics and confusion matrix at once.

        This method sequentially calculates accuracy, precision, recall, F1 score, and the confusion matrix.
        It logs the progress of each calculation and handles errors by logging them appropriately.

        Parameters:
        ----------
        label_true : array-like of shape (n_samples,)
            True labels.

        label_predicted : array-like of shape (n_samples,)
            Predicted labels by the model.

        Returns:
        -------
        dict
            A dictionary containing the calculated metrics (accuracy, precision, recall, F1 score).

        list
            The confusion matrix as a 2D list.

        If any error occurs during the calculations, an empty dictionary and `None` for the confusion matrix
        are returned along with an error message in the logs.
        """
        metrics = {}
        logging.info("Starting to calculate metrics.")

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



        logging.info("Metric calculation completed successfully.")
        return metrics, confusion_matrix_result