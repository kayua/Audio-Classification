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

    from typing import List
    from typing import Dict
    from typing import Tuple
    from typing import Optional
    from typing import Union

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class MetricsCalculator:
    """
    A class to calculate various classification metrics.

    This class provides methods to compute accuracy, precision, recall, F1-score,
    and confusion matrix for a given set of true and predicted labels.

    Example usage:
    >>> python3
    ...    true_labels = [0, 1, 1, 2, 2, 2]
    ...    predicted_labels = [0, 1, 0, 2, 2, 1]
    ...    calculator = MetricsCalculator()
    ...    metrics, conf_matrix = calculator.calculate_metrics(true_labels, predicted_labels)
    ...    print("Metrics:", metrics)
    ...    print("Confusion Matrix:", conf_matrix)
    ```
    """

    @staticmethod
    def calculate_accuracy(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the accuracy of predictions.

        Accuracy is computed as the ratio of correctly predicted instances to the total instances.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Accuracy score (range: 0.0 to 1.0).

        Example:
        >>> python
        ...    accuracy = MetricsCalculator.calculate_accuracy([1, 0, 1], [1, 0, 0])
        ...    print(accuracy)  # Output: 0.666...
        >>>
        """
        try:
            return accuracy_score(label_true, label_predicted)
        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the precision of predictions.

        Precision is the ratio of correctly predicted positive observations to the total predicted positives.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Precision score (weighted average for multi-class classification).
        """
        try:
            return precision_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the recall of predictions.

        Recall is the ratio of correctly predicted positive observations to all actual positives.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Recall score (weighted average for multi-class classification).
        """
        try:
            return recall_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the F1 score of predictions.

        F1-score is the weighted average of precision and recall.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: F1 score (weighted average for multi-class classification).
        """
        try:
            return f1_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true: List[int], label_predicted: List[int]) -> Union[List[List[int]], None]:
        """
        Calculate the confusion matrix.

        A confusion matrix summarizes the performance of a classification algorithm.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        List[List[int]]: Confusion matrix as a nested list.

        Example:
        >>> python
        ...    cm = MetricsCalculator.calculate_confusion_matrix([0, 1, 2, 2], [0, 0, 2, 2])
        ...    print(cm)
        >>>
        """
        return confusion_matrix(label_true, label_predicted).tolist()

    def calculate_metrics(self, label_true: List[int], label_predicted: List[int]) -> Tuple[Dict[str, float], List[List[int]]]:
        """
        Calculate a set of metrics for classification.

        This method computes accuracy, precision, recall, and F1-score, along with the confusion matrix.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        Tuple[Dict[str, float], List[List[int]]]:
            - A dictionary containing the computed metrics.
            - A confusion matrix as a nested list.

        Example:
        >>> python
        ...    calculator = MetricsCalculator()
        ...    metrics, conf_matrix = calculator.calculate_metrics([1, 0, 1], [1, 0, 0])
        ...    print(metrics)
        ...    print(conf_matrix)
        >>>
        """
        metrics = {
            'Accuracy': self.calculate_accuracy(label_true, label_predicted),
            'Precision': self.calculate_precision(label_true, label_predicted),
            'Recall': self.calculate_recall(label_true, label_predicted),
            'F1-Score': self.calculate_f1_score(label_true, label_predicted)
        }
        return metrics, self.calculate_confusion_matrix(label_true, label_predicted)
