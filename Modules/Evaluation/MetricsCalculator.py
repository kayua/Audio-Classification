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
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix

    from typing import List
    from typing import Dict
    from typing import Tuple
    from typing import Optional
    from typing import Union

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
    """

    @staticmethod
    def calculate_accuracy(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the accuracy of predictions.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Accuracy score.
        """
        try:
            return accuracy_score(label_true, label_predicted)
        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the precision of predictions.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Precision score.
        """
        try:
            return precision_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the recall of predictions.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: Recall score.
        """
        try:
            return recall_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true: List[int], label_predicted: List[int]) -> float:
        """
        Calculate the F1 score of predictions.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        float: F1 score.
        """
        try:
            return f1_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true: List[int], label_predicted: List[int]) -> Union[List[List[int]], None]:
        """
        Calculate the confusion matrix.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.

        Returns:
        List[List[int]]: Confusion matrix.
        """

        return confusion_matrix(label_true, label_predicted).tolist()


    def calculate_metrics(self, label_true: List[int], label_predicted: List[int],
                          label_predicted_probability: Optional[List[float]] = None
                          ) -> Tuple[Dict[str, float], Union[List[List[int]], None]]:

        """
        Calculate a set of metrics for classification.

        Parameters:
        label_true (List[int]): True labels.
        label_predicted (List[int]): Predicted labels.
        label_predicted_probability (Optional[List[float]]): Predicted probabilities.

        Returns:
        Tuple[Dict[str, float], Union[List[List[int]], None]]:
            Dictionary of metrics and the confusion matrix.
        """

        metrics = {}

        metrics['Accuracy'] = self.calculate_accuracy(label_true, label_predicted)
        metrics['Precision'] = self.calculate_precision(label_true, label_predicted)
        metrics['Recall'] = self.calculate_recall(label_true, label_predicted)
        metrics['F1-Score'] = self.calculate_f1_score(label_true, label_predicted)

        confusion_matrix_result = self.calculate_confusion_matrix(label_true, label_predicted)

        return metrics, confusion_matrix_result
