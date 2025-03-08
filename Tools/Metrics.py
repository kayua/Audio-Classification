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

    @staticmethod
    def calculate_accuracy(label_true, label_predicted):
        try:
            return accuracy_score(label_true, label_predicted)
        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true, label_predicted):
        try:
            return precision_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true, label_predicted):
        try:
            return recall_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true, label_predicted):
        try:
            return f1_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_auc(label_true, label_predicted_probability):
        try:
            return roc_auc_score(label_true, label_predicted_probability, multi_class='ovr')
        except Exception as e:
            raise ValueError(f"Error calculating AUC: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true, label_predicted):
        try:
            return confusion_matrix(label_true, label_predicted).tolist()
        except Exception as e:
            raise ValueError(f"Error calculating confusion matrix: {e}")

    def calculate_metrics(self, label_true, label_predicted):
        metrics = {}
        logging.info("Starting to calculate metrics.")

        try:
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

        except ValueError as e:
            logging.error(f"An error occurred while calculating metrics: {e}")
            return {}, None

        logging.info("Metric calculation completed successfully.")
        return metrics, confusion_matrix_result