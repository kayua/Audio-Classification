#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from AST import AudioAST

try:
    import sys
    import numpy
    import math
    import gc

    import seaborn as sns
    from typing import List
    from typing import Dict
    from typing import Tuple
    from typing import Union
    from typing import Optional

    from LSTM import AudioLSTM
    from Conformer import Conformer
    from Wav2Vec2 import AudioWav2Vec2
    from ResidualModel import ResidualModel

    import matplotlib.pyplot as plt
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


class EvaluationModels:

    def __init__(self):
        pass

    @staticmethod
    def calculate_accuracy(label_true: List[int], label_predicted: List[int]) -> float:
        try:
            return accuracy_score(label_true, label_predicted)
        except Exception as e:
            raise ValueError(f"Error calculating accuracy: {e}")

    @staticmethod
    def calculate_precision(label_true: List[int], label_predicted: List[int]) -> float:
        try:
            return precision_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating precision: {e}")

    @staticmethod
    def calculate_recall(label_true: List[int], label_predicted: List[int]) -> float:
        try:
            return recall_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating recall: {e}")

    @staticmethod
    def calculate_f1_score(label_true: List[int], label_predicted: List[int]) -> float:
        try:
            return f1_score(label_true, label_predicted, average='weighted')
        except Exception as e:
            raise ValueError(f"Error calculating F1 score: {e}")

    @staticmethod
    def calculate_auc(label_true: List[int], label_predicted_probability: List[float]) -> float:
        try:
            return roc_auc_score(label_true, label_predicted_probability, multi_class='ovr')
        except Exception as e:
            raise ValueError(f"Error calculating AUC: {e}")

    @staticmethod
    def calculate_confusion_matrix(label_true: List[int], label_predicted: List[int]) -> Union[List[List[int]], None]:
        try:
            return confusion_matrix(label_true, label_predicted).tolist()
        except Exception as e:
            raise ValueError(f"Error calculating confusion matrix: {e}")

    def calculate_metrics(self, label_true: List[int], label_predicted: List[int]
                          ) -> Tuple[Dict[str, float], Union[List[List[int]], None]]:

        metrics = {}

        try:
            metrics['accuracy'] = self.calculate_accuracy(label_true, label_predicted)
            metrics['precision'] = self.calculate_precision(label_true, label_predicted)
            metrics['recall'] = self.calculate_recall(label_true, label_predicted)
            metrics['f1_score'] = self.calculate_f1_score(label_true, label_predicted)

            confusion_matrix_result = self.calculate_confusion_matrix(label_true, label_predicted)

        except ValueError as e:
            print(f"An error occurred while calculating metrics: {e}")
            return {}, None

        return metrics, confusion_matrix_result

    @staticmethod
    def plot_comparative_metrics(dictionary_metrics_list, file_name):
        """
        Plots a comparative bar chart for accuracy, recall, f1-score, precision, and auc, and saves it to a file.

        Parameters:
            dictionary_metrics_list (List[Dict]): A list of dictionaries containing the list_metrics values,
                                       their standard deviations, and the model_name names.
            file_name (str): The path and name of the file to save the plot.
        """
        list_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        base_colors = {'Accuracy': 'Blues', 'Precision': 'Greens', 'Recall': 'Reds', 'F1-Score': 'Purples'}

        number_metrics = len(list_metrics)
        number_models = len(dictionary_metrics_list)

        # Initialize the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set width of bar
        bar_width = 0.20

        # Set position of bar on X axis
        positions = numpy.arange(number_metrics)

        # Iterate through list_metrics and plot each
        for i, key_metric in enumerate(list_metrics):

            for j, model_name in enumerate(dictionary_metrics_list):

                values = model_name[key_metric]['value']
                stds = model_name[key_metric]['std']
                color = plt.get_cmap(base_colors[key_metric])(j / (number_models - 1))
                bar = ax.bar(positions[i] + j * bar_width, values, yerr=stds, color=color, width=bar_width,
                             edgecolor='grey', capsize=10,
                             label=f"{model_name['model_name']}" if i == 0 else "")

                # Add text on the top of each bar
                for rect in bar:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 10),
                                textcoords="offset points", ha='center', va='bottom')

        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_xticks([r + bar_width * (number_models - 1) / 2 for r in positions])
        ax.set_xticklabels(list_metrics)

        # Add labels
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Comparative Metrics', fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=number_models)

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def plot_confusion_matrices(confusion_matrix_list, file_name_path: str):
        """
        Plots multiple confusion matrices in a single figure.

        Parameters:
            :param file_name_path: The name for the file to save the plot.
            :param confusion_matrix_list: A list of dictionaries containing confusion matrices.
        """

        number_matrices = len(confusion_matrix_list)

        # Determine the number of rows and columns for subplots
        if number_matrices <= 2:
            rows, cols = 1, number_matrices
        elif number_matrices == 3:
            rows, cols = 1, 3
        elif number_matrices == 4:
            rows, cols = 2, 2
        elif number_matrices == 5:
            rows, cols = 3, 2
        elif number_matrices == 6:
            rows, cols = 2, 3
        elif number_matrices == 7:
            rows, cols = 3, 3
        else:
            rows, cols = math.ceil(number_matrices / 4), 4

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))

        # Flatten axes array for easier iteration if there are multiple rows and columns
        axes = axes.flatten() if number_matrices > 1 else [axes]

        for index, confusion_matrix_dictionary in enumerate(confusion_matrix_list):
            confusion_matrix_instance = numpy.array(confusion_matrix_dictionary["confusion_matrix"])
            ax = axes[index]
            sns.heatmap(confusion_matrix_instance, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=confusion_matrix_dictionary["class_names"],
                        yticklabels=confusion_matrix_dictionary["class_names"],
                        annot_kws={"size": 10})  # Adjust annotation font size

            ax.set_xlabel('Predicted Labels', fontsize=12)
            ax.set_ylabel('True Labels', fontsize=12)
            title = confusion_matrix_dictionary.get("title", f'Confusion Matrix {index + 1}')
            ax.set_title(title, fontsize=14)

        # Remove any unused subplots
        for i in range(number_matrices, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(file_name_path)
        plt.close()

    @staticmethod
    def plot_and_save_loss(history_dict_list):
        for history_dict in history_dict_list:
            model_name = history_dict['Name']
            history = history_dict['History']

            if 'loss' not in history:
                continue

            plt.figure(figsize=(10, 6))

            for metric in history:
                if 'loss' in metric:
                    plt.plot(history['loss'], label=f'Loss - {metric}')

            plt.title(f'Loss para o modelo {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.savefig(f'{model_name}_loss.png')
            plt.close()


InstanceEvaluation = EvaluationModels()

Wav2Vec2_instance = AudioWav2Vec2()
mean_metrics_Wav2Vec2, mean_history_Wav2Vec2, mean_matrices_Wav2Vec2 = Wav2Vec2_instance.train('Dataset')
Wav2Vec2_instance = None
gc.collect()

LSTM_instance = AudioLSTM()
mean_metrics_LSTM, mean_history_LSTM, mean_matrices_LSTM = LSTM_instance.train('Dataset')
LSTM_instance = None
gc.collect()

Conformer_instance = Conformer()
mean_metrics_Conformer, mean_history_Conformer, mean_matrices_Conformer = Conformer_instance.train('Dataset')
Conformer_instance = None
gc.collect()

Residual_instance = ResidualModel()
mean_metrics_Residual, mean_history_Residual, mean_matrices_Residual = Residual_instance.train('Dataset')
Residual_instance = None
gc.collect()

AST_instance = AudioAST()
mean_metrics_AST, mean_history_AST, mean_matrices_AST = AST_instance.train('Dataset')
AST_instance = None
gc.collect()

mean_metrics = [mean_metrics_Wav2Vec2, mean_metrics_LSTM, mean_metrics_Conformer,
                mean_metrics_Residual, mean_metrics_AST]

mean_history = [mean_history_Wav2Vec2, mean_history_LSTM, mean_history_Conformer,
                mean_history_Residual, mean_history_AST]

mean_matrices = [mean_matrices_Wav2Vec2, mean_matrices_LSTM, mean_matrices_Conformer,
                 mean_matrices_Residual, mean_matrices_AST]

InstanceEvaluation.plot_confusion_matrices(mean_matrices, "confusion_matrices.png")
InstanceEvaluation.plot_comparative_metrics(mean_metrics, "metrics.png")
InstanceEvaluation.plot_and_save_loss(mean_history)
