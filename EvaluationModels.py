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
    import numpy
    import math
    import gc
    import subprocess
    import seaborn as sns
    from typing import List
    from typing import Dict
    from typing import Tuple
    from typing import Union
    from typing import Optional

    from Models.AST import AudioAST
    from Models.LSTM import AudioLSTM
    from Models.MLP import AudioDense
    from Models.Conformer import Conformer
    from Models.Wav2Vec2 import AudioWav2Vec2
    from Models.ResidualModel import ResidualModel

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

DEFAULT_MATRIX_FIGURE_SIZE = (5, 5)
DEFAULT_MATRIX_COLOR_MAP = 'Blues'
DEFAULT_MATRIX_ANNOTATION_FONT_SIZE = 10
DEFAULT_MATRIX_LABEL_FONT_SIZE = 12
DEFAULT_MATRIX_TITLE_FONT_SIZE = 14
DEFAULT_SHOW_PLOT = False


class EvaluationModels:

    def __init__(self):

        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []
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
        plt.savefig(f'{file_name}metrics.png')
        plt.close()

    @staticmethod
    def plot_confusion_matrices(confusion_matrix_list, file_name_path: str, fig_size: tuple = (5, 5),
                                cmap: str = 'Blues', annot_font_size: int = 10, label_font_size: int = 12,
                                title_font_size: int = 14, show_plot: bool = False):
        """
        Plots each confusion matrix separately and saves each plot to a file.

        Parameters:
            :param confusion_matrix_list: A list of dictionaries containing confusion matrices.
            :param file_name_path: The file path for saving the plots. The file names will be generated based on this path.
            :param fig_size: Size of the figure for each subplot (width, height).
            :param cmap: Color map to use for the heatmap.
            :param annot_font_size: Font size for the annotations in the heatmap.
            :param label_font_size: Font size for the axis labels.
            :param title_font_size: Font size for the subplot titles.
            :param show_plot: Whether to display the plot or not.
        """

        # Loop through the list of confusion matrices
        for index, confusion_matrix_dictionary in enumerate(confusion_matrix_list):
            confusion_matrix_instance = numpy.array(confusion_matrix_dictionary["confusion_matrix"])

            plt.figure(figsize=fig_size)
            sns.heatmap(
                confusion_matrix_instance,
                annot=True,
                fmt='d',
                cmap=cmap,
                xticklabels=confusion_matrix_dictionary["class_names"],
                yticklabels=confusion_matrix_dictionary["class_names"],
                annot_kws={"size": annot_font_size}  # Adjust annotation font size
            )

            plt.xlabel('Predicted Labels', fontsize=label_font_size)
            plt.ylabel('True Labels', fontsize=label_font_size)
            title = confusion_matrix_dictionary.get("title", f'Confusion Matrix {index + 1}')
            plt.title(title, fontsize=title_font_size)

            # Save each plot to a file
            file_path = f"{file_name_path}matrix_{index + 1}.png"
            plt.tight_layout()
            if show_plot:
                plt.show()
            else:
                plt.savefig(file_path)
            plt.close()

    @staticmethod
    def plot_and_save_loss(history_dict_list, path_output):
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

            plt.savefig(f'{path_output}{model_name}_loss.png')
            plt.close()

    @staticmethod
    def train_and_collect_metrics(model_class, dataset_training_evaluation):
        instance = model_class()
        metrics, history, matrices = instance.train(dataset_training_evaluation)
        gc.collect()
        return metrics, history, matrices

    def run(self, models, dataset_directory, number_epochs=2, batch_size=32, number_splits=2,
            loss='sparse_categorical_crossentropy', sample_rate=8000, overlap=2, number_classes=4):

        for model_class in models:
            metrics, history, matrices = self.train_and_collect_metrics(model_class=model_class,
                                                                        dataset_directory=dataset_directory,
                                                                        number_epochs=number_epochs,
                                                                        batch_size=batch_size,
                                                                        number_splits=number_splits,
                                                                        loss=loss,
                                                                        sample_rate=sample_rate,
                                                                        overlap=overlap,
                                                                        number_classes=number_classes)
            self.mean_metrics.append(metrics)
            self.mean_history.append(history)
            self.mean_matrices.append(matrices)

        self.plot_confusion_matrices(self.mean_matrices, "Results/")
        self.plot_comparative_metrics(self.mean_metrics, "Results/")
        self.plot_and_save_loss(self.mean_history, "Results/")
        self.run_python_script('--output', "TestFile.pdf")

    def get_results(self):
        return self.mean_metrics, self.mean_history, self.mean_matrices

    @staticmethod
    def run_python_script(*args) -> int:
        try:
            # Prepare the command to run the script
            command = ['python3', 'GeneratePDF.py'] + list(args)

            # Execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Print standard output and standard error
            print("Standard Output:\n", result.stdout)
            if result.stderr:
                print("Standard Error:\n", result.stderr)

            return result.returncode

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the script: {e}")
            print("Standard Output:\n", e.stdout)
            print("Standard Error:\n", e.stderr)
            return e.returncode


InstanceEvaluation = EvaluationModels()

model_classes = [AudioWav2Vec2, AudioLSTM, Conformer, ResidualModel, AudioAST, AudioDense]

# Dataset
dataset = 'Dataset'

InstanceEvaluation.run(model_classes, dataset)
