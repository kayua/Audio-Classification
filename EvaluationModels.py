#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/26'
__credits__ = ['unknown']

try:
    import sys
    import numpy as np
    import gc
    import subprocess
    import argparse
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from Models.AST import AudioAST
    from Models.LSTM import AudioLSTM
    from Models.MLP import AudioDense
    from Models.Conformer import Conformer
    from Models.Wav2Vec2 import AudioWav2Vec2
    from Models.ResidualModel import ResidualModel
except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt")
    sys.exit(-1)

DEFAULT_MATRIX_FIGURE_SIZE = (5, 5)
DEFAULT_MATRIX_COLOR_MAP = 'Blues'
DEFAULT_MATRIX_ANNOTATION_FONT_SIZE = 10
DEFAULT_MATRIX_LABEL_FONT_SIZE = 12
DEFAULT_MATRIX_TITLE_FONT_SIZE = 14
DEFAULT_SHOW_PLOT = False
DEFAULT_DATASET_DIRECTORY = "Dataset/"
DEFAULT_NUMBER_EPOCHS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUMBER_SPLITS = 2
DEFAULT_LOSS = 'sparse_categorical_crossentropy'
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_OVERLAP = 2
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_OUTPUT_DIRECTORY = "Results/"
DEFAULT_PLOT_WIDTH = 14
DEFAULT_PLOT_HEIGHT = 8
DEFAULT_PLOT_BAR_WIDTH = 0.15
DEFAULT_PLOT_CAP_SIZE = 10


class EvaluationModels:

    def __init__(self):
        self.mean_metrics = []
        self.mean_history = []
        self.mean_matrices = []

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
    def plot_comparative_metrics(dictionary_metrics_list, file_name, width=12, height=8, bar_width=0.20, cap_size=10):
        list_metrics = ['Acc.', 'Prec.', 'Rec.', 'F1.']
        base_colors = {'Acc.': 'Blues', 'Prec.': 'Greens', 'Rec.': 'Reds', 'F1.': 'Purples'}
        number_metrics = len(list_metrics)
        number_models = len(dictionary_metrics_list)
        fig, ax = plt.subplots(figsize=(width, height))
        positions = np.arange(number_metrics)

        for i, key_metric in enumerate(list_metrics):
            for j, model_name in enumerate(dictionary_metrics_list):
                values = model_name[key_metric]['value']
                stds = model_name[key_metric]['std']
                color = plt.get_cmap(base_colors[key_metric])(j / (number_models - 1))
                label = f"{key_metric} {model_name['model_name']}"
                bar = ax.bar(positions[i] + j * bar_width, values, yerr=stds, color=color, width=bar_width,
                             edgecolor='grey', capsize=cap_size, label=label)

                for rect in bar:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 10),
                                textcoords="offset points", ha='center', va='bottom')

        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_xticks([r + bar_width * (number_models - 1) / 2 for r in positions])
        ax.set_xticklabels(list_metrics)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Comparative Metrics', fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=number_models)
        plt.tight_layout()
        plt.savefig(f'{file_name}metrics.png')
        plt.close()

    @staticmethod
    def plot_confusion_matrices(confusion_matrix_list, file_name_path, fig_size=(5, 5), cmap='Blues',
                                annot_font_size=10, label_font_size=12, title_font_size=14, show_plot=False):
        for index, confusion_matrix_dictionary in enumerate(confusion_matrix_list):
            confusion_matrix_instance = np.array(confusion_matrix_dictionary["confusion_matrix"])
            plt.figure(figsize=fig_size)
            sns.heatmap(
                confusion_matrix_instance,
                annot=True,
                fmt='d',
                cmap=cmap,
                xticklabels=confusion_matrix_dictionary["class_names"],
                yticklabels=confusion_matrix_dictionary["class_names"],
                annot_kws={"size": annot_font_size}
            )
            plt.xlabel('Predicted Labels', fontsize=label_font_size)
            plt.ylabel('True Labels', fontsize=label_font_size)
            title = confusion_matrix_dictionary.get("title", f'Confusion Matrix {index + 1}')
            plt.title(title, fontsize=title_font_size)
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
    def train_and_collect_metrics(model_class, dataset_directory, number_epochs, batch_size, number_splits, loss,
                                  sample_rate, overlap, number_classes):
        instance = model_class()
        metrics, history, matrices = instance.train(dataset_directory, number_epochs, batch_size, number_splits,
                                                    loss, sample_rate, overlap, number_classes)
        gc.collect()
        return metrics, history, matrices

    def run(self, models, dataset_directory, number_epochs, batch_size, number_splits, loss, sample_rate, overlap,
            number_classes, output_directory, plot_width, plot_height, plot_bar_width, plot_cap_size):
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

        self.plot_comparative_metrics(dictionary_metrics_list=self.mean_metrics,
                                      file_name=output_directory,
                                      width=plot_width,
                                      height=plot_height,
                                      bar_width=plot_bar_width,
                                      cap_size=plot_cap_size)

        self.plot_confusion_matrices(confusion_matrix_list=self.mean_matrices,
                                     file_name_path=output_directory,
                                     fig_size=DEFAULT_MATRIX_FIGURE_SIZE,
                                     cmap=DEFAULT_MATRIX_COLOR_MAP,
                                     annot_font_size=DEFAULT_MATRIX_ANNOTATION_FONT_SIZE,
                                     label_font_size=DEFAULT_MATRIX_LABEL_FONT_SIZE,
                                     title_font_size=DEFAULT_MATRIX_TITLE_FONT_SIZE,
                                     show_plot=DEFAULT_SHOW_PLOT)

        self.plot_and_save_loss(history_dict_list=self.mean_history, path_output=output_directory)
        self.run_python_script('--output', "Results.pdf")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation with metrics and confusion matrices.")
    parser.add_argument("--dataset_directory", type=str, default=DEFAULT_DATASET_DIRECTORY,
                        help="Directory containing the dataset.")
    parser.add_argument("--number_epochs", type=int, default=DEFAULT_NUMBER_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Size of the batches for training.")
    parser.add_argument("--number_splits", type=int, default=DEFAULT_NUMBER_SPLITS,
                        help="Number of splits for cross-validation.")
    parser.add_argument("--loss", type=str, default=DEFAULT_LOSS, help="Loss function to use during training.")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate of the audio files.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap for the audio segments.")
    parser.add_argument("--number_classes", type=int, default=DEFAULT_NUMBER_CLASSES,
                        help="Number of classes in the dataset.")
    parser.add_argument("--output_directory", type=str, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="Directory to save output files.")
    parser.add_argument("--plot_width", type=float, default=DEFAULT_PLOT_WIDTH, help="Width of the plots.")
    parser.add_argument("--plot_height", type=float, default=DEFAULT_PLOT_HEIGHT, help="Height of the plots.")
    parser.add_argument("--plot_bar_width", type=float, default=DEFAULT_PLOT_BAR_WIDTH,
                        help="Width of the bars in the bar plots.")
    parser.add_argument("--plot_cap_size", type=float, default=DEFAULT_PLOT_CAP_SIZE,
                        help="Capsize of the error bars in the bar plots.")

    args = parser.parse_args()

    available_models = [AudioAST, AudioLSTM, AudioDense, Conformer, AudioWav2Vec2, ResidualModel]

    evaluation = EvaluationModels()
    evaluation.run(models=available_models,
                   dataset_directory=args.dataset_directory,
                   number_epochs=args.number_epochs,
                   batch_size=args.batch_size,
                   number_splits=args.number_splits,
                   loss=args.loss,
                   sample_rate=args.sample_rate,
                   overlap=args.overlap,
                   number_classes=args.number_classes,
                   output_directory=args.output_directory,
                   plot_width=args.plot_width,
                   plot_height=args.plot_height,
                   plot_bar_width=args.plot_bar_width,
                   plot_cap_size=args.plot_cap_size)
