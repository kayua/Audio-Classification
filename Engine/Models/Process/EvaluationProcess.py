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

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Processing.RawFeature import RawDataLoader
    from Engine.Processing.ClassBalance import ClassBalancer

    from Engine.Evaluation.MetricsCalculator import MetricsCalculator
    from Engine.Processing.SpectrogramFeature import SpectrogramFeature


except ImportError as error:
    print(error)
    sys.exit(-1)



class EvaluationProcess(MetricsCalculator,
                        ClassBalancer,
                        RawDataLoader,
                        SpectrogramFeature):
    """
    RawProcess class encapsulates the full workflow of loading, preprocessing,
    training, evaluating, and aggregating metrics for a machine learning model.
    It integrates multiple functionalities from the `MetricsCalculator` and
    `ClassBalancer` classes to handle data, perform training and evaluation
    tasks, and compute final metrics for model performance.

    Attributes:
        size_batch (int): The size of the batch for training.
        number_splits (int): The number of splits for cross-validation.
        number_epochs (int): The number of epochs for training the model.
        optimizer_function (str): The optimizer function used in the model.
        window_size_factor (int): A factor for calculating the window size for processing data.
        decibel_scale_factor (int): A scaling factor for converting audio signals to decibels.
        hop_length (int): The hop length used for signal processing (e.g., for audio).
        overlap (int): The overlap factor between windows in signal processing.
        window_size (int): The actual window size calculated from the hop length and window size factor.
        sample_rate (int): The sample rate for the data.
        file_extension (str): The file extension for the dataset (e.g., ".wav" for audio files).
    """

    def __init__(self, size_batch: int, number_splits: int, number_epochs: int, optimizer_function: str,
                 window_size_factor: int, decibel_scale_factor: int, hop_length: int, overlap: int, sample_rate: int,
                 file_extension: str):
        """
        Initializes the RawProcess class with all the necessary parameters for training and data
        processing.

        Args:
            size_batch (int): The size of the batch for training.
            number_splits (int): The number of splits for cross-validation.
            number_epochs (int): The number of epochs for training the model.
            optimizer_function (str): The optimizer function used in the model.
            window_size_factor (int): A factor for calculating the window size for processing data.
            decibel_scale_factor (int): A scaling factor for converting audio signals to decibels.
            hop_length (int): The hop length used for signal processing.
            overlap (int): The overlap factor between windows in signal processing.
            sample_rate (int): The sample rate for the data.
            file_extension (str): The file extension for the dataset.
        """
        # Assigning provided parameters to instance variables
        RawDataLoader.__init__(self, sample_rate, self.hop_length * self.window_size_factor, overlap,
                               window_size_factor, file_extension)

        self.size_batch = size_batch
        self.number_splits = number_splits
        self.number_epochs = number_epochs
        self.optimizer_function = optimizer_function
        self.window_size_factor = window_size_factor
        self.decibel_scale_factor = decibel_scale_factor
        self.hop_length = hop_length
        self.overlap = overlap
        self.window_size = self.hop_length * self.window_size_factor  # Calculate window size
        self.sample_rate = sample_rate
        self.file_extension = file_extension

    def load_and_preprocess_data(self, dataset_directory):
        """
        Loads and preprocesses the dataset, splitting it into training and testing sets.

        Args:
            dataset_directory (str): The directory containing the dataset to load.

        Returns:
            tuple: A tuple containing the preprocessed training and testing data.
                (features_train_val, features_test, labels_train_val, labels_test)
        """
        # Loading data from the provided directory and preprocessing
        features, labels = self.load_data_raw_format(dataset_directory)
        labels = numpy.array(labels).astype(float)  # Convert labels to float

        # Splitting data into training/validation and test sets (80-20 split)
        features_train_val, features_test, labels_train_val, labels_test = \
            train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

        # Balancing the test set classes
        features_test, labels_test = self.balance_class(features_test, labels_test)

        return features_train_val, features_test, labels_train_val, labels_test

    def split_and_balance_data(self):
        """
        Prepares the data for cross-validation, creating instances for splits and balancing classes.

        Returns:
            tuple: A tuple containing the StratifiedKFold instance, history model list,
                   probabilities list, and real labels list.
        """

        # Setting up Stratified K-fold cross-validation for splitting and balancing data
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)

        # Initialize lists to store model history, probabilities, and real labels
        list_history_model, probabilities_list, real_labels_list = [], [], []

        return instance_k_fold, list_history_model, probabilities_list, real_labels_list

    def train_fold(self, features_train, features_val, labels_train, labels_val, features_test):
        """
        Trains the model for a single fold of cross-validation.

        Args:
            features_train (numpy.ndarray): The training features for the fold.
            features_val (numpy.ndarray): The validation features for the fold.
            labels_train (numpy.ndarray): The training labels for the fold.
            labels_val (numpy.ndarray): The validation labels for the fold.
            features_test (numpy.ndarray): The test features for the fold.

        Returns:
            tuple: A tuple containing the history model, model predictions, and predicted labels.
        """

        # Build and summarize the model
        self.build_model()
        self.neural_network_model.summary()

        # Compile and train the model
        history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                               batch_size=self.size_batch,
                                               validation_data=(features_val, labels_val))

        # Get predictions from the model
        model_predictions = self.neural_network_model.predict(features_test)

        # Convert model predictions to predicted labels
        predicted_labels = numpy.argmax(model_predictions, axis=1)

        return history_model, model_predictions, predicted_labels

    def calculate_fold_metrics(self, predicted_labels, labels_val):
        """
        Calculates performance metrics and confusion matrix for a single fold.

        Args:
            predicted_labels (numpy.ndarray): The predicted labels for the fold.
            labels_val (numpy.ndarray): The actual labels for the fold.
        Returns:
            tuple: A tuple containing the calculated metrics and confusion matrix.
        """

        # Calculate various metrics and confusion matrix
        metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val, predicted_labels)

        return metrics, confusion_matrix

    def aggregate_results(self, metrics_list, confusion_matriz_list, probabilities_list, real_labels_list):
        """
        Aggregates the results from all cross-validation folds.

        Args:
            metrics_list (list): A list of metrics from each fold.
            confusion_matriz_list (list): A list of confusion matrices from each fold.
            probabilities_list (list): A list of predicted probabilities from each fold.
            real_labels_list (list): A list of real labels from the test set.

        Returns:
            tuple: A tuple containing the mean metrics, predicted probabilities, and mean confusion matrix.
        """

        # Calculate the mean and standard deviation for each metric
        mean_metrics = {
            'model_name': self.model_name,
            'Acc.': {'value': numpy.mean([metric['Accuracy'] for metric in metrics_list]),
                     'std': numpy.std([metric['Accuracy'] for metric in metrics_list])},
            'Prec.': {'value': numpy.mean([metric['Precision'] for metric in metrics_list]),
                      'std': numpy.std([metric['Precision'] for metric in metrics_list])},
            'Rec.': {'value': numpy.mean([metric['Recall'] for metric in metrics_list]),
                     'std': numpy.std([metric['Recall'] for metric in metrics_list])},
            'F1.': {'value': numpy.mean([metric['F1-Score'] for metric in metrics_list]),
                    'std': numpy.std([metric['F1-Score'] for metric in metrics_list])},
        }

        # Aggregate the predicted probabilities and real labels
        probabilities_predicted = {
            'model_name': self.model_name,
            'predicted': numpy.concatenate(probabilities_list),
            'ground_truth': numpy.concatenate(real_labels_list)
        }

        # Calculate the mean confusion matrix
        confusion_matrix_array = numpy.array(confusion_matriz_list)
        mean_confusion_matrix = numpy.mean(confusion_matrix_array, axis=0)
        mean_confusion_matrix = numpy.round(mean_confusion_matrix).astype(numpy.int32).tolist()

        mean_confusion_matrices = {
            "confusion_matrix": mean_confusion_matrix,
            "class_names": ['Class {}'.format(i) for i in range(self.number_classes)],
            "title": self.model_name
        }

        return mean_metrics, probabilities_predicted, mean_confusion_matrices

    def train(self, input_arguments) -> tuple:
        """
        Executes the entire training and evaluation process including data loading, preprocessing,
        model training, and metrics calculation.

        Returns:
            tuple: A tuple containing the aggregated mean metrics, model history, mean confusion matrix,
                   and predicted probabilities for the dataset.
        """

        # Load and preprocess the data
        features_train_val, features_test, labels_train_val, labels_test = \
            self.load_and_preprocess_data(input_arguments.dataset_directory)

        # Prepare data for cross-validation and model training
        instance_k_fold, list_history_model, probabilities_list, real_labels_list = \
            self.split_and_balance_data()

        # Initialize lists to store metrics and confusion matrices
        metrics_list, confusion_matriz_list = [], []

        # Loop over each fold of cross-validation
        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):

            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the class distribution in the training set
            features_train, labels_train = self.balance_class(features_train, labels_train)

            # Train the model for the fold and make predictions
            history_model, model_predictions, predicted_labels = self.train_fold(features_train,
                                                                                 features_val,
                                                                                 labels_train,
                                                                                 labels_val,
                                                                                 features_test)

            # Calculate metrics and confusion matrix for the fold
            metrics, confusion_matrix = self.calculate_fold_metrics(predicted_labels, labels_val)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)

            # Append model predictions and real labels for evaluation
            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_test)

        # Aggregate and return the results from all folds
        mean_metrics, probabilities_predicted, mean_confusion_matrices = self.aggregate_results(
            metrics_list, confusion_matriz_list, probabilities_list, real_labels_list)

        return (mean_metrics, {"Name": self.model_name, "History": history_model.history}, mean_confusion_matrices,
                probabilities_predicted)
