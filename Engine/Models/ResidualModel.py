#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

try:
    import os
    import sys
    import glob
    import numpy
    import librosa
    import tensorflow

    from tqdm import tqdm
    import librosa.display
    from tensorflow.keras import Model
    from sklearn.utils import resample
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import MaxPooling2D
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:

    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_FILTERS_PER_BLOCK = [16, 32, 64, 96]


class ResidualModel(MetricsCalculator):

    def __init__(self,
                 sample_rate: int,
                 hop_length: int,
                 window_size_factor: int,
                 number_filters_spectrogram: int,
                 number_layers: int,
                 input_dimension: tuple[int, int, int],
                 overlap: int,
                 convolutional_padding: str,
                 intermediary_activation: str,
                 last_layer_activation: str,
                 number_classes: int,
                 size_convolutional_filters: tuple[int, int],
                 size_pooling: tuple[int, int],
                 window_size_fft: int,
                 decibel_scale_factor: int,
                 filters_per_block: list[int],
                 size_batch: int,
                 number_splits: int,
                 number_epochs: int,
                 loss_function: str,
                 optimizer_function: str,
                 dropout_rate: float,
                 file_extension: str):


        if filters_per_block is None:
            filters_per_block = DEFAULT_FILTERS_PER_BLOCK

        self.model_name = "ResidualModel"
        self.neural_network_model = None
        self.sample_rate = sample_rate
        self.size_batch = size_batch
        self.number_splits = number_splits
        self.loss_function = loss_function
        self.size_pooling = size_pooling
        self.filters_per_block = filters_per_block
        self.hop_length = hop_length
        self.decibel_scale_factor = decibel_scale_factor
        self.window_size_fft = window_size_fft
        self.window_size_factor = window_size_factor
        self.window_size = hop_length * (self.window_size_factor - 1)
        self.number_filters_spectrogram = number_filters_spectrogram
        self.number_layers = number_layers
        self.input_shape = input_dimension
        self.overlap = overlap
        self.number_epochs = number_epochs
        self.optimizer_function = optimizer_function
        self.dropout_rate = dropout_rate
        self.file_extension = file_extension
        self.size_convolutional_filters = size_convolutional_filters
        self.number_classes = number_classes
        self.last_layer_activation = last_layer_activation
        self.convolutional_padding = convolutional_padding
        self.intermediary_activation = intermediary_activation

    def build_model(self):

        inputs = Input(shape=self.input_shape)
        neural_network_flow = inputs

        for number_filters in self.filters_per_block:
            residual_flow = neural_network_flow

            # Apply convolutional layers
            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)
            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)

            # Add residual connection
            neural_network_flow = Concatenate()([neural_network_flow, residual_flow])

            # Apply pooling and dropout
            neural_network_flow = MaxPooling2D(self.size_pooling)(neural_network_flow)
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        # Flatten and apply dense layer
        neural_network_flow = Flatten()(neural_network_flow)
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)

        # Define the model
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)


    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

    def train(self, dataset_directory, number_epochs, batch_size, number_splits,
              loss, sample_rate, overlap, number_classes, arguments) -> tuple:

        # Use default values if not provided
        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or self.overlap
        self.number_classes = number_classes or self.number_classes

        self.size_pooling = arguments.residual_size_pooling
        self.filters_per_block = arguments.residual_filters_per_block
        self.hop_length = arguments.residual_hop_length
        self.decibel_scale_factor = arguments.residual_decibel_scale_factor
        self.window_size_factor = arguments.residual_window_size_factor
        self.window_size = self.hop_length * (self.window_size_factor - 1)
        self.number_filters_spectrogram = arguments.residual_number_filters_spectrogram
        self.number_layers = arguments.residual_number_layers
        self.overlap = arguments.residual_overlap
        self.dropout_rate = arguments.residual_dropout_rate
        self.size_convolutional_filters = arguments.residual_size_convolutional_filters
        self.last_layer_activation = arguments.residual_last_layer_activation
        self.convolutional_padding = arguments.residual_convolutional_padding
        self.intermediary_activation = arguments.residual_intermediary_activation

        history_model = None
        features, labels = self.load_data(dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_val, features_test, labels_train_val, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance training/validation set
        features_train_val, labels_train_val = balance_classes(features_train_val, labels_train_val)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list = []
        real_labels_list = []

        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):

            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the training set for this fold
            features_train, labels_train = balance_classes(features_train, labels_train)

            self.build_model()
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_val, labels_val))

            model_predictions = self.neural_network_model.predict(features_val)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_val)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val, predicted_labels)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)

        # Calculate mean metrics across all folds
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

        probabilities_predicted = {
            'model_name': self.model_name,
            'predicted': numpy.concatenate(probabilities_list),
            'ground_truth': numpy.concatenate(real_labels_list)
        }

        confusion_matrix_array = numpy.array(confusion_matriz_list)
        mean_confusion_matrix = numpy.mean(confusion_matrix_array, axis=0)
        mean_confusion_matrix = numpy.round(mean_confusion_matrix).astype(numpy.int32).tolist()

        mean_confusion_matrices = {
            "confusion_matrix": mean_confusion_matrix,
            "class_names": ['Class {}'.format(i) for i in range(self.number_classes)],
            "title": self.model_name
        }

        return (mean_metrics, {"Name": self.model_name, "History": history_model.history}, mean_confusion_matrices,
                probabilities_predicted)

