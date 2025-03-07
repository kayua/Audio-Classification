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
    import argparse
    import tensorflow

    from tqdm import tqdm
    from sklearn.utils import resample
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Bidirectional
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Modules.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_INPUT_DIMENSION = (40, 256)
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_LIST_LSTM_CELLS = [128, 129]
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_SIZE_BATCH = 32
DEFAULT_OVERLAP = 2
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_RECURRENT_ACTIVATION = 'sigmoid'
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'tanh'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


class AudioLSTM(MetricsCalculator):


    def __init__(self,
                 number_classes: int = DEFAULT_NUMBER_CLASSES,
                 last_layer_activation: str = DEFAULT_LAST_LAYER_ACTIVATION,
                 size_batch: int = DEFAULT_SIZE_BATCH,
                 number_splits: int = DEFAULT_NUMBER_SPLITS,
                 number_epochs: int = DEFAULT_NUMBER_EPOCHS,
                 loss_function: str = DEFAULT_LOSS_FUNCTION,
                 optimizer_function: str = DEFAULT_OPTIMIZER_FUNCTION,
                 window_size_factor: int = DEFAULT_WINDOW_SIZE_FACTOR,
                 decibel_scale_factor: int = DEFAULT_DECIBEL_SCALE_FACTOR,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 overlap: int = DEFAULT_OVERLAP,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 dropout_rate: float = DEFAULT_DROPOUT_RATE,
                 file_extension: str = DEFAULT_FILE_EXTENSION,
                 intermediary_layer_activation: str = DEFAULT_INTERMEDIARY_LAYER_ACTIVATION,
                 recurrent_activation: str = DEFAULT_RECURRENT_ACTIVATION,
                 input_dimension: tuple = DEFAULT_INPUT_DIMENSION,
                 list_lstm_cells=None):

        if list_lstm_cells is None:
            list_lstm_cells = DEFAULT_LIST_LSTM_CELLS

        self.model_name = "LSTM"
        self.neural_network_model = None
        self.size_batch = size_batch
        self.list_lstm_cells = list_lstm_cells
        self.number_splits = number_splits
        self.number_epochs = number_epochs
        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.window_size_factor = window_size_factor
        self.decibel_scale_factor = decibel_scale_factor
        self.hop_length = hop_length
        self.recurrent_activation = recurrent_activation
        self.intermediary_layer_activation = intermediary_layer_activation
        self.overlap = overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = sample_rate
        self.file_extension = file_extension
        self.input_dimension = input_dimension
        self.number_classes = number_classes
        self.dropout_rate = dropout_rate
        self.last_layer_activation = last_layer_activation
        self.model_name = "LSTM"

    def build_model(self) -> None:

        inputs = Input(shape=self.input_dimension)

        neural_network_flow = inputs

        for i, cells in enumerate(self.list_lstm_cells):
            neural_network_flow = LSTM(cells, activation=self.intermediary_layer_activation,
                                       recurrent_activation=self.recurrent_activation,
                                       return_sequences=True)(neural_network_flow)
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        neural_network_flow = GlobalAveragePooling1D()(neural_network_flow)
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow)

    def compile_and_train(self,
                          train_data: tensorflow.Tensor,
                          train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int,
                          validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history


    def train(self, dataset_directory, number_epochs, batch_size, number_splits,
              loss, sample_rate, overlap, number_classes, arguments) -> tuple:

        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or self.overlap
        self.number_classes = number_classes or self.number_classes

        self.list_lstm_cells = arguments.lstm_list_lstm_cells
        self.window_size_factor = arguments.lstm_window_size_factor
        self.decibel_scale_factor = arguments.lstm_decibel_scale_factor
        self.hop_length = arguments.lstm_hop_length
        self.recurrent_activation = arguments.lstm_recurrent_activation
        self.intermediary_layer_activation = arguments.lstm_intermediary_layer_activation
        self.overlap =  arguments.lstm_overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.number_classes = number_classes
        self.dropout_rate = arguments.lstm_dropout_rate
        self.last_layer_activation = arguments.lstm_last_layer_activation
        self.model_name = "LSTM"

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
        list_history_model = []
        probabilities_list = []
        real_labels_list = []

        print("STARTING TRAINING MODEL: {}".format(self.model_name))
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
