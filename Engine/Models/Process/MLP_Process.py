#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

import logging

from Engine.Models.Process.Base_Process import BaseProcess
from Engine.Processing.ClassBalance import ClassBalancer
from Engine.Processing.WindowGenerator import WindowGenerator

try:

    import os
    import sys
    import glob
    import numpy
    import librosa
    import tensorflow

    from tqdm import tqdm

    from tensorflow.keras import Model
    from sklearn.utils import resample

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Bidirectional
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import GlobalAveragePooling1D


except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

class MLPProcess(ClassBalancer, WindowGenerator, BaseProcess):
    """
    A Long Short-Term Memory (LSTM) model for audio classification, integrating LSTM layers and a final classification layer.

    This class inherits from MetricsCalculator to enable calculation of various metrics for model evaluation.
    """

    def __init__(self, arguments):

        self.neural_network_model = None
        self.batch_size = arguments.batch_size
        self.number_splits = arguments.number_splits
        self.number_epochs = arguments.number_epochs
        self.loss_function = arguments.mlp_loss_function
        self.optimizer_function = arguments.mlp_optimizer_function
        self.window_size_factor = arguments.mlp_window_size_factor
        self.decibel_scale_factor = arguments.mlp_decibel_scale_factor
        self.hop_length = arguments.mlp_hop_length
        self.overlap = arguments.mlp_overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension
        self.input_dimension = arguments.mlp_input_dimension
        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory
        WindowGenerator.__init__(self, self.window_size, self.overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:

        logging.info("Starting data loading process.")

        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        # Process each subdirectory
        for idx, sub_directory in enumerate(list_class_path):
            logging.info(f"Loading class {idx + 1}/{len(list_class_path)} from directory: {sub_directory}")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                # Load the audio signal
                signal, _ = librosa.load(file_name, sr=self.sample_rate)

                # Extract label from the file path (assumes label is part of directory structure)
                label = file_name.split('/')[-2].split('_')[0]

                # Segment the audio into windows
                for (start, end) in self.generate_windows(signal):
                    if len(signal[start:end]) == self.window_size:
                        local_window = len(signal[start:end]) // self.window_size_factor

                        # Divide the window into smaller segments
                        signal_segments = [signal[i:i + local_window] for i in range(0, len(signal[start:end]), local_window)]
                        list_spectrogram.append(self.normalization_signal(signal_segments))
                        list_labels.append(label)

        # Convert lists to numpy arrays
        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)

        # Adding channel dimension for model compatibility
        array_features = numpy.expand_dims(array_features, axis=-1)

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Data loading complete.")
        return array_features, array_labels

    def train(self) -> tuple:
        history_model = None
        features, labels = self.load_data(self.dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_val, features_test, labels_train_val, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance training/validation set
        features_train_val, labels_train_val = self.balance_class(features_train_val, labels_train_val)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list = []
        real_labels_list = []

        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):
            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the training set for this fold
            features_train, labels_train = self.balance_class(features_train, labels_train)

            self.build_model()
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.batch_size,
                                                   validation_data=(features_val, labels_val))

            model_predictions = self.neural_network_model.predict(features_val)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_val)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val, predicted_labels)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)


        return self.__cast_to_dic__(metrics_list, probabilities_list,
                                    real_labels_list, confusion_matriz_list, history_model)