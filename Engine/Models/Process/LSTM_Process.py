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

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class ProcessLSTM(ClassBalancer, WindowGenerator, BaseProcess):
    """
    A Long Short-Term Memory (LSTM) model for audio classification, integrating LSTM layers and a final classification layer.

    This class inherits from MetricsCalculator to enable calculation of various metrics for model evaluation.
    """

    def __init__(self, arguments):

        self.neural_network_model = None
        self.batch_size = arguments.batch_size
        self.number_splits = arguments.number_splits
        self.number_epochs = arguments.number_epochs
        self.loss_function = arguments.lstm_loss_function
        self.optimizer_function = arguments.lstm_optimizer_function
        self.window_size_factor = arguments.lstm_window_size_factor
        self.decibel_scale_factor = arguments.lstm_decibel_scale_factor
        self.hop_length = arguments.lstm_hop_length
        self.overlap = arguments.lstm_overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension
        self.input_dimension = arguments.lstm_input_dimension
        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory
        WindowGenerator.__init__(self, self.window_size, self.overlap)


    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads audio data, extracts features, and prepares labels.

        This method reads audio files from the specified directories, extracts spectrogram features,
        and prepares the corresponding labels.

        Parameters
        ----------
        sub_directories : str
            Path to the directory containing subdirectories of audio files.
        file_extension : str, optional
            The file extension for audio files (e.g., '*.wav').

        Returns
        -------
        tuple
            A tuple containing the feature array and label array.
        """
        logging.info("Starting to load data...")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        logging.info(f"Reading subdirectories in '{sub_directories}'...")

        # Collect class paths
        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            if os.path.isdir(class_path):
                list_class_path.append(class_path)

        logging.info(f"Found {len(list_class_path)} classes.")

        # Process each subdirectory
        for _, sub_directory in enumerate(list_class_path):

            logging.info(f"Processing class directory: {sub_directory}...")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                try:

                    # Load the audio signal
                    signal, _ = librosa.load(file_name, sr=self.sample_rate)
                    # Extract label from the file path (assumes label is part of directory structure)
                    label = file_name.split('/')[-2].split('_')[0]

                    # Segment the audio into windows
                    for (start, end) in self.generate_windows(signal):

                        if len(signal[start:end]) == self.window_size:

                            local_window = len(signal[start:end]) // self.window_size_factor
                            # Divide the window into smaller segments
                            signal_segments = [signal[i:i + local_window] for i in
                                               range(0, len(signal[start:end]), local_window)]
                            signal_segments = numpy.abs(numpy.array(signal_segments))

                            # Normalize each segment
                            signal_min = numpy.min(signal_segments)
                            signal_max = numpy.max(signal_segments)

                            if signal_max != signal_min:
                                normalized_signal = (signal_segments - signal_min) / (signal_max - signal_min)

                            else:
                                normalized_signal = numpy.zeros_like(signal_segments)

                            list_spectrogram.append(normalized_signal)
                            list_labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)

        logging.info(f"Loaded {len(array_features)} feature arrays.")
        logging.info("Data loading complete.")

        return array_features, numpy.array(list_labels, dtype=numpy.int32)


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
        list_history_model = []
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