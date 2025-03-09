#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

import logging

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

    from sklearn.utils import resample

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Modules.Layers.ConformerBlock import ConformerBlock
    from Modules.Evaluation.MetricsCalculator import MetricsCalculator
    from Modules.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling


except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)



class Conformer(ClassBalancer, WindowGenerator):
    """
    A Conformer model for audio classification, integrating convolutional subsampling, conformer blocks,
    and a final classification layer.

    """

    def __init__(self, arguments):

        self.neural_network_model = None
        self.size_batch = arguments.size_batch
        self.number_splits = arguments.number_splits
        self.number_epochs = arguments.number_epochs
        self.loss_function = arguments.loss_function
        self.optimizer_function = arguments.optimizer_function
        self.window_size_factor = arguments.window_size_factor
        self.decibel_scale_factor = arguments.decibel_scale_factor
        self.hop_length = arguments.hop_length
        self.window_size_fft = arguments.window_size_fft
        self.overlap = arguments.overlap
        self.window_size = arguments.hop_length * (self.window_size_factor - 1)
        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension
        self.input_dimension = arguments.input_dimension
        self.number_classes = arguments.number_classes
        self.number_filters_spectrogram = arguments.number_filters_spectrogram
        WindowGenerator.__init__(self, self.window_size, self.overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads audio data, extracts spectrogram features, and prepares labels.

        This method reads audio files from the specified directories, extracts mel spectrogram features,
        and prepares the corresponding labels.

        Parameters
        ----------
        sub_directories : str, optional
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

        # Check if directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Collect all class directories
        logging.info(f"Reading subdirectories in '{sub_directories}'...")
        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            if os.path.isdir(class_path):
                list_class_path.append(class_path)

        logging.info(f"Found {len(list_class_path)} class directories.")

        # Process each audio file in subdirectories
        for sub_directory in list_class_path:
            logging.info(f"Processing class directory: {sub_directory}...")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                try:
                    signal, _ = librosa.load(file_name, sr=self.sample_rate)
                    label = file_name.split('/')[-2].split('_')[0]

                    for (start, end) in self.generate_windows(signal):
                        if len(signal[start:end]) == self.window_size:
                            signal_window = signal[start:end]

                            # Generate mel spectrogram
                            spectrogram = librosa.feature.melspectrogram(
                                y=signal_window,
                                n_mels=self.number_filters_spectrogram,
                                sr=self.sample_rate,
                                n_fft=self.window_size_fft,
                                hop_length=self.hop_length
                            )

                            # Convert spectrogram to decibels
                            spectrogram_decibel_scale = librosa.power_to_db(spectrogram, ref=numpy.max)
                            spectrogram_decibel_scale = (spectrogram_decibel_scale / self.decibel_scale_factor) + 1

                            # Append spectrogram and label
                            list_spectrogram.append(spectrogram_decibel_scale)
                            list_labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        # Reshape the feature array to the expected dimensions
        array_features = numpy.array(list_spectrogram).reshape(
            len(list_spectrogram),
            self.number_filters_spectrogram,
            self.window_size // self.hop_length,  # Time frames depend on hop length
            1
        )

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info(f"Loaded {len(array_features)} spectrogram features.")
        logging.info("Data loading complete.")

        return numpy.array(array_features, dtype=numpy.float32), array_labels



    def train(self, dataset_directory) -> tuple:

        history_model = None
        features, labels = self.load_data(dataset_directory)
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


