#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

import glob
import logging
import os

import librosa
import numpy
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from Engine.Models.Process.Base_Process import BaseProcess
from Engine.Processing.ClassBalance import ClassBalancer
from Engine.Processing.WindowGenerator import WindowGenerator
from Engine.Transformations.SpectrogramPatcher import SpectrogramPatcher
from Tools.Metrics import Metrics


class ProcessAST(ClassBalancer, SpectrogramPatcher, WindowGenerator, BaseProcess, Metrics):
    """
    A class used to build and train an audio classification model.

    Attributes
    ----------
    Various attributes with default values for model parameters.
    """

    def __init__(self, arguments):


        self.neural_network_model = None

        self.number_classes = arguments.number_classes
        self.sample_rate = arguments.sample_rate
        self.hop_length = arguments.ast_hop_length
        self.size_fft = arguments.ast_size_fft
        self.patch_size = arguments.ast_patch_size
        self.overlap = arguments.overlap
        self.number_epochs = arguments.number_epochs
        self.number_splits = arguments.number_splits
        self.batch_size = arguments.batch_size
        self.dataset_directory = arguments.dataset_directory

        self.audio_duration = 10
        self.sound_file_format = arguments.file_extension
        self.decibel_scale_factor = arguments.ast_decibel_scale_factor
        self.window_size_fft = arguments.ast_window_size_fft
        self.window_size_factor = arguments.ast_window_size_factor
        self.window_size = arguments.ast_hop_length * (self.window_size_factor - 1)
        self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram

        SpectrogramPatcher.__init__(self, self.patch_size)
        WindowGenerator.__init__(self, self.window_size, self.overlap)


    def load_data(self, data_dir: str) -> tuple:
        """
        Loads audio file paths and labels from the given directory.

        Parameters
        ----------
        data_dir : str
            Directory containing the audio files.

        Returns
        -------
        tuple
            A tuple containing the file paths and labels.
        """
        file_paths, labels = [], []

        # Iterate over each class directory in the given data directory
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)

            # Check if the path is a directory
            if os.path.isdir(class_path):

                # Convert the directory name to an integer label
                class_label = int(class_dir)

                # Get all audio files matching the sound file format within the class directory
                class_files = glob.glob(os.path.join(class_path, self.sound_file_format))

                # Extend the file_paths list with the found files
                file_paths.extend(class_files)

                # Extend the labels list with the corresponding class label
                labels.extend([class_label] * len(class_files))

        # Return the list of file paths and corresponding labels
        return file_paths, labels


    def load_dataset(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads audio data, extracts features, and prepares labels.

        This method reads audio files from the specified directories, extracts mel spectrogram features,
        and prepares the corresponding labels. It also supports splitting spectrograms into patches.

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
        logging.info("Starting to load the dataset...")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.sound_file_format

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        logging.info(f"Found {len(list_class_path)} class directories.")

        # Process each audio file in subdirectories
        for sub_directory in list_class_path:
            logging.info(f"Processing class directory: {sub_directory}...")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                try:
                    signal_raw, _ = librosa.load(file_name, sr=self.sample_rate)
                    label = file_name.split('/')[-2].split('_')[0]

                    for (start, end) in self.generate_windows(signal_raw):
                        if len(signal_raw[start:end]) == self.window_size:
                            signal_window = signal_raw[start:end]

                            # Generate mel spectrogram
                            spectrogram = librosa.feature.melspectrogram(
                                y=signal_window,
                                n_mels=self.number_filters_spectrogram,
                                sr=self.sample_rate,
                                n_fft=self.window_size_fft,
                                hop_length=self.hop_length
                            )

                            # Convert spectrogram to decibels
                            spectrogram_decibel_scale = (librosa.power_to_db(spectrogram,
                                                                             ref=numpy.max) / self.decibel_scale_factor) + 1

                            # Split spectrogram into patches
                            spectrogram_decibel_scale = self.split_spectrogram_into_patches(spectrogram_decibel_scale)

                            # Append spectrogram and label
                            list_spectrogram.append(spectrogram_decibel_scale)
                            list_labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        # Convert lists to arrays
        array_features = numpy.array(list_spectrogram)
        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info(f"Loaded {len(array_features)} spectrogram features.")
        logging.info("Dataset loading complete.")

        return numpy.array(array_features, dtype=numpy.float32), array_labels


    def train(self) -> tuple:

        history_model = None
        features, labels = self.load_dataset(self.dataset_directory)
        number_patches, metrics_list, confusion_matriz_list, labels = (features.shape[1], [], [],
                                                                       numpy.array(labels).astype(float))

        # Split data into train/val and test sets
        features_train_validation, features_test, labels_train_validation, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance training/validation set
        features_train_validation, labels_train_validation = self.balance_class(features_train_validation, labels_train_validation)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list, real_labels_list = [], []


        for train_indexes, val_indexes in instance_k_fold.split(features_train_validation, labels_train_validation):

            features_train, features_validation = (features_train_validation[train_indexes],
                                                   features_train_validation[val_indexes])

            labels_train, labels_validation = (labels_train_validation[train_indexes],
                                               labels_train_validation[val_indexes])

            # Balance the training set for this fold
            features_train, labels_train = self.balance_class(features_train, labels_train)

            self.build_model(number_patches)
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train,
                                                   epochs=self.number_epochs,
                                                   batch_size=self.batch_size,
                                                   validation_data=(features_validation, labels_validation))

            model_predictions = self.neural_network_model.predict(features_validation)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_validation)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_validation)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)


        return self.__cast_to_dic__(metrics_list, probabilities_list,
                                    real_labels_list, confusion_matriz_list, history_model)
