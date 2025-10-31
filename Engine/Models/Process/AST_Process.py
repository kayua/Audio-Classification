#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    import os
    import sys
    import glob
    import numpy

    import logging
    import librosa

    from tqdm import tqdm

    from Tools.Metrics import Metrics

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Processing.ClassBalance import ClassBalancer

    from Engine.Models.Process.Base_Process import BaseProcess
    from Engine.Processing.WindowGenerator import WindowGenerator

    from Engine.Transformations.SpectrogramPatcher import SpectrogramPatcher

except ImportError as error:
    print(error)
    sys.exit(-1)

class ProcessAST(ClassBalancer, SpectrogramPatcher, WindowGenerator, BaseProcess, Metrics):
    """
    A class used to build and train an audio classification model.

    Attributes:
    ----------
        @neural_network_model : model
            The neural network model (initialized as None).
        @number_classes : int
            The number of output classes (e.g., number of audio categories).
        @sample_rate : int
            The sample rate of the audio files.
        @hop_length : int
            The hop length for the spectrogram computation.
        @size_fft : int
            The FFT size for the spectral transformation.
        @patch_size : int
            The patch size for splitting the spectrogram.
        @overlap : int
            The overlap between the audio windows.
        @number_epochs : int
            The number of epochs for model training.
        @number_splits : int
            The number of splits for cross-validation.
        @batch_size : int
            The batch size for training.
        @dataset_directory : str
            Path to the directory where the dataset is stored.
        @audio_duration : int
            Fixed audio duration for analysis (default is 10 seconds).
        @sound_file_format : str
            The audio file format (e.g., '*.wav').
        @window_size_fft : int
            The FFT window size.
        @window_size_factor : float
            The window size factor for FFT scaling.
        @decibel_scale_factor : float
            The decibel scaling factor for spectrogram conversion.
        @window_size : int
            The window size for processing (calculated using hop_length and window_size_factor).
        @number_filters_spectrogram : int
            The number of filters for spectrogram computation.

    Methods:
    -------
        @__init__(self, arguments)
            Initializes class parameters based on the provided arguments.
        @load_data(self, data_dir: str) -> tuple
            Loads audio file paths and labels from a directory.
        @load_dataset(self, sub_directories: str = None, file_extension: str = None) -> tuple
            Loads audio data, extracts features, and prepares labels.
        @train(self) -> tuple
        Trains the neural network model using cross-validation and returns training and validation metrics.

    Example:
    --------
        >>> arguments = Namespace(number_classes=10,
        ...               sample_rate=22050,
        ...               ast_hop_length=512,
        ...               ast_size_fft=2048,
        ...               ast_patch_size=16,
        ...               overlap=256,
        ...               number_epochs=10,
        ...               number_splits=5,
        ...               batch_size=32,
        ...               dataset_directory='path/to/dataset',
        ...               file_extension='*.wav',
        ...               ast_window_size_fft=1024,
        ...               ast_window_size_factor=4.0,
        ...               ast_decibel_scale_factor=80,
        ...               ast_number_filters_spectrogram=64)
        ...               process_ast = ProcessAST(arguments)
        ...
        ...               # Load the data
        ...               file_paths, labels = process_ast.load_data('path/to/data')
        ...
        ...               # Load and process the dataset
        ...               features, labels = process_ast.load_dataset()
        ...               # Train the model
        ...               metrics = process_ast.train()
        >>>
    """

    def __init__(self, arguments):
        """
        Initializes the class parameters using the values provided in the arguments.

        Parameters
        ----------
        arguments : Namespace
            Arguments defining the model and audio processing parameters.
        """

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
        self.window_size_fft = arguments.ast_window_size_fft
        self.window_size_factor = arguments.ast_window_size_factor
        self.decibel_scale_factor = arguments.ast_decibel_scale_factor

        self.window_size = arguments.ast_hop_length * (self.window_size_factor - 1)
        self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram

        SpectrogramPatcher.__init__(self, self.patch_size)
        WindowGenerator.__init__(self, self.window_size, self.overlap)


    def load_data(self, data_dir: str) -> tuple:
        """
        Loads the audio file paths and labels from a directory.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing audio files.

        Returns
        -------
        tuple
            A tuple containing the file paths and their corresponding labels.

        Example Usage:
        --------------
        file_paths, labels = process_ast.load_data('path/to/data')
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
        Loads the audio dataset, extracts spectrogram features, and prepares labels.

        Parameters
        ----------
        sub_directories : str, optional
            Path to the directory containing subdirectories of audio files.
        file_extension : str, optional
            The audio file extension (e.g., '*.wav').

        Returns
        -------
        tuple
            A tuple containing the extracted feature array and corresponding label array.

        Example Usage:
        --------------
        features, labels = process_ast.load_dataset(sub_directories='path/to/subdirs', file_extension='*.wav')
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
                    label = self.__get_label__(file_name)

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
        """
        Trains the neural network model using stratified cross-validation and returns training and validation metrics.

        Returns
        -------
        tuple
            A tuple containing the model's performance metrics, predicted probabilities,
            true labels, and confusion matrices.

        Example Usage:
        --------------
        metrics = process_ast.train()
        """
        history_model = None
        features, labels = self.load_dataset(self.dataset_directory)
        number_patches, metrics_list, confusion_matriz_list, labels = (features.shape[1], [], [],
                                                                       numpy.array(labels).astype(float))

        # Split data into train/val and test sets
        features_train_validation, features_test, labels_train_validation, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance test set
        features_test, labels_test = self.balance_class(features_test, labels_test)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list, real_labels_list = [], []


        for train_indexes, val_indexes in instance_k_fold.split(features_train_validation,
                                                                labels_train_validation):

            features_train, features_validation = (features_train_validation[train_indexes],
                                                   features_train_validation[val_indexes])

            labels_train, labels_validation = (labels_train_validation[train_indexes],
                                               labels_train_validation[val_indexes])

            # Balance the training set for this fold
            features_train, labels_train = self.balance_class(features_train, labels_train)

            self.build_model(number_patches)
            self.neural_network_model.summary()

            history_model = self.compile_and_train(train_data = features_train,
                                                   train_labels = labels_train,
                                                   epochs=self.number_epochs,
                                                   batch_size=self.batch_size,
                                                   validation_data=(features_validation, labels_validation))



            model_predictions = self.neural_network_model.predict(features_test)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_test)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_test)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)


        return self.__cast_to_dic__(metrics_list, probabilities_list,
                                    real_labels_list, confusion_matriz_list, history_model)
