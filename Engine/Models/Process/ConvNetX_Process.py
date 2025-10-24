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
    import librosa
    import logging

    from tqdm import tqdm
    import librosa.display

    from Tools.Metrics import Metrics

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Processing.ClassBalance import ClassBalancer
    from Engine.Models.Process.Base_Process import BaseProcess
    from Engine.Processing.WindowGenerator import WindowGenerator

except ImportError as error:
    print(error)
    sys.exit(-1)


class ConvNetXProcess(ClassBalancer, WindowGenerator, BaseProcess, Metrics):

    def __init__(self, arguments):

        self.neural_network_model = None
        self.sample_rate = arguments.sample_rate
        self.batch_size = arguments.batch_size
        self.number_splits = arguments.number_splits
        self.loss_function = arguments.convnext_loss_function
        self.hop_length = arguments.convnext_hop_length
        self.decibel_scale_factor = arguments.convnext_decibel_scale_factor
        self.window_size_fft = arguments.convnext_window_size
        self.window_size_factor = arguments.convnext_window_size_factor
        self.window_size = arguments.convnext_hop_length * (self.window_size_factor - 1)
        self.input_shape = arguments.convnext_input_dimension
        self.overlap = arguments.convnext_overlap
        self.number_epochs = arguments.number_epochs
        self.optimizer_function = arguments.convnext_optimizer_function
        self.file_extension = arguments.file_extension
        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory
        self.number_filters_spectrogram = arguments.convnext_number_filters_spectrogram
        WindowGenerator.__init__(self, self.window_size, self.overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:

        logging.info("Starting data loading process.")

        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        # Process each subdirectory
        for _, sub_directory in enumerate(list_class_path):
            logging.info(f"Processing directory: {sub_directory}")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                # Load the audio signal
                raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)
                label = self.__get_label__(file_name)  # Extract label from the file path

                # Segment the audio into windows
                for (start, end) in self.generate_windows(raw_signal):

                    if len(raw_signal[start:end]) == self.window_size:
                        signal_segment = raw_signal[start:end]

                        # Generate a mel spectrogram
                        spectrogram = librosa.feature.melspectrogram(y=signal_segment,
                                                                     n_mels=self.number_filters_spectrogram,
                                                                     sr=self.sample_rate,
                                                                     n_fft=self.window_size_fft,
                                                                     hop_length=self.hop_length)

                        # Convert the spectrogram to decibel scale
                        spectrogram_decibel_scale = (librosa.power_to_db(spectrogram,
                                                                         ref=numpy.max) / self.decibel_scale_factor) + 1
                        list_spectrogram.append(spectrogram_decibel_scale)
                        list_labels.append(label)

        # Convert lists to arrays
        array_features = numpy.array(list_spectrogram).reshape(len(list_spectrogram),
                                                            self.number_filters_spectrogram,
                                                            self.window_size_factor, 1)
        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        # Adjust array shape for additional dimensions
        logging.info("Reshaping feature array.")
        new_shape = list(array_features.shape)
        new_shape[1] += 1  # Adding an additional filter dimension
        new_array = numpy.zeros(new_shape)
        new_array[:, :self.number_filters_spectrogram, :, :] = array_features

        logging.info("Data loading complete.")
        return numpy.array(new_array, dtype=numpy.float32), array_labels


    def train(self) -> tuple:

        history_model = None
        features, labels = self.load_data(self.dataset_directory)
        metrics_list, confusion_matriz_list, labels = [], [], numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_validation, features_test, labels_train_validation, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance evaluation set
        features_test, labels_test = self.balance_class(features_test, labels_test)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list, real_labels_list = [], []

        for train_indexes, validation_indexes in instance_k_fold.split(features_train_validation,
                                                                       labels_train_validation):

            features_train, features_validation = (features_train_validation[train_indexes],
                                                   features_train_validation[validation_indexes])

            labels_train, labels_validation = (labels_train_validation[train_indexes],
                                               labels_train_validation[validation_indexes])

            # Balance the training set for this fold
            features_train, labels_train = self.balance_class(features_train, labels_train)

            self.build_model()
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train,
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