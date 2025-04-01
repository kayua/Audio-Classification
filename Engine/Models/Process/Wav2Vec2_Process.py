
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

    import logging
    import librosa

    from tqdm import tqdm

    from Tools.Metrics import Metrics
    from tensorflow.keras import Model

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Processing.ClassBalance import ClassBalancer
    from Engine.Models.Process.Base_Process import BaseProcess
    from Engine.Processing.WindowGenerator import WindowGenerator

except ImportError as error:
    print(error)
    sys.exit(-1)



class Wav2Vec2Process(ClassBalancer, WindowGenerator, BaseProcess, Metrics):

    def __init__(self, arguments):

        self.neural_network_model = None
        self.batch_size = arguments.batch_size
        self.number_splits = arguments.number_splits
        self.number_epochs = arguments.number_epochs
        self.loss_function = arguments.wav_to_vec_loss_function
        self.optimizer_function = arguments.wav_to_vec_optimizer_function
        self.window_size_factor = arguments.wav_to_vec_window_size_factor
        self.decibel_scale_factor = arguments.wav_to_vec_decibel_scale_factor
        self.hop_length = arguments.wav_to_vec_hop_length
        self.overlap = arguments.wav_to_vec_overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension
        self.input_dimension = arguments.wav_to_vec_input_dimension
        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory
        WindowGenerator.__init__(self, self.window_size, self.overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads audio data, extracts spectrogram's using sliding windows, normalizes them, and
        associates labels based on the directory structure.

        Args:
            sub_directories (str): Path to the main directory containing class subdirectories.
            file_extension (str): Optional file extension to filter the files.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - array_features (numpy.ndarray): Array of normalized signal features.
                - array_labels (numpy.ndarray): Array of integer labels for each sample.
        """
        logging.info("Starting data loading process.")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        # Process each class path
        for _, sub_directory in enumerate(list_class_path):
            logging.info(f"Processing directory: {sub_directory}")

            # Iterate through all audio files in the directory with the specified extension
            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                # Load the audio signal
                raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)

                # Extract label from file name (assumes directory structure encodes label)
                label = self.__get_label__(file_name)

                # Segment the signal using sliding windows
                for (start, end) in self.generate_windows(raw_signal):

                    # Check if the windowed signal has the required length
                    if len(raw_signal[start:end]) == self.window_size:

                        list_spectrogram.append(self.normalization_signal(raw_signal[start:end]))
                        list_labels.append(label)

        # Convert lists to numpy arrays for efficient processing
        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)  # Add channel dimension

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Data loading completed successfully.")
        logging.info(f"Total samples loaded: {len(array_labels)}")

        return array_features, array_labels

    def train(self) -> tuple:

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
        history_model, probabilities_list, real_labels_list = None, [], []

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
                                                   epochs=self.number_epochs, batch_size=self.batch_size,
                                                   validation_data=(features_validation, labels_validation))

            model_predictions = self.neural_network_model.predict(features_test, batch_size=self.batch_size)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_test)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_test)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)

        return self.__cast_to_dic__(metrics_list, probabilities_list,
                                    real_labels_list, confusion_matriz_list, history_model)