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

    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Bidirectional
    from sklearn.model_selection import StratifiedKFold
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
DEFAULT_NUMBER_EPOCHS = 2
DEFAULT_NUMBER_SPLITS = 2
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_RECURRENT_ACTIVATION = 'sigmoid'
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'tanh'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


class AudioLSTM(MetricsCalculator):
    """
    A Long Short-Term Memory (LSTM) model for audio classification, integrating LSTM layers and a final classification layer.

    This class inherits from MetricsCalculator to enable calculation of various metrics for model evaluation.
    """

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
        """
        Initializes the AudioLSTM model with the specified parameters.

        :param number_classes: Number of classes for classification.
        :param last_layer_activation: Activation function for the final dense layer.
        :param size_batch: Batch size for training.
        :param number_splits: Number of splits for cross-validation.
        :param number_epochs: Number of training epochs.
        :param loss_function: Loss function for model compilation.
        :param optimizer_function: Optimizer function for model compilation.
        :param window_size_factor: Factor to determine the window size for audio processing.
        :param decibel_scale_factor: Factor to scale decibel values.
        :param hop_length: Length of the hop between successive frames in audio processing.
        :param overlap: Overlap between consecutive windows in audio processing.
        :param sample_rate: Sampling rate of the audio signals.
        :param dropout_rate: Dropout rate for regularization.
        :param file_extension: File extension of the audio files.
        :param intermediary_layer_activation: Activation function for the LSTM layers.
        :param recurrent_activation: Recurrent activation function for the LSTM layers.
        :param input_dimension: Dimension of the input data.
        :param list_lstm_cells: List containing the number of LSTM cells for each layer.
        """

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
        """
        Builds the LSTM model architecture using the initialized parameters.

        The model consists of:
        - LSTM layers with specified number of cells and activation functions
        - Dropout layers for regularization
        - Global average pooling layer
        - Dense layer for final classification

        The constructed model is stored in the `neural_network_model` attribute.
        """
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

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the LSTM model on the provided training data.

        :param train_data: Tensor containing the training data.
        :param train_labels: Tensor containing the training labels.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param validation_data: Tuple containing validation data and labels (optional).
        :return: Training history containing metrics and loss values for each epoch.
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

    @staticmethod
    def windows(data, window_size, overlap):
        """
        Generates windowed segments of the input data.

        Parameters
        ----------
        data : numpy.ndarray
            The input data array.
        window_size : int
            The size of each window.
        overlap : int
            The overlap between consecutive windows.

        Yields
        ------
        tuple
            Start and end indices of each window.
        """
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads audio data, extracts features, and prepares labels.

        This method reads audio files from the specified directories, extracts spectrogram features,
        and prepares the corresponding labels.

        Parameters
        ----------
        sub_directories : list of str, optional
            List of subdirectories containing audio files.
        file_extension : str, optional
            The file extension for audio files.

        Returns
        -------
        tuple
            A tuple containing the feature array and label array.
        """
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            list_class_path.append(class_path)

        for _, sub_directory in enumerate(list_class_path):
            print("Class Load: {}".format(_))
            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                signal, _ = librosa.load(file_name, sr=self.sample_rate)

                label = file_name.split('/')[-2].split('_')[0]

                for (start, end) in self.windows(signal, self.window_size, self.overlap):

                    if len(signal[start:end]) == self.window_size:
                        local_window = len(signal[start:end]) // self.window_size_factor
                        signal = [signal[i:i + local_window] for i in range(0, len(signal[start:end]), local_window)]
                        signal = numpy.abs(numpy.array(signal))

                        signal_min = numpy.min(signal)
                        signal_max = numpy.max(signal)

                        if signal_max != signal_min:
                            normalized_signal = (signal - signal_min) / (
                                    signal_max - signal_min)
                        else:
                            normalized_signal = numpy.zeros_like(signal)

                        list_spectrogram.append(normalized_signal)
                        list_labels.append(label)

        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)

        return array_features, numpy.array(list_labels, dtype=numpy.int32)

    def train(self, dataset_directory, number_epochs, batch_size, number_splits,
              loss, sample_rate, overlap, number_classes) -> tuple:
        """
        Trains the model using cross-validation.

        Parameters
        ----------
        train_data_directory
            Directory containing the training data.
        number_epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size for training.
        number_splits : int, optional
            Number of splits for cross-validation.

        Returns
        -------
        tuple
            A tuple containing the mean metrics and the training history.
        """
        # Use default values if not provided
        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or overlap
        self.number_classes = number_classes or self.number_classes

        features, labels = self.load_data(dataset_directory)
        self.number_epochs = number_epochs or self.number_epochs
        self.size_batch = batch_size or self.size_batch
        self.number_splits = number_splits or self.number_splits
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        instance_k_fold = StratifiedKFold(n_splits=self.number_splits)
        list_history_model = None
        probabilities = None
        real_labels = None
        print("STARTING TRAINING MODEL: {}".format(self.model_name))
        for train_indexes, test_indexes in instance_k_fold.split(features, labels):
            features_train, features_test = features[train_indexes], features[test_indexes]
            labels_train, labels_test = labels[train_indexes], labels[test_indexes]

            self.build_model()
            self.neural_network_model.summary()
            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_test, labels_test))

            model_predictions = self.neural_network_model.predict(features_test)
            predicted_labels = numpy.argmax(model_predictions, axis=1)
            probabilities = model_predictions
            real_labels = labels[test_indexes]
            y_validation_predicted_probability = numpy.array([numpy.argmax(model_predictions[i], axis=-1)
                                                              for i in range(len(model_predictions))])

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_test,
                                                               y_validation_predicted_probability)
            list_history_model = history_model.history
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
            'predicted': probabilities,
            'ground_truth': real_labels
        }

        confusion_matrix_array = numpy.array(confusion_matriz_list)
        confusion_matrix_array = numpy.mean(confusion_matrix_array, axis=0)
        mean_confusion_matrix = numpy.round(confusion_matrix_array).astype(numpy.int32)

        # Calculate the average across the first dimension (number of matrices)
        mean_confusion_matrix = mean_confusion_matrix.tolist()

        mean_confusion_matrices = {
            "confusion_matrix": mean_confusion_matrix,
            "class_names": ['Class {}'.format(i) for i in range(self.number_classes)],
            "title": self.model_name
        }

        return (mean_metrics, {"Name": self.model_name, "History": list_history_model}, mean_confusion_matrices,
                probabilities_predicted)


if __name__ == "__main__":
    lstm_model = AudioLSTM()
    lstm_model.train('Dataset')
