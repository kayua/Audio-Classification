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

    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import MaxPooling2D
    from sklearn.model_selection import StratifiedKFold

    from Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:

    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default values
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_SIZE_BATCH = 32
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_FILTERS_PER_BLOCK = [16, 32, 64, 96]
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUMBER_LAYERS = 4
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_OVERLAP = 2
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_CONVOLUTIONAL_PADDING = 'same'
DEFAULT_INPUT_DIMENSION = (513, 40, 1)
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SIZE_POOLING = (2, 2)
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 3
DEFAULT_NUMBER_SPLITS = 2
DEFAULT_SIZE_CONVOLUTIONAL_FILTERS = (3, 3)


class ResidualModel(MetricsCalculator):
    """
    A class for creating and training a Convolutional Neural Network (CNN) with residual connections.

    Methods
    -------
    build_model()
        Constructs the CNN model with residual connections.
    windows(data, window_size, overlap)
        Generates windowed segments of the input data.
    load_data(sub_directories: str = None, file_extension: str = None) -> tuple
        Loads audio data, extracts features, and prepares labels.
    compile_model() -> None
        Compiles the CNN model with the specified loss function and optimizer.
    train(train_data_dir: str, number_epochs: int = None, batch_size: int = None,
          number_splits: int = None) -> tuple
        Trains the model using cross-validation and returns the mean metrics and training history.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE,
                 hop_length=DEFAULT_HOP_LENGTH,
                 window_size_factor=DEFAULT_WINDOW_SIZE_FACTOR,
                 number_filters_spectrogram=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                 number_layers=DEFAULT_NUMBER_LAYERS,
                 input_dimension=DEFAULT_INPUT_DIMENSION,
                 overlap=DEFAULT_OVERLAP,
                 convolutional_padding=DEFAULT_CONVOLUTIONAL_PADDING,
                 intermediary_activation=DEFAULT_INTERMEDIARY_ACTIVATION,
                 last_layer_activation=DEFAULT_LAST_LAYER_ACTIVATION,
                 number_classes=DEFAULT_NUMBER_CLASSES,
                 size_convolutional_filters=DEFAULT_SIZE_CONVOLUTIONAL_FILTERS,
                 size_pooling=DEFAULT_SIZE_POOLING,
                 window_size_fft=DEFAULT_WINDOW_SIZE,
                 decibel_scale_factor=DEFAULT_DECIBEL_SCALE_FACTOR,
                 filters_per_block=None,
                 size_batch=DEFAULT_SIZE_BATCH,
                 number_splits=DEFAULT_NUMBER_SPLITS,
                 number_epochs=DEFAULT_NUMBER_EPOCHS,
                 loss_function=DEFAULT_LOSS_FUNCTION,
                 optimizer_function=DEFAULT_OPTIMIZER_FUNCTION,
                 dropout_rate=DEFAULT_DROPOUT_RATE,
                 file_extension=DEFAULT_FILE_EXTENSION):

        """
        Initializes the ResidualModel with the given parameters.

        Parameters
        ----------
        sample_rate: The sample rate of the audio data.
        hop_length: The hop length for the spectrogram.
        window_size_factor: The factor by which the window size is multiplied.
        number_filters_spectrogram: The number of filters in the spectrogram.
        number_layers: The number of layers in the model.
        input_dimension: The shape of the input data.
        overlap: The overlap between consecutive windows.
        convolutional_padding: The padding type for convolutional layers.
        intermediary_activation: The activation function for intermediate layers.
        last_layer_activation: The activation function for the last layer.
        number_classes: The number of output classes.
        size_convolutional_filters: The size of the convolutional filters.
        size_pooling: The size of the pooling layers.
        window_size_fft: The size of the FFT window.
        decibel_scale_factor: The scale factor for converting power spectrogram to decibels.
        filters_per_block: List specifying the number of filters for each convolutional block.
        size_batch: The batch size for training.
        number_splits: The number of splits for cross-validation.
        number_epochs: The number of epochs for training.
        loss_function: The loss function used for training the model.
        optimizer_function: The optimizer function used for training the model.
        dropout_rate: The dropout rate used in the model.
        file_extension: The file extension for audio files.
        """

        if filters_per_block is None:
            filters_per_block = DEFAULT_FILTERS_PER_BLOCK

        self.model_name = "Residual Model"
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
        """
        Constructs a Convolutional Neural Network (CNN) with residual connections.

        This method creates a CNN model architecture by stacking convolutional layers with residual connections,
        followed by pooling and dropout layers.

        Returns
        -------
        keras.Model
            The compiled Convolutional model.
        """
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
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow)

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
            print("Class {}".format(_))
            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))[0:100]):

                signal, _ = librosa.load(file_name, sr=self.sample_rate)
                label = file_name.split('/')[-2].split('_')[0]

                for (start, end) in self.windows(signal, self.window_size, self.overlap):

                    if len(signal[start:end]) == self.window_size:
                        signal = signal[start:end]

                        spectrogram = librosa.feature.melspectrogram(y=signal,
                                                                     n_mels=self.number_filters_spectrogram,
                                                                     sr=self.sample_rate,
                                                                     n_fft=self.window_size_fft,
                                                                     hop_length=self.hop_length)

                        # Convert spectrogram to decibels
                        spectrogram_decibel_scale = librosa.power_to_db(spectrogram, ref=numpy.max)
                        spectrogram_decibel_scale = (spectrogram_decibel_scale / self.decibel_scale_factor) + 1
                        list_spectrogram.append(spectrogram_decibel_scale)
                        list_labels.append(label)

        array_features = numpy.array(list_spectrogram).reshape(len(list_spectrogram),
                                                               self.number_filters_spectrogram,
                                                               self.window_size_factor, 1)

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        # Adjust shape to include an additional dimension
        new_shape = list(array_features.shape)
        new_shape[1] += 1
        new_array = numpy.zeros(new_shape)
        new_array[:, :self.number_filters_spectrogram, :, :] = array_features

        return numpy.array(new_array, dtype=numpy.float32), array_labels

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

    def train(self, train_data_directory: str, number_epochs: int = None, batch_size: int = None,
              number_splits: int = None) -> tuple:

        features, labels = self.load_data(train_data_directory)
        self.number_epochs = number_epochs or self.number_epochs
        self.size_batch = batch_size or self.size_batch
        self.number_splits = number_splits or self.number_splits
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        instance_k_fold = StratifiedKFold(n_splits=self.number_splits)
        list_history_model = None
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
            'Accuracy': {'value': numpy.mean([metric['Accuracy'] for metric in metrics_list]),
                         'std': numpy.std([metric['Accuracy'] for metric in metrics_list])},
            'Precision': {'value': numpy.mean([metric['Precision'] for metric in metrics_list]),
                          'std': numpy.std([metric['Precision'] for metric in metrics_list])},
            'Recall': {'value': numpy.mean([metric['Recall'] for metric in metrics_list]),
                       'std': numpy.std([metric['Recall'] for metric in metrics_list])},
            'F1-Score': {'value': numpy.mean([metric['F1-Score'] for metric in metrics_list]),
                         'std': numpy.std([metric['F1-Score'] for metric in metrics_list])},
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

        return mean_metrics, {"Name": self.model_name, "History": list_history_model}, mean_confusion_matrices
