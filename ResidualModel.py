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

    from tqdm import tqdm
    import librosa.display
    from tensorflow.keras import Model
    from keras.utils import to_categorical

    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import MaxPooling2D
    from sklearn.model_selection import StratifiedKFold

    from MetricsCalculator import MetricsCalculator

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
DEFAULT_FILTERS_PER_BLOCK = [16, 32, 64, 128]
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
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_NUMBER_SPLITS = 5
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
        list_spectrogram, list_labels = [], []
        file_extension = file_extension or self.file_extension

        for _, sub_directory in tqdm(enumerate(sub_directories)):

            for file_name in glob.glob(os.path.join(sub_directory, file_extension)):

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

        array_features = numpy.asarray(list_spectrogram).reshape(len(list_spectrogram), self.number_filters_spectrogram,
                                                                 self.window_size_factor, 1)
        array_labels = to_categorical(numpy.array(list_labels, dtype=numpy.float32))

        return numpy.array(array_features, dtype=numpy.float32), array_labels

    def compile_model(self) -> None:
        """
        Compiles the CNN model with the specified loss function and optimizer.

        This method prepares the model for training by setting the optimizer, loss function, and metrics.
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

    def train(self, train_data_dir: str, number_epochs: int = None, batch_size: int = None,
              number_splits: int = None) -> tuple:
        """
        Trains the model using cross-validation.

        This method performs cross-validation to train the model and evaluate its performance.

        Parameters
        ----------
        train_data_dir : str
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
        number_epochs = number_epochs or self.number_epochs
        number_splits = number_splits or self.number_splits
        batch_size = batch_size or self.size_batch

        # Load the entire dataset
        dataset_features, dataset_labels = self.load_data(train_data_dir, self.file_extension)

        # Initialize stratified k-fold cross-validation
        stratified_k_fold = StratifiedKFold(n_splits=number_splits)
        list_history_model, metrics_list = [], []

        # Perform k-fold cross-validation
        for train_index, val_index in stratified_k_fold.split(dataset_features, dataset_labels):
            # Build and compile the model for each fold
            self.build_model()
            self.compile_model()
            self.neural_network_model.summary()

            # Split the data into training and validation sets
            x_train_fold, x_validation_fold = dataset_features[train_index], dataset_features[val_index]
            y_train_fold, y_validation_fold = dataset_labels[train_index], dataset_labels[val_index]

            # Train the model
            history = self.neural_network_model.fit(x_train_fold, y_train_fold,
                                                    validation_data=(x_validation_fold, y_validation_fold),
                                                    epochs=number_epochs, batch_size=batch_size)
            list_history_model.append(history.history)

            # Predict the validation set
            y_validation_predicted = self.neural_network_model.predict(x_validation_fold)
            y_validation_predicted_classes = numpy.argmax(y_validation_predicted, axis=1)
            y_validation_predicted_probability = y_validation_predicted if y_validation_predicted.shape[1] > 1 else None

            # Calculate and store the metrics for this fold
            metrics, _ = self.calculate_metrics(y_validation_fold, y_validation_predicted_classes,
                                                y_validation_predicted_probability)
            metrics_list.append(metrics)

        # Calculate mean metrics across all folds
        mean_metrics = {
            'accuracy': numpy.mean([metric['accuracy'] for metric in metrics_list]),
            'precision': numpy.mean([metric['precision'] for metric in metrics_list]),
            'recall': numpy.mean([metric['recall'] for metric in metrics_list]),
            'f1_score': numpy.mean([metric['f1_score'] for metric in metrics_list]),
            'auc': numpy.mean([metric['auc'] for metric in metrics_list]) if 'auc' in metrics_list[0] else None
        }

        return mean_metrics, list_history_model
