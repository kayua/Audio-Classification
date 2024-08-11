#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

try:
    import sys
    import os
    from tqdm import tqdm
    import glob
    import librosa
    import numpy
    import tensorflow

    from tensorflow.keras import models
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Add, Flatten
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Concatenate, TimeDistributed
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Modules.Layers.CLSTokenLayer import CLSTokenLayer
    from Modules.Evaluation.MetricsCalculator import MetricsCalculator
    from sklearn.model_selection import StratifiedKFold
    from Modules.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default constants for the Audio Classification Model
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_PROJECTION_DIMENSION = 64  # Dimension of the linear projection
DEFAULT_HEAD_SIZE = 256  # Size of each attention head
DEFAULT_NUMBER_HEADS = 2  # Number of attention heads
DEFAULT_NUMBER_BLOCKS = 2  # Number of transformer encoder blocks
DEFAULT_NUMBER_CLASSES = 4  # Number of output classes for classification
DEFAULT_SAMPLE_RATE = 8000  # Sample rate for loading audio
DEFAULT_NUMBER_FILTERS = 128  # Number of filters for the Mel spectrogram
DEFAULT_HOP_LENGTH = 512  # Hop length for the Mel spectrogram
DEFAULT_SIZE_FFT = 1024  # FFT size for the Mel spectrogram
DEFAULT_SIZE_PATCH = (16, 16)  # Size of the patches to be extracted from the spectrogram
DEFAULT_OVERLAP = 2  # Overlap ratio between patches
DEFAULT_DROPOUT_RATE = 0.2  # Dropout rate
DEFAULT_NUMBER_EPOCHS = 2  # Number of training epochs
DEFAULT_SIZE_BATCH = 32  # Batch size for training
DEFAULT_NUMBER_SPLITS = 2  # Number of splits for cross-validation
DEFAULT_NORMALIZATION_EPSILON = 1e-6  # Epsilon value for layer normalization
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'  # Activation function for intermediary layers
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'  # Activation function for the output layer
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'  # Loss function for model compilation
DEFAULT_OPTIMIZER_FUNCTION = 'adam'  # Optimizer function for model compilation
DEFAULT_FILE_EXTENSION = "*.wav"  # File format for sound files
DEFAULT_AUDIO_DURATION = 10  # Duration of audio to be considered
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512


class AudioAST(MetricsCalculator):
    """
    A class used to build and train an audio classification model.

    Attributes
    ----------
    Various attributes with default values for model parameters.
    """

    def __init__(self, projection_dimension: int = DEFAULT_PROJECTION_DIMENSION,
                 head_size: int = DEFAULT_HEAD_SIZE,
                 num_heads: int = DEFAULT_NUMBER_HEADS,
                 number_blocks: int = DEFAULT_NUMBER_BLOCKS,
                 number_classes: int = DEFAULT_NUMBER_CLASSES,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 size_fft: int = DEFAULT_SIZE_FFT,
                 patch_size: tuple = DEFAULT_SIZE_PATCH,
                 overlap: int = DEFAULT_OVERLAP,
                 number_epochs: int = DEFAULT_NUMBER_EPOCHS,
                 size_batch: int = DEFAULT_SIZE_BATCH,
                 dropout: float = DEFAULT_DROPOUT_RATE,
                 intermediary_activation: str = DEFAULT_INTERMEDIARY_ACTIVATION,
                 loss_function: str = DEFAULT_LOSS_FUNCTION,
                 last_activation_layer: str = DEFAULT_LAST_LAYER_ACTIVATION,
                 optimizer_function: str = DEFAULT_OPTIMIZER_FUNCTION,
                 number_splits: int = DEFAULT_NUMBER_SPLITS,
                 normalization_epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
                 audio_duration: int = DEFAULT_AUDIO_DURATION,
                 decibel_scale_factor=DEFAULT_DECIBEL_SCALE_FACTOR,
                 window_size_fft=DEFAULT_SIZE_FFT,
                 window_size_factor=DEFAULT_WINDOW_SIZE_FACTOR,
                 number_filters_spectrogram=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                 file_extension=DEFAULT_FILE_EXTENSION):

        """
        Parameters

        ----------
        projection_dimension: Dimension of the projection in the linear layer.
        head_size: Size of each attention head.
        num_heads: Number of attention heads.
        mlp_output: Output size of the MLP layer.
        number_blocks: Number of transformer encoder blocks.
        number_classes: Number of output classes for classification.
        sample_rate: Sample rate for loading audio.
        number_filters: Number of filters for the Mel spectrogram.
        hop_length: Hop length for the Mel spectrogram.
        size_fft: FFT size for the Mel spectrogram.
        patch_size: Size of the patches to be extracted from the spectrogram.
        overlap: Overlap ratio between patches.
        number_epochs: Number of training epochs.
        size_batch: Batch size for training.
        dropout: Dropout rate.
        intermediary_activation: Activation function for intermediary layers.
        loss_function: Loss function for model compilation.
        last_activation_layer: Activation function for the output layer.
        optimizer_function: Optimizer function for model compilation.
        sound_file_format: File format for sound files.
        kernel_size: Kernel size for convolutional layers.
        number_splits: Number of splits for cross-validation.
        normalization_epsilon: Epsilon value for layer normalization.
        audio_duration: Duration of audio to be considered.

        """
        self.neural_network_model = None
        self.head_size = head_size
        self.number_heads = num_heads
        self.number_blocks = number_blocks
        self.number_classes = number_classes
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.size_fft = size_fft
        self.patch_size = patch_size
        self.overlap = overlap
        self.number_epochs = number_epochs
        self.number_splits = number_splits
        self.size_batch = size_batch
        self.dropout = dropout
        self.optimizer_function = optimizer_function
        self.loss_function = loss_function
        self.normalization_epsilon = normalization_epsilon
        self.last_activation_layer = last_activation_layer
        self.projection_dimension = projection_dimension
        self.intermediary_activation = intermediary_activation
        self.audio_duration = audio_duration
        self.model_name = "AST"
        self.sound_file_format = file_extension
        self.decibel_scale_factor = decibel_scale_factor
        self.window_size_fft = window_size_fft
        self.window_size_factor = window_size_factor
        self.window_size = hop_length * (self.window_size_factor - 1)
        self.number_filters_spectrogram = number_filters_spectrogram

    def load_audio(self, filename: str) -> tuple:
        """
        Loads an audio file and pads or truncates it to the required duration.

        Parameters
        ----------
        filename : str
            Path to the audio file.

        Returns
        -------
        tuple
            A tuple containing the signal and the sample rate. The signal is a numpy array representing the audio waveform,
            and the sample rate is an integer representing the number of samples per second.
        """
        # Load the audio file with the specified sample rate
        signal, sample_rate = librosa.load(filename, sr=self.sample_rate)

        # Calculate the maximum length of the signal based on the desired duration
        max_length = int(self.sample_rate * self.audio_duration)

        # Pad the signal if it's shorter than the required length
        if len(signal) < max_length:
            padding = max_length - len(signal)
            signal = numpy.pad(signal, (0, padding), 'constant')

        # Truncate the signal to the maximum length
        signal = signal[:max_length]

        # Return the processed signal and the sample rate
        return signal, sample_rate

    def split_spectrogram_into_patches(self, spectrogram: numpy.ndarray) -> numpy.ndarray:
        """
        Splits a spectrogram into non-overlapping patches of a fixed size with padding.

        Parameters
        ----------
        spectrogram : numpy.ndarray
            The spectrogram to be split. This is a 2D numpy array representing the Mel spectrogram.

        Returns
        -------
        numpy.ndarray
            An array of patches. Each patch is a 2D numpy array extracted from the spectrogram.
        """

        # Calculate the padding needed to make the dimensions divisible by patch_size
        pad_height = (self.patch_size[0] - (spectrogram.shape[0] % self.patch_size[0])) % self.patch_size[0]
        pad_width = (self.patch_size[1] - (spectrogram.shape[1] % self.patch_size[1])) % self.patch_size[1]

        # Pad the spectrogram with zeros
        padded_spectrogram = numpy.pad(spectrogram, ((0, pad_height), (0, pad_width)), mode='constant',
                                       constant_values=0)

        num_patches_x = padded_spectrogram.shape[0] // self.patch_size[0]
        num_patches_y = padded_spectrogram.shape[1] // self.patch_size[1]

        list_patches = []

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                patch = padded_spectrogram[
                        i * self.patch_size[0]:(i + 1) * self.patch_size[0],
                        j * self.patch_size[1]:(j + 1) * self.patch_size[1]]

                list_patches.append(patch)

        return numpy.array(list_patches)

    def linear_projection(self, tensor_patches: numpy.ndarray) -> numpy.ndarray:
        """
        Applies a linear projection to the patches.

        Parameters
        ----------
        tensor_patches : numpy.ndarray
            The tensor of patches.

        Returns
        -------
        numpy.ndarray
            The projected patches.
        """
        patches_flat = tensor_patches.reshape(tensor_patches.shape[0], -1)
        return Dense(self.projection_dimension)(patches_flat)

    def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Builds the transformer encoder.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            The input tensor.

        Returns
        -------
        tensorflow.Tensor
            The output tensor of the transformer encoder.
        """

        # Iterate over the number of transformer blocks
        for _ in range(self.number_blocks):
            # Apply layer normalization to the input tensor
            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(inputs)
            # Apply multi-head self-attention
            neural_model_flow = MultiHeadAttention(key_dim=self.head_size, num_heads=self.number_heads,
                                                   dropout=self.dropout)(neural_model_flow, neural_model_flow)

            # Apply dropout for regularization
            neural_model_flow = Dropout(self.dropout)(neural_model_flow)
            # Add the input tensor to the output of the self-attention layer (residual connection)
            neural_model_flow = Add()([neural_model_flow, inputs])

            # Apply layer normalization after the residual connection
            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)

            # Apply a feedforward layer (MLP layer) to transform the features
            neural_model_flow = Dense(neural_model_flow.shape[2],
                                      activation=self.intermediary_activation)(neural_model_flow)

            # Apply dropout for regularization
            neural_model_flow = Dropout(self.dropout)(neural_model_flow)
            # Apply a convolutional layer with kernel size of 1 for dimensionality reduction
            # neural_model_flow = Conv1D(filters=inputs.shape[-1], kernel_size=1)(neural_model_flow)

            # Add the input tensor to the output of the MLP layer (residual connection)
            inputs = Add()([neural_model_flow, inputs])

        return inputs

    def build_model(self, number_patches: int) -> tensorflow.keras.models.Model:
        """
        Builds the audio classification model.

        Parameters
        ----------
        number_patches : int
            The number of patches in the input.

        Returns
        -------
        tensorflow.keras.models.Model
            The built Keras model.
        """
        # Define the input layer with shape (number_patches, projection_dimension)
        inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
        input_flatten = TimeDistributed(Flatten())(inputs)
        linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)

        cls_tokens_layer = CLSTokenLayer(self.projection_dimension)(linear_projection)
        # Concatenate the CLS token to the input patches
        neural_model_flow = Concatenate(axis=1)([cls_tokens_layer, linear_projection])

        # Add positional embeddings to the input patches
        positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches,
                                                                self.projection_dimension)(linear_projection)
        neural_model_flow += positional_embeddings_layer

        # Pass the input through the transformer encoder
        neural_model_flow = self.transformer_encoder(neural_model_flow)

        # Apply layer normalization
        neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)
        # Apply global average pooling
        neural_model_flow = GlobalAveragePooling1D()(neural_model_flow)
        # Apply dropout for regularization
        neural_model_flow = Dropout(self.dropout)(neural_model_flow)
        # Define the output layer with the specified number of classes and activation function
        outputs = Dense(self.number_classes, activation=self.last_activation_layer)(neural_model_flow)

        # Create the Keras model
        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)

        return self.neural_network_model

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the neural network model.

        Parameters
        ----------
        train_data : tf.Tensor
            Training data tensor with shape (samples, ...), where ... represents the feature dimensions.
        train_labels : tf.Tensor
            Training labels tensor with shape (samples,), representing the class labels.
        epochs : int
            Number of epochs to train the model.
        batch_size : int
            Number of samples per batch.
        validation_data : tuple, optional
            A tuple (validation_data, validation_labels) for validation during training. If not provided,
             no validation is performed.

        Returns
        -------
        tf.keras.callbacks.History
            History object containing the training history, including loss and metrics over epochs.
        """
        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model with the training data and labels, and optionally validation data
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

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

    def load_dataset(self, sub_directories: str = None, file_extension: str = None) -> tuple:
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
        file_extension = file_extension or self.sound_file_format

        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            list_class_path.append(class_path)

        for _, sub_directory in enumerate(list_class_path):

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

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
                        spectrogram_decibel_scale = self.split_spectrogram_into_patches(spectrogram_decibel_scale)

                        list_spectrogram.append(spectrogram_decibel_scale)
                        list_labels.append(label)

        array_features = numpy.array(list_spectrogram)

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        return numpy.array(array_features, dtype=numpy.float32), array_labels

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
        number_epochs = number_epochs or self.number_epochs
        number_splits = number_splits or self.number_splits
        batch_size = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or overlap
        self.number_classes = number_classes or self.number_classes

        # Load the training file paths and labels
        features, labels = self.load_dataset(dataset_directory)

        number_patches = features.shape[1]
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        instance_k_fold = StratifiedKFold(n_splits=number_splits, shuffle=True)
        list_history_model = None
        probabilities = None
        real_labels = None
        print("STARTING TRAINING MODEL: {}".format(self.model_name))
        for train_indexes, test_indexes in instance_k_fold.split(features, labels):
            features_train, features_test = features[train_indexes], features[test_indexes]
            labels_train, labels_test = labels[train_indexes], labels[test_indexes]

            self.build_model(number_patches)
            self.neural_network_model.summary()
            history_model = self.compile_and_train(features_train, labels_train, epochs=number_epochs,
                                                   batch_size=batch_size,
                                                   validation_data=(features_test, labels_test))

            model_predictions = self.neural_network_model.predict(features_test, batch_size=batch_size)
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
