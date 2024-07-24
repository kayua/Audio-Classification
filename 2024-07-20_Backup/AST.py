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
    import glob
    import librosa
    import numpy
    import tensorflow

    from tensorflow.keras import models
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Layers.CLSTokenLayer import CLSTokenLayer
    from Evaluation.MetricsCalculator import MetricsCalculator
    from sklearn.model_selection import StratifiedKFold
    from Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default constants for the Audio Classification Model

DEFAULT_PROJECTION_DIMENSION = 64  # Dimension of the linear projection
DEFAULT_HEAD_SIZE = 256  # Size of each attention head
DEFAULT_NUMBER_HEADS = 4  # Number of attention heads
DEFAULT_MLP_OUTPUT = 128  # Output size of the MLP layer
DEFAULT_NUMBER_BLOCKS = 4  # Number of transformer encoder blocks
DEFAULT_NUMBER_CLASSES = 4  # Number of output classes for classification
DEFAULT_SAMPLE_RATE = 8000  # Sample rate for loading audio
DEFAULT_NUMBER_FILTERS = 128  # Number of filters for the Mel spectrogram
DEFAULT_HOP_LENGTH = 512  # Hop length for the Mel spectrogram
DEFAULT_SIZE_FFT = 1024  # FFT size for the Mel spectrogram
DEFAULT_SIZE_PATCH = (16, 16)  # Size of the patches to be extracted from the spectrogram
DEFAULT_OVERLAP = 0.5  # Overlap ratio between patches
DEFAULT_DROPOUT_RATE = 0.2  # Dropout rate
DEFAULT_NUMBER_EPOCHS = 10  # Number of training epochs
DEFAULT_SIZE_BATCH = 32  # Batch size for training
DEFAULT_KERNEL_SIZE = 1  # Kernel size for convolutional layers
DEFAULT_NUMBER_SPLITS = 5  # Number of splits for cross-validation
DEFAULT_NORMALIZATION_EPSILON = 1e-6  # Epsilon value for layer normalization
DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'  # Activation function for intermediary layers
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'  # Activation function for the output layer
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'  # Loss function for model compilation
DEFAULT_OPTIMIZER_FUNCTION = 'adam'  # Optimizer function for model compilation
DEFAULT_SOUND_FILE_FORMAT = '*.wav'  # File format for sound files
DEFAULT_AUDIO_DURATION = 10  # Duration of audio to be considered


class AudioSpectrogramTransformer(MetricsCalculator):
    """
    A class used to build and train an audio classification model.

    Attributes
    ----------
    Various attributes with default values for model parameters.
    """

    def __init__(self, projection_dimension: int = DEFAULT_PROJECTION_DIMENSION,
                 head_size: int = DEFAULT_HEAD_SIZE,
                 num_heads: int = DEFAULT_NUMBER_HEADS,
                 mlp_output: int = DEFAULT_MLP_OUTPUT,
                 number_blocks: int = DEFAULT_NUMBER_BLOCKS,
                 number_classes: int = DEFAULT_NUMBER_CLASSES,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 number_filters: int = DEFAULT_NUMBER_FILTERS,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 size_fft: int = DEFAULT_SIZE_FFT,
                 patch_size: tuple = DEFAULT_SIZE_PATCH,
                 overlap: float = DEFAULT_OVERLAP,
                 number_epochs: int = DEFAULT_NUMBER_EPOCHS,
                 size_batch: int = DEFAULT_SIZE_BATCH,
                 dropout: float = DEFAULT_DROPOUT_RATE,
                 intermediary_activation: str = DEFAULT_INTERMEDIARY_ACTIVATION,
                 loss_function: str = DEFAULT_LOSS_FUNCTION,
                 last_activation_layer: str = DEFAULT_LAST_LAYER_ACTIVATION,
                 optimizer_function: str = DEFAULT_OPTIMIZER_FUNCTION,
                 sound_file_format: str = DEFAULT_SOUND_FILE_FORMAT,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 number_splits: int = DEFAULT_NUMBER_SPLITS,
                 normalization_epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
                 audio_duration: int = DEFAULT_AUDIO_DURATION):

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
        self.neural_model = None
        self.head_size = head_size
        self.number_heads = num_heads
        self.mlp_output = mlp_output
        self.number_blocks = number_blocks
        self.number_classes = number_classes
        self.sample_rate = sample_rate
        self.number_filters = number_filters
        self.hop_length = hop_length
        self.size_fft = size_fft
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.number_epochs = number_epochs
        self.number_splits = number_splits
        self.size_batch = size_batch
        self.dropout = dropout
        self.sound_file_format = sound_file_format
        self.optimizer_function = optimizer_function
        self.loss_function = loss_function
        self.normalization_epsilon = normalization_epsilon
        self.last_activation_layer = last_activation_layer
        self.projection_dimension = projection_dimension
        self.intermediary_activation = intermediary_activation
        self.audio_duration = audio_duration

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

    def audio_to_mel_spectrogram(self, signal: numpy.ndarray, sample_rate: int) -> numpy.ndarray:
        """
        Converts an audio signal to a Mel spectrogram.

        Parameters
        ----------
        signal : numpy.ndarray
            The audio signal.
        sample_rate : int
            The sample rate of the audio signal.

        Returns
        -------
        numpy.ndarray
            The Mel spectrogram of the audio signal.
        """
        signal = librosa.feature.melspectrogram(y=signal, n_mels=self.number_filters,
                                                hop_length=self.hop_length, n_fft=self.size_fft)
        spectrogram_signal = librosa.power_to_db(signal, ref=numpy.max)
        return spectrogram_signal

    def split_spectrogram_into_patches(self, spectrogram: numpy.ndarray) -> numpy.ndarray:
        """
        Splits a spectrogram into overlapping patches.

        Parameters
        ----------
        spectrogram : numpy.ndarray
            The spectrogram to be split. This is a 2D numpy array representing the Mel spectrogram.

        Returns
        -------
        numpy.ndarray
            An array of patches. Each patch is a 2D numpy array extracted from the spectrogram.
        """
        list_patches = []

        # Calculate the step size for extracting patches based on the overlap ratio
        step_size = (int(self.patch_size[0] * (1 - self.overlap)), int(self.patch_size[1] * (1 - self.overlap)))

        # Iterate over the spectrogram to extract patches
        for i in range(0, spectrogram.shape[0] - self.patch_size[0] + 1, step_size[0]):

            for j in range(0, spectrogram.shape[1] - self.patch_size[1] + 1, step_size[1]):
                # Extract a patch from the spectrogram
                patch = spectrogram[i:i + self.patch_size[0], j:j + self.patch_size[1]]

                # Append the patch to the list of patches
                list_patches.append(patch)

        # Convert the list of patches to a numpy array
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
        inputs = Input(shape=(number_patches, self.projection_dimension))

        # Add a CLS token layer
        cls_tokens_layer = CLSTokenLayer(self.projection_dimension)(inputs)
        # Concatenate the CLS token to the input patches
        neural_model_flow = Concatenate(axis=1)([cls_tokens_layer, inputs])

        # Add positional embeddings to the input patches
        positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches, self.projection_dimension)(inputs)
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
        self.neural_model = models.Model(inputs, outputs)

        return self.neural_model

    def compile_model(self) -> None:
        """
        Compiles the model with the specified optimizer and loss function.

        Returns
        -------
        None
        """
        self.neural_model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=['accuracy'])

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

    def load_dataset(self, file_paths: list, labels: list) -> tuple:
        """
        Loads the dataset by converting audio files to spectrograms and patches.

        Parameters
        ----------
        file_paths : list
            List of file paths to audio files.
        labels : list
            List of labels corresponding to the audio files.

        Returns
        -------
        tuple
            A tuple containing the dataset features and labels.
        """
        list_spectrogram = []

        # Iterate over each file path
        for path_file in file_paths:
            # Load the audio file and get the signal and sample rate
            signal, sample_rate = self.load_audio(path_file)

            # Skip files that couldn't be loaded
            if signal is None:
                continue

            # Convert the audio signal to a Mel spectrogram
            spectrogram_decibel_scale = self.audio_to_mel_spectrogram(signal, sample_rate)
            # Split the spectrogram into patches
            spectrogram_patches = self.split_spectrogram_into_patches(spectrogram_decibel_scale)
            # Add the patches to the list
            list_spectrogram.append(spectrogram_patches)

        # Apply linear projection to all patches and convert to numpy array
        list_spectrogram = numpy.array([self.linear_projection(list_patches) for list_patches in list_spectrogram])

        # Convert labels to numpy array and return both features and labels
        return list_spectrogram, numpy.array(labels)

    def train(self, train_data_dir: str, number_epochs: int = None, batch_size: int = None,
              number_splits: int = None) -> tuple:
        """
        Trains the model using cross-validation.

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

        # Load the training file paths and labels
        train_file_paths, train_labels = self.load_data(train_data_dir)

        # Load a sample audio file and process it to determine the number of patches
        sample_audio, _ = self.load_audio(train_file_paths[10])
        sample_spectrogram = self.audio_to_mel_spectrogram(sample_audio, self.sample_rate)
        sample_patches = self.split_spectrogram_into_patches(sample_spectrogram)
        sample_projected_patches = self.linear_projection(sample_patches)

        # Determine the number of patches
        number_patches = sample_projected_patches.shape[0]

        # Load the entire dataset
        dataset_features, dataset_labels = self.load_dataset(train_file_paths, train_labels)

        # Initialize stratified k-fold cross-validation
        stratified_k_fold = StratifiedKFold(n_splits=number_splits)
        list_history_model, metrics_list = [], []

        # Perform k-fold cross-validation
        for train_index, val_index in stratified_k_fold.split(dataset_features, dataset_labels):
            # Build and compile the model for each fold
            self.build_model(number_patches)
            self.compile_model()
            self.neural_model.summary()

            # Split the data into training and validation sets
            x_train_fold, x_validation_fold = dataset_features[train_index], dataset_features[val_index]
            y_train_fold, y_validation_fold = dataset_labels[train_index], dataset_labels[val_index]

            # Train the model
            history = self.neural_model.fit(x_train_fold, y_train_fold,
                                            validation_data=(x_validation_fold, y_validation_fold),
                                            epochs=number_epochs, batch_size=batch_size)
            list_history_model.append(history.history)

            # Predict the validation set
            y_validation_predicted = self.neural_model.predict(x_validation_fold)
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


audio_classifier = AudioSpectrogramTransformer()
audio_classifier.train(train_data_dir='Dataset')
