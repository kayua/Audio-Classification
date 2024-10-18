#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

import logging

try:
    import os
    import sys
    import glob
    import numpy
    import librosa
    import argparse
    import tensorflow

    from tqdm import tqdm

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Activation

    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Modules.Layers.QuantizerLayerMLP import QuantizationLayer
    from Modules.Loss.ContrastiveLoss import ContrastiveLoss

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from sklearn.utils import resample

    from Modules.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_INPUT_DIMENSION = (10240,)
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_NUMBER_HEADS = 2
DEFAULT_KEY_DIMENSION = 16
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_SIZE_BATCH = 8
DEFAULT_OVERLAP = 1
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_CONTEXT_DIMENSION = 16

DEFAULT_PROJECTION_MLP_DIMENSION = 128
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_LIST_FILTERS_ENCODER = [8, 16, 32]
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_QUANTIZATION_BITS = 8
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'relu'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


class AudioWav2Vec2(MetricsCalculator):

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
                 quantization_units: int = DEFAULT_QUANTIZATION_BITS,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 key_dimension: int = DEFAULT_KEY_DIMENSION,
                 dropout_rate: float = DEFAULT_DROPOUT_RATE,
                 file_extension: str = DEFAULT_FILE_EXTENSION,
                 intermediary_layer_activation: str = DEFAULT_INTERMEDIARY_LAYER_ACTIVATION,
                 input_dimension: tuple = DEFAULT_INPUT_DIMENSION,
                 number_heads: int = DEFAULT_NUMBER_HEADS,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 projection_mlp_dimension: int = DEFAULT_PROJECTION_MLP_DIMENSION,
                 context_dimension: int = DEFAULT_CONTEXT_DIMENSION,
                 list_filters_encoder=None):

        if list_filters_encoder is None:
            list_filters_encoder = DEFAULT_LIST_FILTERS_ENCODER

        self.model_name = "Wav2Vec2"
        self.neural_network_model = None

        self.size_batch = size_batch
        self.contex_dimension = context_dimension
        self.list_filters_encoder = list_filters_encoder
        self.projection_mlp_dimension = projection_mlp_dimension
        self.number_splits = number_splits
        self.number_epochs = number_epochs
        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.window_size_factor = window_size_factor
        self.decibel_scale_factor = decibel_scale_factor
        self.hop_length = hop_length
        self.kernel_size = kernel_size
        self.quantization_units = quantization_units
        self.key_dimension = key_dimension
        self.intermediary_layer_activation = intermediary_layer_activation
        self.overlap = overlap
        self.number_heads = number_heads
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = sample_rate
        self.file_extension = file_extension
        self.input_dimension = input_dimension
        self.number_classes = number_classes
        self.dropout_rate = dropout_rate
        self.last_layer_activation = last_layer_activation

    def build_model(self) -> None:
        # Define the input layer
        inputs = Input(shape=self.input_dimension)
        neural_network_flow = Reshape((128, 80, 1))(inputs)

        # Apply Convolutional layers
        for number_filters in self.list_filters_encoder:
            neural_network_flow = TimeDistributed(
                Conv1D(number_filters,
                       self.kernel_size, strides=(2,),
                       activation=self.intermediary_layer_activation))(neural_network_flow)

        flatten_flow = TimeDistributed(Flatten())(neural_network_flow)

        # Dense layer
        dense_layer = TimeDistributed(Dense(self.number_classes,
                                            activation=self.intermediary_layer_activation))(flatten_flow)

        def create_mask(seq_len):
            mask = tensorflow.linalg.band_part(tensorflow.ones((seq_len, seq_len)), -1, 0)
            return mask

        causal_mask = create_mask(128)
        # Transformer Block
        transformer_attention = MultiHeadAttention(
            num_heads=self.number_heads, key_dim=4)(dense_layer, dense_layer, attention_mask=causal_mask)

        # Add & Normalize (LayerNormalization)
        transformer_attention = Add()([dense_layer, transformer_attention])
        transformer_attention = LayerNormalization()(transformer_attention)

        # Feed Forward Network
        ff_network = Dense(self.number_classes, activation="relu")(transformer_attention)
        ff_network = Dense(self.number_classes, activation="relu")(ff_network)

        # Add & Normalize (LayerNormalization)
        transformer_output = Add()([transformer_attention, ff_network])
        transformer_output = LayerNormalization()(transformer_output)

        # Quantize Layer
        quantize_layer = TimeDistributed(QuantizationLayer(4), name="Quantization")(dense_layer)

        self.neural_network_model = Model(inputs=inputs, outputs=[transformer_output, quantize_layer],
                                          name=self.model_name)
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=ContrastiveLoss(margin=0.5),
                                          metrics=['accuracy'])

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the neural network model using the specified training data and configuration.

        Args:
            train_data (tensorflow.Tensor): The input training data.
            train_labels (tensorflow.Tensor): The corresponding labels for the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of the batches for each training step.
            validation_data (tuple, optional): A tuple containing validation data and labels.

        Returns:
            tensorflow.keras.callbacks.History: The history object containing training metrics and performance.
        """
        logging.info("Starting the initial compilation and training phase.")

        # Step 1: Compile the model for initial training using ContrastiveLoss
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=ContrastiveLoss(margin=0.75))
        logging.info("Model compiled with ContrastiveLoss.")

        # Step 2: Train the model on the training data
        logging.info(f"Training model for {epochs} epochs with batch size {batch_size}.")
        self.neural_network_model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        logging.info("Initial training completed. Setting the model as non-trainable.")

        # Step 3: Set the model as non-trainable and flatten the output
        self.neural_network_model.trainable = False
        neural_network_flow = Flatten()(self.neural_network_model.output[0])

        # Step 4: Add a Dense layer with the number of classes and specified activation function
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)
        logging.info(f"Added Dense layer with {self.number_classes} classes and '{self.last_layer_activation}' activation.")

        # Step 5: Recreate the model with new output
        self.neural_network_model = Model(inputs=self.neural_network_model.inputs, outputs=neural_network_flow)

        # Step 6: Compile the new model with the specified optimizer, loss function, and accuracy metric
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])
        logging.info("Recompiled the model with the final configuration (optimizer, loss, metrics).")

        # Step 7: Train the model with the actual training data and labels
        logging.info(f"Final training for {epochs} epochs with batch size {batch_size}.")
        training_history = self.neural_network_model.fit(train_data, train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        logging.info("Training completed successfully.")

        return training_history

    @staticmethod
    def windows(data, window_size, overlap):

        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // overlap)

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

        # Traverse through class directories
        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            list_class_path.append(class_path)
            logging.info(f"Added class path: {class_path}")

        # Process each class path
        for _, sub_directory in enumerate(list_class_path):
            logging.info(f"Processing directory: {sub_directory}")

            # Iterate through all audio files in the directory with the specified extension
            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                # Load the audio signal
                signal, _ = librosa.load(file_name, sr=self.sample_rate)

                # Extract label from file name (assumes directory structure encodes label)
                label = int(file_name.split('/')[-2].split('_')[0])

                # Segment the signal using sliding windows
                for (start, end) in self.windows(signal, self.window_size, self.overlap):

                    # Check if the windowed signal has the required length
                    if len(signal[start:end]) == self.window_size:

                        # Extract the signal window
                        signal_window = numpy.abs(numpy.array(signal[start:end]))

                        # Normalize the signal window
                        signal_min = numpy.min(signal_window)
                        signal_max = numpy.max(signal_window)

                        if signal_max != signal_min:
                            normalized_signal = (signal_window - signal_min) / (signal_max - signal_min)
                        else:
                            normalized_signal = numpy.zeros_like(signal_window)

                        list_spectrogram.append(normalized_signal)
                        list_labels.append(label)

        # Convert lists to numpy arrays for efficient processing
        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)  # Add channel dimension

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Data loading completed successfully.")
        logging.info(f"Total samples loaded: {len(array_labels)}")

        return array_features, array_labels

    def train(self, dataset_directory, number_epochs, batch_size, number_splits,
              loss, sample_rate, overlap, number_classes, arguments) -> tuple:
        """
        Trains the model using cross-validation.

        Parameters
        ----------
        dataset_directory : str
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
            A tuple containing the mean metrics, the training history, the mean confusion matrix,
            and the predicted probabilities along with the ground truth labels.
        """
        # Use default values if not provided
        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or self.overlap
        self.number_classes = number_classes or self.number_classes

        self.contex_dimension = arguments.wav_to_vec_context_dimension
        self.list_filters_encoder = arguments.wav_to_vec_list_filters_encoder
        self.projection_mlp_dimension = arguments.wav_to_vec_projection_mlp_dimension
        self.window_size_factor = arguments.wav_to_vec_window_size_factor
        self.decibel_scale_factor = arguments.wav_to_vec_decibel_scale_factor
        self.hop_length = arguments.wav_to_vec_hop_length
        self.kernel_size = arguments.wav_to_vec_kernel_size
        self.quantization_units = arguments.wav_to_vec_quantization_bits
        self.key_dimension = arguments.wav_to_vec_key_dimension
        self.intermediary_layer_activation = arguments.wav_to_vec_intermediary_layer_activation
        self.overlap = arguments.wav_to_vec_overlap
        self.number_heads = arguments.wav_to_vec_number_heads
        self.window_size = self.hop_length * self.window_size_factor
        self.dropout_rate = arguments.wav_to_vec_dropout_rate
        self.last_layer_activation = arguments.wav_to_vec_last_layer_activation

        features, labels = self.load_data(dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_val, features_test, labels_train_val, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Function to balance the classes by resampling
        def balance_classes(features, labels):
            unique_classes = numpy.unique(labels)
            max_samples = max([sum(labels == c) for c in unique_classes])

            balanced_features = []
            balanced_labels = []

            for c in unique_classes:
                features_class = features[labels == c]
                labels_class = labels[labels == c]

                features_class_resampled, labels_class_resampled = resample(
                    features_class, labels_class,
                    replace=True,
                    n_samples=max_samples,
                    random_state=0
                )

                balanced_features.append(features_class_resampled)
                balanced_labels.append(labels_class_resampled)

            balanced_features = numpy.vstack(balanced_features)
            balanced_labels = numpy.hstack(balanced_labels)

            return balanced_features, balanced_labels

        # Balance training/validation set
        features_train_val, labels_train_val = balance_classes(features_train_val, labels_train_val)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        print("STARTING TRAINING MODEL: {}".format(self.model_name))
        list_history_model = []
        history_model = None
        probabilities_list = []
        real_labels_list = []

        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):
            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the training set for this fold
            features_train, labels_train = balance_classes(features_train, labels_train)

            self.build_model()
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_val, labels_val))

            model_predictions = self.neural_network_model.predict(features_val, batch_size=self.size_batch)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_val)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val,
                                                               predicted_labels)
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


def get_wav_to_vec_args(parser):

    parser.add_argument('--wav_to_vec_number_heads', type=int,
                        default=DEFAULT_NUMBER_HEADS, help='Number of heads in multi-head attention')

    parser.add_argument('--wav_to_vec_key_dimension', type=int,
                        default=DEFAULT_KEY_DIMENSION, help='Dimensionality of attention key vectors')

    parser.add_argument('--wav_to_vec_hop_length', type=int,
                        default=DEFAULT_HOP_LENGTH, help='Hop length for STFT')

    parser.add_argument('--wav_to_vec_overlap', type=int,
                        default=DEFAULT_OVERLAP, help='Overlap between patches in the spectrogram')

    parser.add_argument('--wav_to_vec_dropout_rate', type=float,
                        default=DEFAULT_DROPOUT_RATE, help='Dropout rate in the network')

    parser.add_argument('--wav_to_vec_window_size', type=int,
                        default=DEFAULT_WINDOW_SIZE, help='Size of the FFT window')

    parser.add_argument('--wav_to_vec_kernel_size', type=int,
                        default=DEFAULT_KERNEL_SIZE, help='Size of the convolutional kernel')

    parser.add_argument('--wav_to_vec_decibel_scale_factor', type=float,
                        default=DEFAULT_DECIBEL_SCALE_FACTOR, help='Scale factor for converting to decibels')

    parser.add_argument('--wav_to_vec_context_dimension', type=int,
                        default=DEFAULT_CONTEXT_DIMENSION, help='Context dimension for attention mechanisms')

    parser.add_argument('--wav_to_vec_projection_mlp_dimension', type=int,
                        default=DEFAULT_PROJECTION_MLP_DIMENSION, help='Dimension of the MLP projection layer')

    parser.add_argument('--wav_to_vec_window_size_factor', type=int,
                        default=DEFAULT_WINDOW_SIZE_FACTOR, help='Factor applied to FFT window size')

    parser.add_argument('--wav_to_vec_list_filters_encoder',
                        default=DEFAULT_LIST_FILTERS_ENCODER, help='List of filters for each encoder block')

    parser.add_argument('--wav_to_vec_last_layer_activation', type=str,
                        default=DEFAULT_LAST_LAYER_ACTIVATION, help='Activation function for the last layer')

    parser.add_argument('--wav_to_vec_quantization_bits', type=int,
                        default=DEFAULT_QUANTIZATION_BITS, help='Number of quantization bits for the model')

    parser.add_argument('--wav_to_vec_intermediary_layer_activation', type=str,
                        default=DEFAULT_INTERMEDIARY_LAYER_ACTIVATION, help='Activation function for intermediary layers')

    parser.add_argument('--wav_to_vec_loss_function', type=str,
                        default=DEFAULT_LOSS_FUNCTION, help='Loss function to use during training')


    return parser