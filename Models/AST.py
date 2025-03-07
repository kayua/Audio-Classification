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
    import argparse
    import tensorflow

    from tqdm import tqdm

    from sklearn.utils import resample
    from tensorflow.keras import models

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.layers import TimeDistributed
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from Modules.Layers.CLSTokenLayer import CLSTokenLayer

    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Modules.Evaluation.MetricsCalculator import MetricsCalculator

    from Modules.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class AudioAST(MetricsCalculator):

    def __init__(self,
                 projection_dimension: int,
                 head_size: int,
                 num_heads: int,
                 number_blocks: int,
                 number_classes: int,
                 sample_rate: int,
                 hop_length: int,
                 size_fft: int,
                 patch_size: tuple,
                 overlap: int,
                 number_epochs: int,
                 size_batch: int,
                 dropout: float,
                 intermediary_activation: str,
                 loss_function: str,
                 last_activation_layer: str,
                 optimizer_function: str,
                 number_splits: int,
                 normalization_epsilon: float,
                 audio_duration: int,
                 decibel_scale_factor,
                 window_size_fft,
                 window_size_factor,
                 number_filters_spectrogram,
                 file_extension):


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

    def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:

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

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

    def load_data(self, data_dir: str) -> tuple:

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

        logging.info("Starting to load the dataset...")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.sound_file_format

        # Check if the directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Collect all class directories
        logging.info(f"Reading subdirectories in '{sub_directories}'...")
        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            if os.path.isdir(class_path):
                list_class_path.append(class_path)

        logging.info(f"Found {len(list_class_path)} class directories.")

        # Process each audio file in subdirectories
        for sub_directory in list_class_path:
            logging.info(f"Processing class directory: {sub_directory}...")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                try:
                    signal, _ = librosa.load(file_name, sr=self.sample_rate)
                    label = file_name.split('/')[-2].split('_')[0]

                    for (start, end) in self.windows(signal, self.window_size, self.overlap):
                        if len(signal[start:end]) == self.window_size:
                            signal_window = signal[start:end]

                            # Generate mel spectrogram
                            spectrogram = librosa.feature.melspectrogram(
                                y=signal_window,
                                n_mels=self.number_filters_spectrogram,
                                sr=self.sample_rate,
                                n_fft=self.window_size_fft,
                                hop_length=self.hop_length
                            )

                            # Convert spectrogram to decibels
                            spectrogram_decibel_scale = librosa.power_to_db(spectrogram, ref=numpy.max)
                            spectrogram_decibel_scale = (spectrogram_decibel_scale / self.decibel_scale_factor) + 1

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


    def train(self, dataset_directory, number_epochs, batch_size, number_splits, loss, sample_rate, overlap,
              number_classes, arguments) -> tuple:

        # Use default values if not provided
        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or self.overlap
        self.number_classes = number_classes or self.number_classes

        self.head_size = arguments.ast_head_size
        self.number_heads = arguments.ast_number_heads
        self.number_blocks = arguments.ast_number_blocks
        self.hop_length = arguments.ast_hop_length
        self.size_fft = arguments.ast_size_fft
        self.patch_size = arguments.ast_patch_size
        self.overlap = arguments.ast_overlap
        self.dropout = arguments.ast_dropout
        self.normalization_epsilon = arguments.ast_normalization_epsilon
        self.last_activation_layer = arguments.ast_last_activation_layer
        self.projection_dimension = arguments.ast_projection_dimension
        self.intermediary_activation = arguments.ast_intermediary_activation
        self.decibel_scale_factor = arguments.ast_decibel_scale_factor
        self.window_size_fft = arguments.ast_window_size_fft
        self.window_size_factor = arguments.ast_window_size_factor
        self.window_size = arguments.ast_hop_length * (arguments.ast_window_size_factor - 1)
        self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram

        history_model = None
        features, labels = self.load_dataset(dataset_directory)
        number_patches = features.shape[1]
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_val, features_test, labels_train_val, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance training/validation set
        features_train_val, labels_train_val = balance_classes(features_train_val, labels_train_val)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list = []
        real_labels_list = []


        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):

            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the training set for this fold
            features_train, labels_train = balance_classes(features_train, labels_train)

            self.build_model(number_patches)
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_val, labels_val))

            model_predictions = self.neural_network_model.predict(features_val)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_val)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val, predicted_labels)
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

