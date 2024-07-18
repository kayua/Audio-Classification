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
    from tensorflow.keras.layers import Add, Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from sklearn.model_selection import StratifiedKFold

    from MetricsCalculator import MetricsCalculator

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_PROJECTION_DIMENSION = 64
DEFAULT_HEAD_SIZE = 256
DEFAULT_NUMBER_HEADS = 4
DEFAULT_MLP_OUTPUT = 128
DEFAULT_NUMBER_BLOCKS = 4
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_NUMBER_FILTERS = 128
DEFAULT_HOP_LENGTH = 512
DEFAULT_SIZE_FFT = 1024
DEFAULT_SIZE_PATCH = (16, 16)
DEFAULT_OVERLAP = 0.5
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_SIZE_BATCH = 32
DEFAULT_KERNEL_SIZE = 1
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_NORMALIZATION_EPSILON = 1e-6

DEFAULT_INTERMEDIARY_ACTIVATION = 'relu'
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_SOUND_FILE_FORMAT = '*.wav'
DEFAULT_AUDIO_DURATION = 10


class PositionalEmbeddingsLayer(Layer):
    def __init__(self, num_patches, projection_dimension, **kwargs):
        super(PositionalEmbeddingsLayer, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dimension = projection_dimension
        self.embedding_layer = Embedding(input_dim=num_patches + 1, output_dim=projection_dimension)

    def call(self, inputs):
        positions =tensorflow.range(start=0, limit=self.num_patches + 1, delta=1)
        positional_embeddings = self.embedding_layer(positions)
        positional_embeddings = tensorflow.expand_dims(positional_embeddings, axis=0)
        batch_size = tensorflow.shape(inputs)[0]
        positional_embeddings = tensorflow.tile(positional_embeddings, [batch_size, 1, 1])
        return positional_embeddings


class CLSTokenLayer(Layer):
    def __init__(self, projection_dimension, **kwargs):
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.projection_dimension = projection_dimension

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dimension),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, inputs):
        batch_size = tensorflow.shape(inputs)[0]
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])
        return cls_tokens


class AudioClassificationModel(MetricsCalculator):

    def __init__(self, projection_dimension=DEFAULT_PROJECTION_DIMENSION,
                 head_size=DEFAULT_HEAD_SIZE,
                 num_heads=DEFAULT_NUMBER_HEADS,
                 mlp_output=DEFAULT_MLP_OUTPUT,
                 number_blocks=DEFAULT_NUMBER_BLOCKS,
                 number_classes=DEFAULT_NUMBER_CLASSES,
                 sample_rate=DEFAULT_SAMPLE_RATE,
                 number_filters=DEFAULT_NUMBER_FILTERS,
                 hop_length=DEFAULT_HOP_LENGTH,
                 size_fft=DEFAULT_SIZE_FFT,
                 patch_size=DEFAULT_SIZE_PATCH,
                 overlap=DEFAULT_OVERLAP,
                 number_epochs=DEFAULT_NUMBER_EPOCHS,
                 size_batch=DEFAULT_SIZE_BATCH,
                 dropout=DEFAULT_DROPOUT_RATE,
                 intermediary_activation=DEFAULT_INTERMEDIARY_ACTIVATION,
                 loss_function=DEFAULT_LOSS_FUNCTION,
                 last_activation_layer=DEFAULT_LAST_LAYER_ACTIVATION,
                 optimizer_function=DEFAULT_OPTIMIZER_FUNCTION,
                 sound_file_format=DEFAULT_SOUND_FILE_FORMAT,
                 kernel_size=DEFAULT_KERNEL_SIZE,
                 number_splits=DEFAULT_NUMBER_SPLITS,
                 normalization_epsilon=DEFAULT_NORMALIZATION_EPSILON,
                 audio_duration=DEFAULT_AUDIO_DURATION):

        self.model = None
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

    def load_audio(self, filename: str):

        signal, sample_rate = librosa.load(filename, sr=self.sample_rate)
        max_length = int(self.sample_rate * self.audio_duration)

        if len(signal) < max_length:
            padding = max_length - len(signal)
            signal = numpy.pad(signal, (0, padding), 'constant')

        signal = signal[:max_length]
        return signal, sample_rate

    def audio_to_mel_spectrogram(self, signal, sample_rate):

        signal = librosa.feature.melspectrogram(y=signal, n_mels=self.number_filters,
                                                hop_length=self.hop_length, n_fft=self.size_fft)
        spectrogram_signal = librosa.power_to_db(signal, ref=numpy.max)
        return spectrogram_signal

    def split_spectrogram_into_patches(self, spectrogram):
        list_patches = []
        step_size = (int(self.patch_size[0] * (1 - self.overlap)), int(self.patch_size[1] * (1 - self.overlap)))

        for i in range(0, spectrogram.shape[0] - self.patch_size[0] + 1, step_size[0]):
            for j in range(0, spectrogram.shape[1] - self.patch_size[1] + 1, step_size[1]):
                patch = spectrogram[i:i + self.patch_size[0], j:j + self.patch_size[1]]
                list_patches.append(patch)

        return numpy.array(list_patches)

    def linear_projection(self, tensor_patches):
        patches_flat = tensor_patches.reshape(tensor_patches.shape[0], -1)
        return Dense(self.projection_dimension)(patches_flat)

    def transformer_encoder(self, inputs):

        for _ in range(self.number_blocks):
            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(inputs)
            neural_model_flow = MultiHeadAttention(key_dim=self.head_size, num_heads=self.number_heads,
                                                   dropout=self.dropout)(neural_model_flow, neural_model_flow)

            neural_model_flow = Dropout(self.dropout)(neural_model_flow)
            neural_model_flow = Add()([neural_model_flow, inputs])

            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)
            neural_model_flow = Conv1D(filters=self.mlp_output, kernel_size=self.kernel_size,
                                       activation=self.intermediary_activation)(neural_model_flow)

            neural_model_flow = Dropout(self.dropout)(neural_model_flow)

            neural_model_flow = Conv1D(filters=inputs.shape[-1], kernel_size=1)(neural_model_flow)

            inputs = Add()([neural_model_flow, inputs])

        return inputs

    def build_model(self, num_patches):
        print("Dimension: {} {}".format(num_patches, self.projection_dimension))
        inputs = Input(shape=(num_patches, self.projection_dimension))

        cls_tokens_layer = CLSTokenLayer(self.projection_dimension)(inputs)
        neural_model_flow = Concatenate(axis=1)([cls_tokens_layer, inputs])

        positional_embeddings_layer = PositionalEmbeddingsLayer(num_patches, self.projection_dimension)(inputs)
        neural_model_flow += positional_embeddings_layer

        neural_model_flow = self.transformer_encoder(neural_model_flow)

        neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)
        neural_model_flow = GlobalAveragePooling1D()(neural_model_flow)
        neural_model_flow = Dropout(self.dropout)(neural_model_flow)
        outputs = Dense(self.number_classes, activation=self.last_activation_layer)(neural_model_flow)

        self.model = models.Model(inputs, outputs)

        return self.model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=['accuracy'])

    def load_data(self, data_dir):
        file_paths = []
        labels = []

        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                class_label = int(class_dir)
                class_files = glob.glob(os.path.join(class_path, self.sound_file_format))
                file_paths.extend(class_files)
                labels.extend([class_label] * len(class_files))

        return file_paths, labels

    def load_dataset(self, file_paths, labels):
        list_spectrogram = []

        for path_file in file_paths:

            signal, sample_rate = self.load_audio(path_file)

            if signal is None:
                continue
            spectrogram_decibel_scale = self.audio_to_mel_spectrogram(signal, sample_rate)
            spectrogram_patches = self.split_spectrogram_into_patches(spectrogram_decibel_scale)
            list_spectrogram.append(spectrogram_patches)

        list_spectrogram = numpy.array([self.linear_projection(list_patches) for list_patches in list_spectrogram])
        return list_spectrogram, numpy.array(labels)

    def train(self, train_data_dir, number_epochs=None, batch_size=None, number_splits=None):

        number_epochs = number_epochs or self.number_epochs
        number_splits = number_splits or self.number_splits
        batch_size = batch_size or self.size_batch

        train_file_paths, train_labels = self.load_data(train_data_dir)

        sample_audio, _ = self.load_audio(train_file_paths[10])
        sample_spectrogram = self.audio_to_mel_spectrogram(sample_audio, self.sample_rate)
        sample_patches = self.split_spectrogram_into_patches(sample_spectrogram)
        sample_projected_patches = self.linear_projection(sample_patches)

        number_patches = sample_projected_patches.shape[0]
        dataset_features, dataset_labels = self.load_dataset(train_file_paths, train_labels)

        stratified_k_fold = StratifiedKFold(n_splits=number_splits)
        list_history_model = []
        metrics_list = []

        for train_index, val_index in stratified_k_fold.split(dataset_features, dataset_labels):

            self.build_model(number_patches)
            self.compile_model()
            self.model.summary()

            x_train_fold, x_validation_fold = dataset_features[train_index], dataset_features[val_index]
            y_train_fold, y_validation_fold = dataset_labels[train_index], dataset_labels[val_index]

            history = self.model.fit(x_train_fold, y_train_fold, validation_data=(x_validation_fold, y_validation_fold),
                                     epochs=number_epochs, batch_size=batch_size)
            list_history_model.append(history.history)

            y_validation_predicted = self.model.predict(x_validation_fold)
            y_validation_predicted_classes = numpy.argmax(y_validation_predicted, axis=1)
            y_validation_predicted_probability = y_validation_predicted if y_validation_predicted.shape[1] > 1 else None

            metrics, _ = self.calculate_metrics(y_validation_fold, y_validation_predicted_classes,
                                                y_validation_predicted_probability)
            metrics_list.append(metrics)

        mean_metrics = {
            'accuracy': numpy.mean([metric['accuracy'] for metric in metrics_list]),
            'precision': numpy.mean([metric['precision'] for metric in metrics_list]),
            'recall': numpy.mean([metric['recall'] for metric in metrics_list]),
            'f1_score': numpy.mean([metric['f1_score'] for metric in metrics_list]),
            'auc': numpy.mean([metric['auc'] for metric in metrics_list]) if 'auc' in metrics_list[0] else None
        }

        return mean_metrics, list_history_model


audio_classifier = AudioClassificationModel()
audio_classifier.train(train_data_dir='Dataset')
