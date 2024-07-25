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
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Lambda, Activation
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import TimeDistributed
    from Layers.QuantizerLayerMLP import QuantizationLayer
    from Loss.ContrastiveLoss import ContrastiveLoss
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Evaluation.MetricsCalculator import MetricsCalculator

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
DEFAULT_NUMBER_EPOCHS = 40
DEFAULT_NUMBER_SPLITS = 2
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
        dense_layer = TimeDistributed(Dense(4,
                                            activation=self.intermediary_layer_activation))(flatten_flow)
        def create_mask(seq_len):
            mask = tensorflow.linalg.band_part(tensorflow.ones((seq_len, seq_len)), -1, 0)
            return mask

        causal_mask = create_mask(128)
        # Transformer Block
        transformer_attention = MultiHeadAttention(
            num_heads=2, key_dim=4)(dense_layer, dense_layer, attention_mask=causal_mask)

        # Add & Normalize (LayerNormalization)
        transformer_attention = Add()([dense_layer, transformer_attention])
        transformer_attention = LayerNormalization()(transformer_attention)

        # Feed Forward Network
        ff_network = Dense(4, activation="relu")(transformer_attention)
        ff_network = Dense(4, activation="relu")(ff_network)

        # Add & Normalize (LayerNormalization)
        transformer_output = Add()([transformer_attention, ff_network])
        transformer_output = LayerNormalization()(transformer_output)

        # Quantize Layer
        quantize_layer = TimeDistributed(QuantizationLayer(4), name="Quantization")(dense_layer)

        self.neural_network_model = Model(inputs=inputs, outputs=[transformer_output, quantize_layer])
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=ContrastiveLoss(margin=0.5),
                                          metrics=['accuracy'])

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=ContrastiveLoss(margin=0.75))

        self.neural_network_model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        self.neural_network_model.trainable = False
        neural_network_flow = Flatten()(self.neural_network_model.output[0])
        neural_network_flow = Dense(self.number_classes,
                                    activation=self.last_layer_activation)(neural_network_flow)

        self.neural_network_model = Model(inputs=self.neural_network_model.inputs, outputs=neural_network_flow)

        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

    @staticmethod
    def windows(data, window_size, overlap):

        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // overlap)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:

        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            list_class_path.append(class_path)

        for _, sub_directory in enumerate(list_class_path):

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                signal, _ = librosa.load(file_name, sr=self.sample_rate)

                label = int(file_name.split('/')[-2].split('_')[0])

                for (start, end) in self.windows(signal, self.window_size, self.overlap):

                    if len(signal[start:end]) == self.window_size:

                        signal = numpy.abs(numpy.array(signal[start:end]))

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

    def train(self, train_data_directory: str, number_epochs: int = None, batch_size: int = None,
              number_splits: int = None) -> tuple:

        features, labels = self.load_data(train_data_directory)
        self.number_epochs = number_epochs or self.number_epochs
        self.size_batch = batch_size or self.size_batch
        self.number_splits = number_splits or self.number_splits
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        instance_k_fold = StratifiedKFold(n_splits=self.number_splits)
        print("STARTING TRAINING MODEL: {}".format(self.model_name))
        list_history_model = None
        for train_indexes, test_indexes in instance_k_fold.split(features, labels):
            features_train, features_test = features[train_indexes], features[test_indexes]
            labels_train, labels_test = labels[train_indexes], labels[test_indexes]

            self.build_model()
            self.neural_network_model.summary()
            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_test, labels_test))

            model_predictions = self.neural_network_model.predict(features_test, batch_size=self.size_batch)
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
