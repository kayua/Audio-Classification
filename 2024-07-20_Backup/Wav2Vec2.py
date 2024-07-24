#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Layers.QuantizedEncoderLayer import QuantizedEncoderLayer

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
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

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

DEFAULT_INPUT_DIMENSION = (10240, 1)
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_NUMBER_HEADS = 4
DEFAULT_KEY_DIMENSION = 128
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_SIZE_BATCH = 8
DEFAULT_OVERLAP = 2
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 40
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_CONTEXT_DIMENSION = 64
DEFAULT_PROJECTION_MLP_DIMENSION = 128
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_LIST_FILTERS_ENCODER = [16, 32, 64]
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_QUANTIZATION_UNITS = 32
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
                 quantization_units: int = DEFAULT_QUANTIZATION_UNITS,
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

        # Apply the custom QuantizedEncoderLayer
        neural_network_flow = QuantizedEncoderLayer(self.list_filters_encoder,
                                                    self.kernel_size,
                                                    self.quantization_units,
                                                    self.dropout_rate,
                                                    self.intermediary_layer_activation)(inputs)

        # Apply a Dense layer for intermediate representation
        neural_network_flow = Dense(self.projection_mlp_dimension,
                                    activation=self.intermediary_layer_activation)(neural_network_flow)

        # Apply mask using Lambda layer (optional, depending on use case)
        mask = Lambda(lambda x: x * tensorflow.random.uniform(tensorflow.shape(x), 0, 1))(neural_network_flow)

        # Apply MultiHeadAttention
        attention = MultiHeadAttention(num_heads=self.number_heads, key_dim=self.key_dimension)(mask, mask)
        neural_network_flow = LayerNormalization()(attention)
        neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        neural_network_flow = Dense(self.contex_dimension,
                                    activation=self.intermediary_layer_activation)(neural_network_flow)

        context = Dense(self.contex_dimension,
                        activation=self.intermediary_layer_activation)(neural_network_flow)

        # Create the model with the specified inputs and outputs
        self.neural_network_model = Model(inputs=inputs, outputs=[neural_network_flow, context])

        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=ContrastiveLoss(margin=1.0),
                                          metrics=['accuracy'])

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=ContrastiveLoss(margin=1.0))

        training_history = self.neural_network_model.fit(train_data, train_data, epochs=epochs,
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
            print("Class Load: {}".format(_))
            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

                signal, _ = librosa.load(file_name, sr=self.sample_rate)

                label = file_name.split('/')[-2].split('_')[0]

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

        array_features = numpy.asarray(list_spectrogram)
        array_features = numpy.expand_dims(array_features, axis=-1)

        return array_features, list_labels

    def train(self, train_data_dir: str, number_epochs: int = None, batch_size: int = None,
              number_splits: int = None) -> tuple:

        features, labels = self.load_data(train_data_dir)
        list_history_model, metrics_list = [], []
        labels = numpy.array(labels).astype(float)

        k_fold = StratifiedKFold(n_splits=self.number_splits)

        for train_indexes, test_indexes in k_fold.split(features, labels):
            features_train, features_test = features[train_indexes], features[test_indexes]
            labels_train, labels_test = labels[train_indexes], labels[test_indexes]

            self.build_model()
            self.neural_network_model.summary()
            self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                   batch_size=self.size_batch, validation_data=(features_test, labels_test))

            model_predictions = self.neural_network_model.predict(features_test)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            y_validation_predicted_probability = model_predictions if model_predictions.shape[1] > 1 else None

            # Calculate and store the metrics for this fold
            metrics, _ = self.calculate_metrics(predicted_labels, labels_test,
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


if __name__ == "__main__":
    lstm_model = AudioWav2Vec2()
    lstm_model.train(train_data_dir='Dataset')
