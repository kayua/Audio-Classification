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
    from sklearn.utils import resample
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import MaxPooling2D
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:

    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_FILTERS_PER_BLOCK = [16, 32, 64, 96]


class ResidualModel(MetricsCalculator):

    def __init__(self,
                 sample_rate: int,
                 hop_length: int,
                 window_size_factor: int,
                 number_filters_spectrogram: int,
                 number_layers: int,
                 input_dimension: tuple[int, int, int],
                 overlap: int,
                 convolutional_padding: str,
                 intermediary_activation: str,
                 last_layer_activation: str,
                 number_classes: int,
                 size_convolutional_filters: tuple[int, int],
                 size_pooling: tuple[int, int],
                 window_size_fft: int,
                 decibel_scale_factor: int,
                 filters_per_block: list[int],
                 size_batch: int,
                 number_splits: int,
                 number_epochs: int,
                 loss_function: str,
                 optimizer_function: str,
                 dropout_rate: float,
                 file_extension: str):


        if filters_per_block is None:
            filters_per_block = DEFAULT_FILTERS_PER_BLOCK

        self.neural_network_model = None
        self.loss_function = loss_function
        self.size_pooling = size_pooling
        self.filters_per_block = filters_per_block
        self.input_shape = input_dimension
        self.optimizer_function = optimizer_function
        self.dropout_rate = dropout_rate
        self.size_convolutional_filters = size_convolutional_filters
        self.number_classes = number_classes
        self.last_layer_activation = last_layer_activation
        self.convolutional_padding = convolutional_padding
        self.intermediary_activation = intermediary_activation
        self.model_name = "ResidualModel"

    def build_model(self):

        inputs = Input(shape=self.input_shape)
        neural_network_flow = inputs

        for number_filters in self.filters_per_block:

            residual_flow = neural_network_flow


            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)
            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)


            neural_network_flow = Concatenate()([neural_network_flow, residual_flow])

            neural_network_flow = MaxPooling2D(self.size_pooling)(neural_network_flow)

            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        neural_network_flow = Flatten()(neural_network_flow)
        neural_network_flow = Dense(self.number_classes,
                                    activation=self.last_layer_activation)(neural_network_flow)

        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)


    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

