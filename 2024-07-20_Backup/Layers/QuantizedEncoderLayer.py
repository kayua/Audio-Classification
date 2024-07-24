#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Layers.QuantizerLayerMLP import QuantizeLayer

try:
    import sys
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Layer, MaxPooling1D

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default values for convolutional layer parameters
DEFAULT_CONVOLUTIONAL_PADDING = 'same'
DEFAULT_LIST_CONVOLUTIONAL_FILTERS = [16, 32, 64]
DEFAULT_CONVOLUTIONAL_KERNEL_SIZE = 3
DEFAULT_QUANTIZATION_UNITS = 64
DEFAULT_DROPOUT_DECAY = 0.1
DEFAULT_SIZE_POOLING = 2
DEFAULT_INTERMEDIARY_LAYER_ACTIVATION = 'relu'
DEFAULT_NUMBER_BITS_QUANTIZATION = 8


class QuantizedEncoderLayer(Layer):
    """
    A custom Keras layer that applies a sequence of 1D convolutional layers followed by max pooling,
    and includes quantization on a dense layer output. This layer is used for encoding features with
    quantization support for potential compression or efficiency improvements.

    Attributes:
        convolutional_filters_list (list): List of integers specifying the number of filters for each Conv1D layer.
        kernel_size (int): Size of the convolutional kernel.
        quantization_units (int): Number of units in the Dense layer before quantization.
        dropout_rate (float): Dropout rate for regularization.
        intermediary_layer_activation (str): Activation function applied to Conv1D and Dense layers.
        convolutional_padding (str): Padding strategy for Conv1D layers ('same' or 'valid').
        size_pooling (int): Pool size for MaxPooling1D layers.
        number_bits_quantization (QuantizeLayer): Optional quantizer layer for future use.
    """

    def __init__(self,
                 convolutional_filters_list=None,
                 kernel_size=DEFAULT_CONVOLUTIONAL_KERNEL_SIZE,
                 quantization_units=DEFAULT_QUANTIZATION_UNITS,
                 dropout_rate=DEFAULT_DROPOUT_DECAY,
                 intermediary_layer_activation=DEFAULT_INTERMEDIARY_LAYER_ACTIVATION,
                 convolutional_padding=DEFAULT_CONVOLUTIONAL_PADDING,
                 size_pooling=DEFAULT_SIZE_POOLING,
                 number_bits_quantization=DEFAULT_NUMBER_BITS_QUANTIZATION,
                 **kwargs):
        """
        Initializes the QuantizedEncoderLayer with specified parameters.

        Args:
            convolutional_filters_list (list): List of integers for Conv1D layer filters.
            kernel_size (int): Size of the convolutional kernel.
            quantization_units (int): Number of units in the Dense layer.
            dropout_rate (float): Dropout rate.
            intermediary_layer_activation (str): Activation function for layers.
            convolutional_padding (str): Padding type for Conv1D layers.
            size_pooling (int): Pool size for MaxPooling1D layers.
            quantizer (QuantizeLayer): Optional quantizer for future use.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(QuantizedEncoderLayer, self).__init__(**kwargs)

        if convolutional_filters_list is None:
            convolutional_filters_list = DEFAULT_LIST_CONVOLUTIONAL_FILTERS
        self.quantize_layer = None
        self.dense_layer = None
        if convolutional_filters_list is None:
            convolutional_filters_list = convolutional_filters_list  # Default values

        self.dense_quantization = None
        self.list_convolutions = []
        self.list_pooling = []

        self.convolutional_filters_list = convolutional_filters_list
        self.kernel_size = kernel_size
        self.size_pooling = size_pooling
        self.intermediary_layer_activation = intermediary_layer_activation
        self.quantization_units = quantization_units
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)
        self.convolutional_padding = convolutional_padding
        self.number_bits_quantization = number_bits_quantization

    def build(self, input_shape):
        """
        Builds the QuantizedEncoderLayer by creating Conv1D and MaxPooling1D layers based on the specified parameters.
        Also initializes the Dense and Quantize layers.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        for filters in self.convolutional_filters_list:
            self.list_convolutions.append(Conv1D(filters=filters,
                                                 kernel_size=self.kernel_size,
                                                 activation=self.intermediary_layer_activation,
                                                 padding=self.convolutional_padding))
            self.list_pooling.append(MaxPooling1D(pool_size=self.size_pooling))

        # Define and apply quantization on Dense layer
        self.dense_layer = Dense(self.quantization_units,
                                 activation=self.intermediary_layer_activation)
        self.quantize_layer = QuantizeLayer(num_bits=self.number_bits_quantization)  # Assuming 8-bit quantization

        super(QuantizedEncoderLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Defines the forward pass of the layer.

        Args:
            inputs (tensor): Input tensor to the layer.
            training (bool): Flag indicating whether the layer is in training mode.

        Returns:
            tensor: Output tensor after applying convolution, pooling, dense layer, quantization, and dropout.
        """
        neural_network_flow = inputs

        for convolutional_layer, pooling_layer in zip(self.list_convolutions, self.list_pooling):
            neural_network_flow = convolutional_layer(neural_network_flow)
            neural_network_flow = pooling_layer(neural_network_flow)

        neural_network_flow = self.dense_layer(neural_network_flow)
        neural_network_flow = self.quantize_layer(neural_network_flow)

        if training:
            random_mask = tf.random.uniform(tf.shape(neural_network_flow), 0, 1)
            neural_network_flow = neural_network_flow * random_mask

        if training:
            neural_network_flow = self.dropout(neural_network_flow)

        return neural_network_flow
