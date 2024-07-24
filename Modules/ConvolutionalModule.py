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
    import tensorflow as tf

    from Layers.GLU import GLU
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import DepthwiseConv1D

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_NUMBER_FILTERS = 128
DEFAULT_SIZE_KERNEL = 3
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_CONVOLUTIONAL_PADDING = "same"


class ConvolutionalModule(Layer):
    """
    A convolutional module that applies a series of transformations to a 1D tensor, including point-wise
    convolutions, depth-wise convolutions, activations, batch normalization, and dropout.
    """

    def __init__(self,
                 number_filters: int = DEFAULT_NUMBER_FILTERS,
                 size_kernel: int = DEFAULT_SIZE_KERNEL,
                 dropout_decay: float = DEFAULT_DROPOUT_RATE,
                 convolutional_padding: str = DEFAULT_CONVOLUTIONAL_PADDING,
                 **kwargs):
        """
        Initializes the ConvolutionalModule with the specified parameters.

        Parameters
        ----------
        number_filters : int
            Number of filters for the convolutional layers.
        size_kernel : int
            Size of the kernel for the depth-wise convolutional layer.
        dropout_decay : float
            Dropout rate for the dropout layer.
        convolutional_padding : str
            Padding type for the convolutional layers.
        **kwargs : Additional keyword arguments.
        """
        super(ConvolutionalModule, self).__init__(**kwargs)
        self.convolutional_padding = convolutional_padding
        self.layer_normalization = LayerNormalization()
        self.first_point_wise_convolutional = Conv1D(number_filters * 2, kernel_size=1)
        self.glu_activation = GLU()
        self.depth_wise_convolutional = DepthwiseConv1D(kernel_size=size_kernel, padding=self.convolutional_padding)
        self.batch_normalization = BatchNormalization()
        self.swish_activation = Activation(tf.nn.swish)
        self.second_point_wise_convolutional = Conv1D(1, kernel_size=1)
        self.dropout = Dropout(dropout_decay)

    def call(self, neural_network_flow: tf.Tensor) -> tf.Tensor:
        """
        Applies the convolutional transformations to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor to be processed by the convolutional module.

        Returns
        -------
        tf.Tensor
            Output tensor after applying the convolutional transformations.
        """
        residual_flow = neural_network_flow
        neural_network_flow = self.layer_normalization(neural_network_flow)

        neural_network_flow = self.first_point_wise_convolutional(neural_network_flow)
        neural_network_flow = self.glu_activation(neural_network_flow)

        neural_network_flow = self.depth_wise_convolutional(neural_network_flow)

        neural_network_flow = self.batch_normalization(neural_network_flow)
        neural_network_flow = self.swish_activation(neural_network_flow)
        neural_network_flow = self.second_point_wise_convolutional(neural_network_flow)
        neural_network_flow = self.dropout(neural_network_flow)

        return neural_network_flow + residual_flow
