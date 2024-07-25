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
    import tensorflow
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class GLU(Layer):
    """
    A class implementing the Gated Linear Unit (GLU) activation function.

    The GLU activation function applies a gating mechanism to the input tensor,
    which can help in learning complex patterns and improving model performance.

    Methods
    -------
    call(neural_network_flow: tf.Tensor) -> tf.Tensor
        Applies the GLU activation function to the input tensor and returns the output tensor.
    """

    def __init__(self, **kwargs):
        """
        Initializes the GLU layer.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(GLU, self).__init__(**kwargs)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the Gated Linear Unit (GLU) activation function to the input tensor.

        The GLU activation function splits the input tensor into two halves along the last axis.
        It then applies a sigmoid activation to the second half and multiplies it with the first half.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor to which the GLU activation function is applied.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the GLU activation function.
        """
        a, b = tensorflow.split(neural_network_flow, 2, axis=-1)
        return a * tensorflow.nn.sigmoid(b)
