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
    Gated Linear Unit (GLU) Layer.

    The Gated Linear Unit (GLU) is an activation function introduced by Dauphin et al. (2017)
    in the paper "Language Modeling with Gated Convolutional Networks" (https://arxiv.org/abs/1612.08083).
    It applies a gating mechanism to the input tensor, where the input is split into two halves:
    one half is transformed directly, while the other half is passed through a sigmoid gate.
    The element-wise product of these two halves forms the final output.

    This mechanism helps the network selectively retain or suppress information, improving
    the modeling of complex relationships in the data. GLUs have shown effectiveness in both
    convolutional and sequential models.

    Mathematical Definition
    ----------------------
    Given an input tensor `X` with shape (..., 2d), the GLU activation is defined as:

        GLU(X) = X1 ⊗ σ(X2)

    Where:
        - X1 ∈ ℝ^{..., d} (first half of the last dimension of X)
        - X2 ∈ ℝ^{..., d} (second half of the last dimension of X)
        - σ denotes the sigmoid activation function
        - ⊗ denotes element-wise multiplication

    The input tensor must have an even number of channels in the last dimension.

    Attributes
    ----------
        None (inherits attributes from the base Layer class)

    Methods
    -------
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the GLU activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, sequence_length, 8) — divisible by 2
    ...    input_tensor = tf.random.uniform((2, 5, 8))
    ...    # Instantiate and apply GLU
    ...    glu_layer = GLU()
    ...    output_tensor = glu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 4)
    ...    print(output_tensor.shape)
    >>>
    """

    def __init__(self, **kwargs):
        """
        Initializes the Gated Linear Unit (GLU) layer.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(GLU, self).__init__(**kwargs)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the Gated Linear Unit (GLU) activation function to the input tensor.

        The input tensor is split along the last dimension into two equal parts.
        The first half represents the linear transformation, and the second half
        passes through a sigmoid gate. The final output is the element-wise product
        of these two halves.

        Parameters
        ----------
        neural_network_flow : tensorflow.Tensor
            Input tensor with shape (..., 2d). The last dimension must be even.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with shape (..., d), after applying the GLU transformation.

        Raises
        ------
        ValueError
            If the last dimension of the input tensor is not divisible by 2.

        Example
        -------
        >>> input_tensor = tf.random.uniform((2, 5, 8))
        ...     glu = GLU()
        ...     output = glu(input_tensor)
        ...     print(output.shape)
        >>>     (2, 5, 4)
        """
        a, b = tensorflow.split(neural_network_flow, 2, axis=-1)
        return a * tensorflow.nn.sigmoid(b)
