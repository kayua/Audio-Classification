#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import sys

    import tensorflow

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class CLSTokenLayer(Layer):
    """
    Implements a layer to add a [CLS] (classification) token to the input sequences.

    The [CLS] token is a special token often used in transformer-based models like BERT to aggregate
    information from the entire input sequence, typically for classification tasks. This token is
    prepended to the beginning of each input sequence, enabling the model to learn an aggregate
    representation of the sequence for downstream tasks, such as sentence classification.

    The [CLS] token is initialized as a learnable parameter and is concatenated with the input tensor
    during the model's forward pass. The dimensionality of the [CLS] token automatically matches the
    embedding dimension of the input sequences.

    Attributes:
    -----------
    cls_token : tf.Variable
        The [CLS] token, represented as a learnable embedding. It is shared across all input sequences
        and has the same dimensionality as the input embeddings.

    Example
    -------
    >>> # Initialize the CLSTokenLayer
    >>> cls_layer = CLSTokenLayer()
    >>> # Example input tensor of shape (batch_size, sequence_length, embedding_dim)
    >>> input_tensor = tf.random.normal([32, 100, 128])  # Batch of 32, sequence length of 100, 128 features
    >>> # Apply the CLSTokenLayer to the input tensor
    >>> output = cls_layer(input_tensor)
    >>> # Print the shape of the output tensor
    >>> print("Output tensor shape with [CLS] token:", output.shape)

    Output:
    -------
    Output tensor shape with [CLS] token: (32, 101, 128)
    """

    def __init__(self, **kwargs):
        """
        Initializes the CLSTokenLayer.

        Args:
            **kwargs: Additional arguments passed to the `Layer` superclass.
        """
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.cls_token = None

    def build(self, input_shape):
        """
        Creates the [CLS] token as a trainable weight variable.

        This method is called once the layer is added to the model and before any computations are
        performed. It creates the [CLS] token as a learnable parameter with shape (1, 1, embedding_dim),
        where `embedding_dim` matches the last dimension of the input tensor.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor. Used to determine the embedding dimension.
        """
        # Get the embedding dimension from the input shape
        embedding_dim = input_shape[-1]

        # Initialize the [CLS] token as a trainable variable with the same embedding dimension as the input
        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

        super(CLSTokenLayer, self).build(input_shape)

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Adds the [CLS] token to the beginning of the input sequences.

        This method takes the input tensor, extracts the batch size, and tiles the [CLS] token to
        match the batch size. The [CLS] token is then concatenated with the input sequences along
        the sequence dimension (axis=1), resulting in an output tensor where each sequence is
        prepended with the [CLS] token.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, sequence_length, embedding_dim)`.

        Returns:
            tf.Tensor: A tensor of shape `(batch_size, sequence_length + 1, embedding_dim)`, containing
                       the [CLS] token prepended to each input sequence.
        """
        # Get the batch size from the input tensor shape
        batch_size = tensorflow.shape(inputs)[0]

        # Tile the [CLS] token to match the batch size
        # Create a tensor of shape (batch_size, 1, embedding_dim)
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])

        # Concatenate the [CLS] token with the input sequences along axis=1 (sequence dimension)
        # Output shape: (batch_size, sequence_length + 1, embedding_dim)
        return tensorflow.concat([cls_tokens, inputs], axis=1)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(CLSTokenLayer, self).get_config()
        return config