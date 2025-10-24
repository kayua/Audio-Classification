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
    import tensorflow
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    import sys

    sys.exit(-1)


class DistillationCLSTokenLayer(Layer):
    """
    A custom Keras layer that adds both classification (CLS) and distillation tokens
    to the input sequence for knowledge distillation in transformer models.

    This layer is typically used in vision transformers (ViTs) and similar architectures
    that employ knowledge distillation techniques. It prepends two special tokens
    to the input sequence:
    - CLS token: Standard classification token used for the main task
    - Distillation token: Additional token used specifically for distillation loss

    The tokens are learnable parameters that are optimized during training.

    Attributes:
        cls_token (tf.Variable): Learnable classification token parameter
        dist_token (tf.Variable): Learnable distillation token parameter

    Example:
        >>> layer = DistillationCLSTokenLayer()
        >>> input_tensor = tf.random.normal((2, 10, 768))  # (batch, seq_len, embedding_dim)
        >>> output = layer(input_tensor)
        >>> print(output.shape)  # (2, 12, 768) - sequence length increased by 2
    """

    def __init__(self, **kwargs):
        """
        Initialize the DistillationCLSTokenLayer.

        Args:
            **kwargs: Additional keyword arguments passed to the parent Layer class
        """
        super().__init__(**kwargs)
        self.cls_token = None
        self.dist_token = None

    def build(self, input_shape):
        """
        Create the layer weights based on the input shape.

        This method is called automatically when the layer is first used.
        It creates two learnable token parameters with the same embedding dimension
        as the input sequence.

        Args:
            input_shape (tuple): Shape of the input tensor, expected to be
                                (batch_size, sequence_length, embedding_dim)

        Raises:
            ValueError: If input_shape doesn't have at least 3 dimensions
        """
        # Extract embedding dimension from the last dimension of input shape
        embedding_dim = input_shape[-1]

        # Initialize CLS token as a learnable parameter
        # Shape: (1, 1, embedding_dim) - single token vector
        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',  # Random initialization from normal distribution
            trainable=True,  # Parameter will be updated during training
            name='cls_token'  # Unique name for the weight
        )

        # Initialize distillation token as a learnable parameter
        # Shape: (1, 1, embedding_dim) - single token vector
        self.dist_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',  # Random initialization from normal distribution
            trainable=True,  # Parameter will be updated during training
            name='distillation_token'  # Unique name for the weight
        )

        # Call parent build method to finalize layer building
        super().build(input_shape)

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Forward pass of the layer.

        This method takes the input sequence and prepends both CLS and distillation tokens
        to the beginning of the sequence. The tokens are replicated for each sample in the batch.

        Args:
            inputs (tensorflow.Tensor): Input tensor of shape
                                       (batch_size, sequence_length, embedding_dim)

        Returns:
            tensorflow.Tensor: Output tensor of shape
                             (batch_size, sequence_length + 2, embedding_dim)
                             with CLS and distillation tokens prepended

        Example:
            Input shape: (batch_size, seq_len, embedding_dim)
            Output shape: (batch_size, seq_len + 2, embedding_dim)
        """
        # Get batch size from input tensor (dynamic for variable batch sizes)
        batch_size = tensorflow.shape(inputs)[0]

        # Replicate CLS token for all samples in the batch
        # From shape (1, 1, embedding_dim) to (batch_size, 1, embedding_dim)
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])

        # Replicate distillation token for all samples in the batch
        # From shape (1, 1, embedding_dim) to (batch_size, 1, embedding_dim)
        dist_tokens = tensorflow.tile(self.dist_token, [batch_size, 1, 1])

        # Concatenate tokens with input sequence along the sequence dimension (axis=1)
        # Result: [CLS_token, DIST_token, original_sequence...]
        return tensorflow.concat([cls_tokens, dist_tokens, inputs], axis=1)

    def get_config(self):
        """
        Get the layer configuration for serialization.

        Returns:
            dict: Configuration dictionary containing the layer's parameters
        """
        return super().get_config()