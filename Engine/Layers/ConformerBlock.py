#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
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

    from tensorflow.keras.layers import LayerNormalization

    from Engine.Modules.FeedForwardModule import FeedForwardModule

    from Engine.Modules.ConvolutionalModule import ConvolutionalModule

    from Engine.Modules.MultiheadSelfAttentionModule import MultiHeadSelfAttentionModule

except ImportError as error:
    print(error)
    sys.exit(-1)

# Default configuration values
DEFAULT_EMBEDDING_DIMENSION = 128
DEFAULT_NUMBER_HEADS = 8
DEFAULT_MAX_LENGTH = 100
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT_DECAY = 0.1


class ConformerBlock(Layer):
    """
    Conformer Block: A hybrid architecture combining self-attention, convolution, and feed-forward layers.

    This implementation is based on the Conformer model proposed by Gulati et al. (2020) in the paper:
    "Conformer: Convolution-Augmented Transformer for Speech Recognition".

    Reference:
        Anmol Gulati, James Qin, et al. "Conformer: Convolution-Augmented Transformer for Speech Recognition."
        Interspeech 2020. [https://arxiv.org/abs/2005.08100]

    The Conformer Block consists of:

        - **Layer Normalization**: Normalizes input to stabilize training.
        - **Feed-Forward Network (FFN)**: Enhances feature extraction through dense layers.
        - **Multi-Head Self-Attention (MHSA)**: Captures long-range dependencies in sequences.
        - **Convolutional Module**: Extracts local contextual features.
        - **Final Feed-Forward Network (FFN)**: Enhances representation learning.

    Attributes:
        @embedding_dimension (int): Dimensionality of the input embeddings.
        @number_heads (int): Number of attention heads in the MHSA layer.
        @max_length (int): Maximum sequence length for self-attention.
        @size_kernel (int): Kernel size for the convolutional module.
        @dropout_decay (float): Dropout rate used in FFN and convolutional layers.
        @first_layer_normalization (LayerNormalization): First normalization layer.
        @first_feedforward_module (FeedForwardModule): First FFN applied after normalization.
        @multi_head_self_attention (MultiHeadSelfAttentionModule): MHSA module.
        @convolutional (ConvolutionalModule): Convolutional module after MHSA.
        @second_layer_normalization (LayerNormalization): Second normalization layer.
        @second_feedforward_module (FeedForwardModule): Final FFN after convolutional operations.

    Example:
        >>> python
        ...    import tensorflow as tf
        ...    #Define input tensor
        ...    input_tensor = tf.random.normal([32, 100, 128])  # Batch size = 32, Sequence length = 100, Embedding size = 128
        ...    # Initialize Conformer Block
        ...    conformer_block = ConformerBlock(embedding_dimension=128, number_heads=8, max_length=100, size_kernel=3, dropout_decay=0.1)
        ...    # Forward pass
        ...    output_tensor = conformer_block(input_tensor)
        ...    print(output_tensor.shape)  # Expected output: (32, 100, 128)
        >>>
    """

    def __init__(self,
                 embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
                 number_heads: int = DEFAULT_NUMBER_HEADS,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 size_kernel: int = DEFAULT_KERNEL_SIZE,
                 dropout_decay: float = DEFAULT_DROPOUT_DECAY,
                 **kwargs):
        """
        Initializes the Conformer Block with specified hyperparameters.

        Args:
            embedding_dimension (int, optional): Size of the input embeddings. Default is `DEFAULT_EMBEDDING_DIMENSION`.
            number_heads (int, optional): Number of attention heads in MHSA. Default is `DEFAULT_NUMBER_HEADS`.
            max_length (int, optional): Maximum sequence length for positional embeddings. Default is `DEFAULT_MAX_LENGTH`.
            size_kernel (int, optional): Kernel size for the convolutional layer. Default is `DEFAULT_KERNEL_SIZE`.
            dropout_decay (float, optional): Dropout rate for FFN and convolution. Default is `DEFAULT_DROPOUT_DECAY`.
            **kwargs: Additional keyword arguments for the base `Layer` class.
        """
        super(ConformerBlock, self).__init__(**kwargs)
        self.first_layer_normalization = LayerNormalization()
        self.first_feedforward_module = FeedForwardModule(embedding_dimension, dropout_decay)
        self.multi_head_self_attention = MultiHeadSelfAttentionModule(embedding_dimension, number_heads, max_length,
                                                                      dropout_decay)
        self.convolutional = ConvolutionalModule(embedding_dimension, size_kernel, dropout_decay)
        self.second_layer_normalization = LayerNormalization()
        self.second_feedforward_module = FeedForwardModule(embedding_dimension, dropout_decay)

    def call(self, neural_network_flow: tensorflow.Tensor, mask: tensorflow.Tensor = None) -> tensorflow.Tensor:
        """
        Applies the Conformer Block transformation pipeline.

        This includes:
            1. Layer Normalization → Feed-Forward Network
            2. Layer Normalization → Multi-Head Self-Attention
            3. Layer Normalization → Convolutional Module
            4. Layer Normalization → Final Feed-Forward Network

        Args:
            neural_network_flow (tensorflow.Tensor): Input tensor with shape `(batch_size, seq_length, embedding_dim)`.
            mask (tensorflow.Tensor, optional): Attention mask for self-attention. Default is `None`.

        Returns:
            tensorflow.Tensor: Processed tensor after applying Conformer Block transformations.
        """
        neural_flow_normalized = self.first_layer_normalization(neural_network_flow)

        first_feedforward_flow = self.first_feedforward_module(neural_flow_normalized)
        neural_network_flow = neural_network_flow + first_feedforward_flow

        first_layer_normalization_flow = self.first_layer_normalization(neural_network_flow)
        mult_head_flow = self.multi_head_self_attention(first_layer_normalization_flow)
        neural_network_flow = neural_network_flow + mult_head_flow

        convolutional_flow = self.convolutional(self.first_layer_normalization(neural_network_flow))
        neural_network_flow = neural_network_flow + convolutional_flow

        second_feedforward_flow = self.second_feedforward_module(self.second_layer_normalization(neural_network_flow))
        neural_network_flow = neural_network_flow + second_feedforward_flow

        return neural_network_flow

    @staticmethod
    def compute_output_shape(input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """
        Computes the output shape of the Conformer Block given an input shape.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input tensor `(batch_size, seq_length, embedding_dim)`.

        Returns:
            tuple[int, int, int]: Output shape `(batch_size, seq_length, embedding_dim)`, identical to the input shape.
        """
        return input_shape