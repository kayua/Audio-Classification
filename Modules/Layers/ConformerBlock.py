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
    from tensorflow.keras.layers import LayerNormalization

    from Modules.ConvolutionalModule import ConvolutionalModule
    from Modules.FeedForwardModule import FeedForwardModule
    from Modules.MultiheadSelfAttentionModule import MultiHeadSelfAttentionModule

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default configuration values
DEFAULT_EMBEDDING_DIMENSION = 128
DEFAULT_NUMBER_HEADS = 8
DEFAULT_MAX_LENGTH = 100
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT_DECAY = 0.1


class ConformerBlock(Layer):
    """
    Conformer Block for combining self-attention, feed-forward, and convolutional layers with normalization.

    This block consists of multiple layers including normalization, multi-head self-attention, feed-forward modules,
    and convolutional operations, designed to enhance feature extraction and contextual understanding.

    Attributes:
        embedding_dimension (int): Dimensionality of the input embeddings.
        number_heads (int): Number of attention heads in the multi-head self-attention layer.
        max_length (int): Maximum sequence length for positional embeddings in self-attention.
        size_kernel (int): Size of the kernel for convolutional operations.
        dropout_decay (float): Dropout rate applied in feed-forward and convolutional modules.
        first_layer_normalization (LayerNormalization): First normalization layer applied to input.
        first_feedforward_module (FeedForwardModule): First feed-forward module applied to the normalized input.
        multi_head_self_attention (MultiHeadSelfAttentionModule): Multi-head self-attention module.
        convolutional (ConvolutionalModule): Convolutional module applied after self-attention.
        second_layer_normalization (LayerNormalization): Second normalization layer applied before the final feed-forward module.
        second_feedforward_module (FeedForwardModule): Second feed-forward module applied after convolutional operations.
    """

    def __init__(self,
                 embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
                 number_heads: int = DEFAULT_NUMBER_HEADS,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 size_kernel: int = DEFAULT_KERNEL_SIZE,
                 dropout_decay: float = DEFAULT_DROPOUT_DECAY,
                 **kwargs):
        """
        Initializes the Conformer Block with specified parameters.

        Args:
            embedding_dimension (int, optional): Dimensionality of the input embeddings. Default is 128.
            number_heads (int, optional): Number of attention heads in the multi-head self-attention layer. Default is 8.
            max_length (int, optional): Maximum sequence length for positional embeddings. Default is 100.
            size_kernel (int, optional): Size of the kernel for convolutional operations. Default is 3.
            dropout_decay (float, optional): Dropout rate applied in feed-forward and convolutional modules. Default is 0.1.
            **kwargs: Additional arguments passed to the base `Layer` class.
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
        Applies the Conformer Block operations: normalization, feed-forward, multi-head self-attention,
        convolutional, and final feed-forward modules.

        Args:
            neural_network_flow (tensorflow.Tensor): Input tensor to be processed through the Conformer Block.
            mask (tensorflow.Tensor, optional): Mask tensor to be used in attention calculations. Default is None.

        Returns:
            tensorflow.Tensor: Output tensor after applying all the layers and operations of the Conformer Block.
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
        Computes the output shape of the Conformer Block given the input shape.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input tensor.

        Returns:
            tuple[int, int, int]: Shape of the output tensor.
        """
        return input_shape