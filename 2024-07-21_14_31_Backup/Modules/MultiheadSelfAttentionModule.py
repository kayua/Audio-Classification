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
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import LayerNormalization
    from Layers.MultiHeadSelfAttention import MultiHeadSelfAttention
    from Layers.RelativePositionalEmbedding import RelativePositionalEmbedding

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class MultiHeadSelfAttentionModule(Layer):
    """
    Multi-Head Self-Attention Module with Layer Normalization and Relative Positional Embeddings.

    This module combines layer normalization, multi-head self-attention, and relative positional embeddings
    with a dropout layer to enhance model performance in natural language processing tasks.

    Attributes:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads in the multi-head attention layer.
        max_len (int): Maximum sequence length for positional embeddings.
        dropout_rate (float): Dropout rate applied after the attention layer.
        layer_normalization (LayerNormalization): Instance of layer normalization.
        multi_head_self_attention (MultiHeadSelfAttention): Instance of the multi-head attention layer.
        relative_pos_embedding (RelativePositionalEmbedding): Instance of the relative positional embedding layer.
        dropout (Dropout): Dropout layer.
    """

    def __init__(self, embedding_dimension, number_heads, max_length, dropout_rate=0.1, **kwargs):
        """
        Initializes the multi-head self-attention module with normalization and relative positional embeddings.

        Args:
            embedding_dimension (int): Dimensionality of the input embeddings.
            number_heads (int): Number of attention heads in the multi-head attention layer.
            max_length (int): Maximum sequence length for positional embeddings.
            dropout_rate (float, optional): Dropout rate applied after the attention layer. Default is 0.1.
            **kwargs: Additional arguments passed to the base `Layer` class.
        """
        super(MultiHeadSelfAttentionModule, self).__init__(**kwargs)
        self.layer_normalization = LayerNormalization()
        self.multi_head_self_attention = MultiHeadSelfAttention(embedding_dimension, number_heads, dropout_rate)
        self.relative_pos_embedding = RelativePositionalEmbedding(max_length, embedding_dimension)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        """
        Applies layer normalization, positional embeddings, multi-head self-attention, and dropout,
        returning the sum of the inputs and the dropout-augmented attention output.

        Args:
            inputs (Tensor): Input tensor with embeddings to be processed.

        Returns:
            Tensor: Resulting tensor after applying normalization, positional embeddings,
            multi-head self-attention, and dropout.
        """
        neural_flow_normalized = self.layer_normalization(inputs)
        positional_flow = self.relative_pos_embedding(inputs)
        attention_flow, _ = self.multi_head_self_attention(positional_flow)
        dropout_flow = self.dropout(attention_flow)

        return inputs + dropout_flow
