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
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention

    from Engine.Layers.RelativePositionalEmbedding import RelativePositionalEmbedding

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
    with a dropout layer to enhance model performance in natural language processing (NLP) tasks.
    It is particularly useful in transformer architectures to model relationships between tokens within
    a sequence, regardless of their absolute positions.

    The module works by first normalizing the input embeddings, then applying relative positional embeddings
    to encode information about token positions. Afterward, a multi-head self-attention mechanism is applied
    to compute a weighted representation of the input sequence. Finally, dropout is applied to prevent overfitting.

    Attributes:
        @embedding_dimension (int): Dimensionality of the input embeddings (size of each input token's vector representation).
        @num_heads (int): Number of attention heads in the multi-head attention mechanism. More heads allow the model to
                         focus on different parts of the input sequence at the same time.
        @max_len (int): Maximum sequence length for positional embeddings, defining the maximum size the model can handle.
        @dropout_rate (float): Dropout rate applied after the attention layer to prevent overfitting.
        @layer_normalization (LayerNormalization): Instance of layer normalization used to stabilize training.
        @multi_head_self_attention (MultiHeadAttention): Instance of the multi-head attention layer that processes input.
        @relative_pos_embedding (RelativePositionalEmbedding): Instance of the relative positional embedding layer.
        @dropout (Dropout): Dropout layer used after attention to regularize the model.

    References:
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser, Ł., & Polosukhin, I. (2017).
      Attention is all you need. Advances in Neural Information Processing Systems, 30.
    - Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations.
      Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).

    Example:
    -------
        >>>  # Create an instance of the MultiHeadSelfAttentionModule
        ...     attention_module = MultiHeadSelfAttentionModule(embedding_dimension=128, num_heads=8, max_length=512)
        ...     # Example input tensor with shape (batch_size, sequence_length, embedding_dim)
        ...     input_tensor = tf.random.normal([32, 100, 128])  # Batch of 32, sequence length of 100, and 128 features
        ...     # Apply the multi-head self-attention module
        ...     output_tensor = attention_module(input_tensor)
        ...     # Print the output shape
        >>>     print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
    Output tensor shape: (32, 100, 128)
    """

    def __init__(self, embedding_dimension, number_heads, max_length, dropout_rate=0.1, **kwargs):
        """
        Initializes the multi-head self-attention module with normalization and relative positional embeddings.

        Args:
            @embedding_dimension (int): Dimensionality of the input embeddings.
            @number_heads (int): Number of attention heads in the multi-head attention layer.
            @max_length (int): Maximum sequence length for positional embeddings.
            @dropout_rate (float, optional): Dropout rate applied after the attention layer. Default is 0.1.
            **kwargs: Additional arguments passed to the base `Layer` class.
        """
        # Call to the parent class (Layer) constructor to initialize the module
        super(MultiHeadSelfAttentionModule, self).__init__(**kwargs)

        # Layer normalization for stabilizing training and accelerating convergence
        self.layer_normalization = LayerNormalization()

        # Multi-head self-attention mechanism for capturing relationships across all tokens in the input
        self.multi_head_self_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Relative positional embeddings to encode the relative distances between tokens
        self.relative_pos_embedding = RelativePositionalEmbedding(max_length, embedding_dimension)

        # Dropout layer to regularize the attention output and prevent overfitting
        self.dropout = Dropout(dropout_rate)


    @tensorflow.function
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
        # Normalize the input tensor using LayerNormalization
        neural_flow_normalized = self.layer_normalization(inputs)

        # Apply relative positional embeddings to the normalized tensor
        positional_flow = self.relative_pos_embedding(neural_flow_normalized)

        # Perform multi-head self-attention on the positional flow (query, key, value all the same here)
        attention_flow = self.multi_head_self_attention(query=positional_flow,
                                                        value=positional_flow,
                                                        key=positional_flow)

        # Apply dropout to the attention output to regularize the model
        dropout_flow = self.dropout(attention_flow)

        # Return the sum of the input and the dropout-augmented attention output (skip connection)
        return inputs + dropout_flow
