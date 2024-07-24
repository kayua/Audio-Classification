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
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import LayerNormalization
    from Layers.RelativePositionalEmbedding import RelativePositionalEmbedding

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default values for multi-head self-attention parameters
DEFAULT_EMBEDDING_DIMENSION = 64
DEFAULT_NUMBER_HEADS = 8
DEFAULT_DROPOUT_DECAY = 0.1


class MultiHeadSelfAttention(Layer):
    """
    Implements the multi-head self-attention mechanism as described in the Transformer architecture.

    Multi-head self-attention allows the model to focus on different parts of the input sequence
    simultaneously, capturing various aspects of relationships between elements.

    Attributes
    ----------
    embedding_dimension : int
        Dimensionality of the embedding space.
    number_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate for regularization.
    depth : int
        Dimensionality of each attention head.
    weight_query : Dense
        Dense layer for generating query vectors.
    weight_key : Dense
        Dense layer for generating key vectors.
    weight_value : Dense
        Dense layer for generating value vectors.
    dense : Dense
        Dense layer for output transformation.
    dropout : Dropout
        Dropout layer for regularization.
    """

    def __init__(self,
                 embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
                 number_heads: int = DEFAULT_NUMBER_HEADS,
                 dropout_decay: float = DEFAULT_DROPOUT_DECAY,
                 **kwargs):
        """
        Initializes the MultiHeadSelfAttention layer with the given parameters.

        Parameters
        ----------
        embedding_dimension : int, optional
            Dimensionality of the embedding space (default is 64).
        number_heads : int, optional
            Number of attention heads (default is 8).
        dropout_decay : float, optional
            Dropout rate for regularization (default is 0.1).
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.dropout_rate = dropout_decay
        # Ensure the embedding dimension is divisible by the number of heads
        assert embedding_dimension % number_heads == 0, "Embedding dimension must be divisible by number of heads."

        # Compute the depth of each attention head
        self.depth = embedding_dimension // number_heads

        # Define the layers for generating query, key, and value vectors
        self.weight_query = Dense(embedding_dimension)
        self.weight_key = Dense(embedding_dimension)
        self.weight_value = Dense(embedding_dimension)

        # Dense layer for output transformation
        self.dense = Dense(embedding_dimension)

        # Dropout layer for regularization
        self.dropout = Dropout(dropout_decay)

    def split_heads(self, tensor_flow: tensorflow.Tensor, batch_size: int) -> tensorflow.Tensor:
        """
        Splits the input tensor into multiple attention heads.

        Parameters
        ----------
        tensor_flow : tf.Tensor
            Input tensor to be split.
        batch_size : int
            Number of samples in the batch.

        Returns
        -------
        tf.Tensor
            Tensor reshaped and split into multiple attention heads.
        """
        # Reshape tensor to (batch_size, sequence_length, number_heads, depth)
        tensor_flow = tensorflow.reshape(tensor_flow, (batch_size, -1, self.number_heads, self.depth))
        # Transpose to (batch_size, number_heads, sequence_length, depth)
        return tensorflow.transpose(tensor_flow, perm=[0, 2, 1, 3])

    def call(self, inputs: tensorflow.Tensor) -> tuple:
        """
        Applies the multi-head self-attention mechanism to the input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to which the attention mechanism is applied.

        Returns
        -------
        tuple
            Tuple containing the output tensor and attention weights.
        """
        batch_size = tensorflow.shape(inputs)[0]

        # Generate query, key, and value vectors
        query = self.weight_query(inputs)
        key = self.weight_key(inputs)
        value = self.weight_value(inputs)

        # Split into multiple attention heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Compute the scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)

        # Transpose and reshape the output tensor
        scaled_attention = tensorflow.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tensorflow.reshape(scaled_attention, (batch_size, -1, self.embedding_dimension))

        # Apply final dense layer and dropout
        neural_flow = self.dense(concat_attention)
        neural_flow = self.dropout(neural_flow)
        return neural_flow, attention_weights

    @staticmethod
    def scaled_dot_product_attention(query: tensorflow.Tensor, key: tensorflow.Tensor,
                                     value: tensorflow.Tensor) -> tuple:
        """
        Computes the scaled dot-product attention.

        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        key : tf.Tensor
            Key tensor.
        value : tf.Tensor
            Value tensor.

        Returns
        -------
        tuple
            Tuple containing the output tensor and attention weights.
        """
        # Compute dot-product of query and key tensors
        matrix_query_key = tensorflow.matmul(query, key, transpose_b=True)

        # Scale the dot-product by the square root of the key size
        key_size = tensorflow.cast(tensorflow.shape(key)[-1], tensorflow.float32)
        scaled_attention_flow = matrix_query_key / tensorflow.math.sqrt(key_size)

        # Compute attention weights
        attention_weights = tensorflow.nn.softmax(scaled_attention_flow, axis=-1)

        # Compute the output tensor
        neural_flow = tensorflow.matmul(attention_weights, value)
        return neural_flow, attention_weights
