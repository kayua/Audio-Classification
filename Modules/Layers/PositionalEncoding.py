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
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class PositionalEncoding(Layer):
    """
    Custom TensorFlow layer that adds positional encoding to input embeddings.
    This layer is typically used in Transformer models to inject information
    about the position of each token in a sequence.

    Args:
        max_sequence_length (int): The maximum length of the input sequences.
        embedding_dimension (int): The dimensionality of the input embeddings.

    Attributes:
        positional_encodings (tf.Tensor): A tensor containing the precomputed positional encodings.
        max_sequence_length (int): The maximum sequence length for which positional encodings are computed.
        embedding_dimension (int): The dimensionality of the embeddings.
    """

    def __init__(self, max_sequence_length, embedding_dimension):
        """
        Initializes the PositionalEncoding layer with the specified maximum sequence length
        and embedding dimension.

        Args:
            max_sequence_length (int): The maximum length of input sequences.
            embedding_dimension (int): The dimensionality of input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = None
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        """
        Builds the PositionalEncoding layer by computing the positional encodings.
        This method is called automatically before the layer is used for the first time.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        # Precompute the positional encodings for the maximum sequence length and embedding dimension
        self.positional_encodings = self._get_positional_encodings(self.max_sequence_length, self.embedding_dimension)

    @staticmethod
    def _get_positional_encodings(max_seq_length, embedding_dimension):
        """
        Computes the positional encodings for the given sequence length and embedding dimension.
        The encodings are based on sine and cosine functions of different frequencies.

        Args:
            max_seq_length (int): The maximum length of input sequences.
            embedding_dimension (int): The dimensionality of input embeddings.

        Returns:
            tf.Tensor: A tensor of shape (1, max_seq_length, embedding_dimension) containing
            the positional encodings.
        """
        # Create a range of positions for the sequence
        positional_array = tensorflow.range(max_seq_length, dtype=tensorflow.float32)[:, tensorflow.newaxis]

        # Create an index array for the embedding dimensions
        index = tensorflow.range(embedding_dimension, dtype=tensorflow.float32)[tensorflow.newaxis, :]

        # Compute the angles for the sine and cosine functions
        angles = positional_array / tensorflow.pow(10000.0, (2 * (index // 2)) / tensorflow.cast(embedding_dimension,
                                                                                                 tensorflow.float32))

        # Apply sine to even indices in the embedding dimensions
        angles_sin = tensorflow.math.sin(angles[:, 0::2])

        # Apply cosine to odd indices in the embedding dimensions
        angles_cos = tensorflow.math.cos(angles[:, 1::2])

        # Concatenate the sine and cosine encodings along the last axis
        positional_encodings = tensorflow.concat([angles_sin, angles_cos], axis=-1)

        # Add a batch dimension to the positional encodings
        return positional_encodings[tensorflow.newaxis, ...]

    def call(self, x):
        """
        Adds the precomputed positional encodings to the input embeddings.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            tf.Tensor: A tensor of the same shape as the input, with positional encodings added
            to the embeddings.
        """
        # Get the sequence length of the input tensor
        sequence_length = tensorflow.shape(x)[1]

        # Retrieve the positional encodings corresponding to the current sequence length
        positional_encodings = self.positional_encodings[:, :sequence_length, :]

        # Add the positional encodings to the input embeddings
        return x + positional_encodings
