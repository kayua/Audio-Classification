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

DEFAULT_MAX_LENGTH = 128
DEFAULT_EMBEDDING_DIMENSION = 64


class RelativePositionalEmbedding(Layer):
    """
    A layer that adds relative positional embeddings to the input tensor.

    This layer creates and applies positional embeddings to the input tensor to encode positional information.

    Attributes
    ----------
    max_length : int
        The maximum length of the sequence for which positional embeddings are created.
    embedding_dimension : int
        The dimension of the positional embeddings.
    positional_embeddings : tf.Variable
        The positional embeddings matrix, initialized with a uniform distribution.
    """

    def __init__(self,
                 max_length=DEFAULT_MAX_LENGTH,
                 embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
                 **kwargs):
        """
        Initializes the RelativePositionalEmbedding layer with the given parameters.

        Parameters
        ----------
        max_length : int
            The maximum length of the sequence for which positional embeddings are created.
        embedding_dimension : int
            The dimension of the positional embeddings.
        **kwargs : Additional keyword arguments.
        """
        super(RelativePositionalEmbedding, self).__init__(**kwargs)
        self.positional_embeddings = None
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        """
        Builds the layer by creating the positional embeddings matrix.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor, used to initialize the positional embeddings matrix.
        """
        self.positional_embeddings = self.add_weight(
            name="pos_embed",
            shape=(self.max_length, self.embedding_dimension),
            initializer='uniform'
        )

    def call(self, inputs):
        """
        Adds the relative positional embeddings to the input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor to which positional embeddings will be added.

        Returns
        -------
        tf.Tensor
            The input tensor with the positional embeddings added.
        """
        positional_index = tensorflow.range(start=0, limit=self.max_length, delta=1)
        positional_embedding = tensorflow.nn.embedding_lookup(self.positional_embeddings, positional_index)
        return inputs + positional_embedding[:tensorflow.shape(inputs)[1], :]