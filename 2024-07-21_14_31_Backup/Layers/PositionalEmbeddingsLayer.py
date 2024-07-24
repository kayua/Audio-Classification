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
    import librosa
    import tensorflow
    from tensorflow.keras.layers import Add, Layer
    from tensorflow.keras.layers import Embedding

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default values for positional embeddings parameters
DEFAULT_PROJECTION_DIMENSION = 64
DEFAULT_NUMBER_PATCH = 8


class PositionalEmbeddingsLayer(Layer):
    """
    Implements a layer for generating positional embeddings.

    Positional embeddings are used to provide positional information to the model,
    indicating the position of each patch in the input sequence.

    Attributes
    ----------
    number_patches : int
        Number of patches or positions for which embeddings will be generated.
    projection_dimension : int
        Dimensionality of the positional embeddings.
    embedding_layer : Embedding
        Keras Embedding layer to generate positional embeddings.
    """

    def __init__(self, number_patches: int = DEFAULT_NUMBER_PATCH,
                 projection_dimension: int = DEFAULT_PROJECTION_DIMENSION,
                 **kwargs):
        """
        Initializes the PositionalEmbeddingsLayer with the given parameters.

        Parameters
        ----------
        number_patches : int, optional
            Number of patches for which positional embeddings are created (default is 8).
        projection_dimension : int, optional
            Dimensionality of the positional embeddings (default is 64).
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(PositionalEmbeddingsLayer, self).__init__(**kwargs)
        self.number_patches = number_patches
        self.projection_dimension = projection_dimension

        # Create an embedding layer for positional embeddings
        self.embedding_layer = Embedding(input_dim=number_patches + 1, output_dim=projection_dimension)

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Computes the positional embeddings for the input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor (e.g., input sequence) to determine the batch size.

        Returns
        -------
        tf.Tensor
            Tensor containing the positional embeddings, with shape (batch_size, number_patches, projection_dimension).
        """

        # Create a range of positions
        positions = tensorflow.range(start=0, limit=self.number_patches + 1, delta=1)

        # Generate positional embeddings
        positional_embeddings = self.embedding_layer(positions)

        # Add a batch dimension
        positional_embeddings = tensorflow.expand_dims(positional_embeddings, axis=0)

        # Get the batch size
        batch_size = tensorflow.shape(inputs)[0]

        # Tile the positional embeddings for each sample in the batch
        positional_embeddings = tensorflow.tile(positional_embeddings, [batch_size, 1, 1])

        return positional_embeddings
