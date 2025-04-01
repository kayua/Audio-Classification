#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
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

    import librosa
    import tensorflow

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Layer

    from tensorflow.keras.layers import Embedding

except ImportError as error:
    print(error)
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
