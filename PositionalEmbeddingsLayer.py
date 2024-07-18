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

DEFAULT_PROJECTION_DIMENSION = 64
DEFAULT_NUMBER_PATCH = 8


class PositionalEmbeddingsLayer(Layer):
    def __init__(self, number_patches=DEFAULT_NUMBER_PATCH, projection_dimension=DEFAULT_PROJECTION_DIMENSION,
                 **kwargs):
        super(PositionalEmbeddingsLayer, self).__init__(**kwargs)
        self.number_patches = number_patches
        self.projection_dimension = projection_dimension
        self.embedding_layer = Embedding(input_dim=number_patches + 1, output_dim=projection_dimension)

    def call(self, inputs):
        positions = tensorflow.range(start=0, limit=self.number_patches + 1, delta=1)
        positional_embeddings = self.embedding_layer(positions)
        positional_embeddings = tensorflow.expand_dims(positional_embeddings, axis=0)
        batch_size = tensorflow.shape(inputs)[0]
        positional_embeddings = tensorflow.tile(positional_embeddings, [batch_size, 1, 1])
        return positional_embeddings
