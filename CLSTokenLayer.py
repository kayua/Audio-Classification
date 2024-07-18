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

DEFAULT_PROJECTION_DIMENSION = 64


class CLSTokenLayer(Layer):
    def __init__(self, projection_dimension=DEFAULT_PROJECTION_DIMENSION, **kwargs):
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.cls_token = None
        self.projection_dimension = projection_dimension

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dimension),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, inputs):
        batch_size = tensorflow.shape(inputs)[0]
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])
        return cls_tokens
