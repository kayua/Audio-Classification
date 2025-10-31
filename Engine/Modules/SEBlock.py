#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']


# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
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
    import numpy
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow as tf
    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Multiply


except ImportError as error:
    print(error)
    sys.exit(-1)


class SEBlock(Layer):
    """
    Squeeze-and-Excitation Block

    Aplica atenção aos canais da feature map, permitindo que o modelo
    aprenda quais canais são mais importantes.

    Args:
        filters: Número de filtros de saída
        se_ratio: Razão de redução no squeeze (padrão: 0.25)
    """

    def __init__(self, filters, se_ratio=0.25, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio
        self.num_reduced_filters = max(1, int(filters * se_ratio))

    def build(self, input_shape):
        self.squeeze = GlobalAveragePooling2D(name=f'{self.name}_squeeze')
        self.excitation_1 = Dense(
            self.num_reduced_filters,
            activation='swish',
            name=f'{self.name}_excite_fc1'
        )
        self.excitation_2 = Dense(
            self.filters,
            activation='sigmoid',
            name=f'{self.name}_excite_fc2'
        )

    def call(self, inputs):
        # Squeeze: Global Average Pooling
        se = self.squeeze(inputs)
        # Excitation: FC -> Swish -> FC -> Sigmoid
        se = self.excitation_1(se)
        se = self.excitation_2(se)
        # Scale: Multiply input by channel weights
        se = tf.reshape(se, [-1, 1, 1, self.filters])
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'se_ratio': self.se_ratio
        })
        return config