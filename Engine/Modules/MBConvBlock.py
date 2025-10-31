#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

from Engine.Layers.Swish import Swish
from Engine.Modules.SEBlock import SEBlock

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

    from Engine.Models.Process.EfficientNet_Process import ProcessEfficientNet
    from Engine.GradientMap.EfficientNetGradientMaps import EfficientNetGradientMaps

except ImportError as error:
    print(error)
    sys.exit(-1)

class MBConvBlock(Layer):

    def __init__(self, filters_in, filters_out, kernel_size, strides,
                 expand_ratio=6, se_ratio=0.25, drop_rate=0.0, **kwargs):
        super(MBConvBlock, self).__init__(**kwargs)

        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_rate = drop_rate

        # Se stride=1 e canais iguais, usa skip connection
        self.use_residual = (strides == 1 and filters_in == filters_out)

        # Número de filtros expandidos
        self.filters_expanded = filters_in * expand_ratio

    def build(self, input_shape):
        # 1. Expansion phase (se expand_ratio > 1)
        if self.expand_ratio != 1:
            self.expand_conv = Conv2D(self.filters_expanded,
                                      kernel_size=1,
                                      padding='same',
                                      use_bias=False,
                                      name=f'{self.name}_expand_conv')

            self.expand_bn = BatchNormalization(name=f'{self.name}_expand_bn')
            self.expand_activation = Swish(name=f'{self.name}_expand_swish')

        # 2. Depthwise Convolution
        self.depthwise_conv = DepthwiseConv2D(kernel_size=self.kernel_size,
                                              strides=self.strides,
                                              padding='same',
                                              use_bias=False,
                                              name=f'{self.name}_dwconv')

        self.depthwise_bn = BatchNormalization(name=f'{self.name}_dwconv_bn')
        self.depthwise_activation = Swish(name=f'{self.name}_dwconv_swish')

        # 3. Squeeze-and-Excitation
        if self.se_ratio > 0:
            self.se_block = SEBlock(self.filters_expanded if self.expand_ratio != 1 else self.filters_in,
                                    se_ratio=self.se_ratio, name=f'{self.name}_se')

        # 4. Projection phase
        self.project_conv = Conv2D(self.filters_out,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False,
                                   name=f'{self.name}_project_conv')

        self.project_bn = BatchNormalization(name=f'{self.name}_project_bn')

        # 5. Stochastic Depth (opcional)
        if self.drop_rate > 0 and self.use_residual:
            self.dropout = Dropout(self.drop_rate, noise_shape=(None, 1, 1, 1), name=f'{self.name}_drop')

    def call(self, inputs, training=None):
        x = inputs

        # 1. Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_activation(x)

        # 2. Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)

        # 3. Squeeze-and-Excitation
        if self.se_ratio > 0:
            x = self.se_block(x)

        # 4. Projection
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        # 5. Skip connection + Stochastic Depth
        if self.use_residual:
            if self.drop_rate > 0:
                x = self.dropout(x, training=training)
            x = Add(name=f'{self.name}_add')([inputs, x])

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters_in': self.filters_in,
            'filters_out': self.filters_out,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'expand_ratio': self.expand_ratio,
            'se_ratio': self.se_ratio,
            'drop_rate': self.drop_rate
        })
        return config
