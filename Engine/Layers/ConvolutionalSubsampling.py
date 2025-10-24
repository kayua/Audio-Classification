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

    import tensorflow

    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv2D

except ImportError as error:
    print(error)
    sys.exit(-1)

# Default values for convolutional parameters
DEFAULT_NUMBER_FILTERS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_CONVOLUTIONAL_STRIDES = (1, 2)
DEFAULT_CONVOLUTIONAL_PADDING = "same"


class ConvolutionalSubsampling(Layer):
    """
    A class for performing convolutional subsampling on 1D input data.

    This layer applies a 1D convolution operation to the input data, which
    can be useful for downsampling or feature extraction in various deep
    learning models.

    Attributes
    ----------
        @number_filters : int Number of filters in the convolutional layer.
        @kernel_size : int Size of the convolutional kernel.
        @convolutional_padding : str Padding method for the convolutional
         operation. Can be 'valid' or 'same'.
        @convolutional_stride : int Stride length for the convolutional operation.
        @convolutional_sub_sampling : Conv1D The Conv1D layer used for convolutional
         subsampling.

    Methods
    -------
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the convolutional subsampling to the input tensor and returns the output tensor.
    """

    def __init__(self,
                 number_filters: int = DEFAULT_NUMBER_FILTERS,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 convolutional_padding: str = DEFAULT_CONVOLUTIONAL_PADDING,
                 convolutional_stride: int = DEFAULT_CONVOLUTIONAL_STRIDES,
                 **kwargs):
        """
        Initializes the ConvolutionalSubsampling layer.

        Parameters
        ----------
        @number_filters : int, optional Number of filters in the convolutional layer (default is 32).
        @kernel_size : int, optional Size of the convolutional kernel (default is 3).
        @convolutional_padding : str, optional Padding method for the convolutional operation (default is 'same').
        @convolutional_stride : int, optional Stride length for the convolutional operation (default is 2).
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(ConvolutionalSubsampling, self).__init__(**kwargs)

        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.convolutional_padding = convolutional_padding
        self.convolutional_stride = convolutional_stride

        self.convolutional_sub_sampling = Conv2D(1, (2, 2),
                                                 strides=self.convolutional_stride,
                                                 padding=self.convolutional_padding)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the convolutional subsampling to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor to which the convolutional subsampling is applied.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the convolutional subsampling.
        """
        return tensorflow.squeeze(self.convolutional_sub_sampling(tensorflow.expand_dims(neural_network_flow, axis=-1)), axis=-1)
