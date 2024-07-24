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
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default values for convolutional parameters
DEFAULT_NUMBER_FILTERS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_CONVOLUTIONAL_STRIDES = 2
DEFAULT_CONVOLUTIONAL_PADDING = "same"


class ConvolutionalSubsampling(Layer):
    """
    A class for performing convolutional subsampling on 1D input data.

    This layer applies a 1D convolution operation to the input data, which can be useful for downsampling
    or feature extraction in various deep learning models.

    Attributes
    ----------
    number_filters : int
        Number of filters in the convolutional layer.
    kernel_size : int
        Size of the convolutional kernel.
    convolutional_padding : str
        Padding method for the convolutional operation. Can be 'valid' or 'same'.
    convolutional_stride : int
        Stride length for the convolutional operation.
    convolutional_sub_sampling : Conv1D
        The Conv1D layer used for convolutional subsampling.

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
        number_filters : int, optional
            Number of filters in the convolutional layer (default is 32).
        kernel_size : int, optional
            Size of the convolutional kernel (default is 3).
        convolutional_padding : str, optional
            Padding method for the convolutional operation (default is 'same').
        convolutional_stride : int, optional
            Stride length for the convolutional operation (default is 2).
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(ConvolutionalSubsampling, self).__init__(**kwargs)

        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.convolutional_padding = convolutional_padding
        self.convolutional_stride = convolutional_stride

        self.convolutional_sub_sampling = Conv1D(self.number_filters, self.kernel_size,
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
        return self.convolutional_sub_sampling(neural_network_flow)
