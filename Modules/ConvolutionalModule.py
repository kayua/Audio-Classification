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
    from Modules.Layers.GLU import GLU
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import DepthwiseConv1D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_NUMBER_FILTERS = 128
DEFAULT_SIZE_KERNEL = 3
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_CONVOLUTIONAL_PADDING = "same"


class ConvolutionalModule(Layer):
    """
    A convolutional module that applies a series of transformations to a 1D tensor,
    including point-wise convolutions, depth-wise convolutions, activations, batch normalization,
    and dropout. This module is designed to capture complex features in sequential data.

    The module consists of the following transformations:
        1. **Point-wise Convolution**: A 1x1 convolution (also known as point-wise convolution),
             used to adjust the number of filters.
        2. **GLU Activation**: Gated Linear Units (GLU) to introduce a non-linearity, improving the
             network's ability to model complex patterns.
        3. **Depth-wise Convolution**: A depth-wise separable convolution that applies a convolutional
             kernel to each input channel independently.
        4. **Batch Normalization**: Normalizes activations across the batch to improve training speed
             and stability.
        5. **Swish Activation**: A smooth non-linear activation function that can improve performance
             over ReLU.
        6. **Dropout**: Applied to prevent overfitting during training by randomly setting a fraction
             of the input units to zero.
        7. **Residual Connection**: The input is added to the output of the module, helping to
             mitigate the vanishing gradient problem.

    Args:
        @number_filters (int): The number of filters for the convolutional layers. Default is 64.
        @size_kernel (int): The size of the kernel for the depth-wise convolution. Default is 3.
        @dropout_decay (float): The dropout rate for the dropout layer. Default is 0.5.
        @convolutional_padding (str): Padding type for the convolutional layers ('same' or 'valid').
        Default is 'same'.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Attributes:
        @convolutional_padding (str): Padding type for the convolutional layers.
        @layer_normalization (LayerNormalization): Layer normalization to stabilize the learning process.
        @first_point_wise_convolutional (Conv1D): Point-wise convolution (1x1 convolution) applied
        to the input tensor.
        @glu_activation (GLU): Gated Linear Unit activation applied after the first point-wise convolution.
        @depth_wise_convolutional (DepthwiseConv1D): Depth-wise convolution applied to the tensor.
        @batch_normalization (BatchNormalization): Batch normalization applied to the activations.
        @swish_activation (Activation): Swish activation applied after batch normalization.
        @second_point_wise_convolutional (Conv1D): Another point-wise convolution applied to the output
        of the depth-wise convolution.
        @dropout (Dropout): Dropout layer applied to prevent overfitting.

    Example
    -------
        >>> Create an instance of the convolutional module with custom parameters
        ...     conv_module = ConvolutionalModule(number_filters=128, size_kernel=3)
        ...     # Example input tensor with shape (batch_size, sequence_length, num_features)
        ...     input_tensor = tf.random.normal([32, 100, 64])  # Batch of 32, sequence length of 100, and 64 features
        ...     # Apply the convolutional module to the input tensor
        ...     output_tensor = conv_module(input_tensor)
        ...     # Print the output shape
        ...     print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
    Output tensor shape: (32, 100, 1)
    """

    def __init__(self,
                 number_filters: int = DEFAULT_NUMBER_FILTERS,
                 size_kernel: int = DEFAULT_SIZE_KERNEL,
                 dropout_decay: float = DEFAULT_DROPOUT_RATE,
                 convolutional_padding: str = DEFAULT_CONVOLUTIONAL_PADDING,
                 **kwargs):
        """
        Initializes the ConvolutionalModule with the specified parameters.

        Parameters
        ----------
        number_filters : int
            Number of filters for the convolutional layers.
        size_kernel : int
            Size of the kernel for the depth-wise convolutional layer.
        dropout_decay : float
            Dropout rate for the dropout layer.
        convolutional_padding : str
            Padding type for the convolutional layers.
        **kwargs : Additional keyword arguments.
        """
        super(ConvolutionalModule, self).__init__(**kwargs)
        self.convolutional_padding = convolutional_padding
        self.layer_normalization = LayerNormalization()
        self.first_point_wise_convolutional = Conv1D(number_filters * 2, kernel_size=1)
        self.glu_activation = GLU()
        self.depth_wise_convolutional = DepthwiseConv1D(kernel_size=size_kernel, padding=self.convolutional_padding)
        self.batch_normalization = BatchNormalization()
        self.swish_activation = Activation(tensorflow.nn.swish)
        self.second_point_wise_convolutional = Conv1D(1, kernel_size=1)
        self.dropout = Dropout(dropout_decay)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the convolutional transformations to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor to be processed by the convolutional module.

        Returns
        -------
        tf.Tensor
            Output tensor after applying the convolutional transformations.
        """
        residual_flow = neural_network_flow
        neural_network_flow = self.layer_normalization(neural_network_flow)

        neural_network_flow = self.first_point_wise_convolutional(neural_network_flow)
        neural_network_flow = self.glu_activation(neural_network_flow)

        neural_network_flow = self.depth_wise_convolutional(neural_network_flow)

        neural_network_flow = self.batch_normalization(neural_network_flow)
        neural_network_flow = self.swish_activation(neural_network_flow)
        neural_network_flow = self.second_point_wise_convolutional(neural_network_flow)
        neural_network_flow = self.dropout(neural_network_flow)

        return neural_network_flow + residual_flow
