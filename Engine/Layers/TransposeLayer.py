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


class TransposeLayer(Layer):
    """
    A custom layer that performs a permutation of the input tensor's dimensions.

    This layer allows you to reorder the dimensions of a tensor according to the specified
    permutation. The `perm` argument defines the order in which the dimensions of the tensor
    will be permuted, enabling flexible manipulation of the tensor's shape. This can be useful
    when working with different data formats or performing operations like matrix transpositions.

    Attributes:
        channels_permutation (list of int): A list or tuple of integers defining the permutation
                                             of the tensor dimensions. For example, [0, 2, 1] swaps
                                             the second and third dimensions of the input tensor.

    Example usage:
    ---------------
    # Create an instance of the TransposeLayer
    transpose_layer = TransposeLayer(perm=[0, 2, 1])

    # Example input tensor of shape (batch_size, sequence_length, embedding_dim)
    input_tensor = tf.random.normal([32, 100, 128])  # Batch of 32, sequence length of 100, 128 features

    # Apply the TransposeLayer
    output_tensor = transpose_layer(input_tensor)

    # Print the output tensor's shape
    print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
    Output tensor shape: (32, 128, 100)
    """

    def __init__(self, perm, **kwargs):
        """
        Initializes the TransposeLayer with the specified permutation.

        Args:
            perm (list or tuple of int): The permutation order of the tensor's dimensions.
                                         This is a list of integers where each integer represents
                                         the index of the dimension in the output tensor.
            **kwargs: Additional arguments passed to the base `Layer` class (e.g., name, trainable).
        """

        # Call to the parent class (Layer) constructor to initialize the layer
        super(TransposeLayer, self).__init__(**kwargs)
        self.channels_permutation = perm

    def call(self, inputs):
        """
        Performs the permutation of the input tensor's dimensions based on the `perm` argument.

        Args:
            inputs (Tensor): The input tensor whose dimensions are to be permuted.

        Returns:
            Tensor: The output tensor with dimensions permuted as per the `perm` argument.
        """
        # Perform the permutation of the input tensor's dimensions
        return tensorflow.transpose(inputs, perm=self.channels_permutation)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the transposed tensor.

        Args:
            input_shape (tuple of int): The shape of the input tensor.

        Returns:
            list of int: The shape of the output tensor after permutation.
        """
        # Compute the output shape by reordering the input dimensions according to `perm`
        return [input_shape[dim] for dim in self.channels_permutation]

    def get_config(self):
        """
        Returns the configuration of the TransposeLayer for serialization.

        Returns:
            dict: A dictionary containing the configuration of the layer, including the `perm` argument.
        """
        # Get the base layer's configuration and add the `perm` attribute
        config = super(TransposeLayer, self).get_config()
        config.update({
            'perm': self.channels_permutation
        })
        return config
