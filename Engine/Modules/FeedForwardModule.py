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
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Dropout

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Default configuration values
DEFAULT_EMBEDDING_DIMENSION = 64
DEFAULT_ACTIVATION_FUNCTION = tensorflow.nn.swish
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_FACTOR_PROJECTION = 4


class FeedForwardModule(Layer):
    """
    Feed-Forward Neural Network Module with Dense Layers and Dropout.

    This module consists of two dense layers with an activation function and
    dropout applied between them. The goal of this module is to process data
    through a fully connected network with regularization techniques, making
    it suitable for tasks like classification, regression, and feature
    transformation in deep learning architectures.

    The first dense layer increases the dimensionality of the input (projection),
    and the second dense layer reduces it back to the original embedding dimension.

    Args:
        @embedding_dimension (int): The dimensionality of the output embeddings
         (final output size). Default is 64.
        @dropout_rate (float): The dropout rate to apply after each dense layer
         to prevent overfitting. Default is 0.1.
        @activation_function (function): The activation function to apply to the
         first dense layer. Default is Swish.
        @factor_projection (int): The factor by which to increase the dimensionality
         in the first dense layer. Default is 4.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Attributes:
        @embedding_dimension (int): Dimensionality of the output embeddings.
        @dropout_rate (float): Dropout rate applied after each dense layer.
        @activation_function (function): Activation function used in the first
         dense layer.
        @factor_projection (int): Factor by which the dimensionality is increased in
         the first dense layer.
        @first_dense_layer (Dense): The first dense layer, applying the activation
         function and dimensionality projection.
        @dropout1 (Dropout): Dropout layer applied after the first dense layer.
        @second_dense_layer (Dense): The second dense layer, reducing the dimensionality
         to the original embedding dimension.
        @dropout2 (Dropout): Dropout layer applied after the second dense layer.

    Example
    -------
        >>>  # Create an instance of the FeedForwardModule with custom parameters
        ...     ff_module = FeedForwardModule(embedding_dimension=128, dropout_rate=0.2)
        ...     # Example input tensor with shape (batch_size, sequence_length, embedding_dim)
        ...     input_tensor = tf.random.normal([32, 100, 64])  # Batch of 32, sequence length of 100, and 64 features
        ...     # Apply the feedforward module to the input tensor
        ...     output_tensor = ff_module(input_tensor)
        ...     # Print the output shape
        >>>     print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
        Output tensor shape: (32, 100, 64)
    """

    def __init__(self,
                 embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
                 dropout_rate=DEFAULT_DROPOUT_RATE,
                 activation_function=DEFAULT_ACTIVATION_FUNCTION,
                 factor_projection=DEFAULT_FACTOR_PROJECTION,
                 **kwargs):
        """
        Initializes the Feed-Forward Module with specified parameters.

        Args:
            @embedding_dimension (int, optional): Dimensionality of the output embeddings.
            @dropout_rate (float, optional): Dropout rate applied after each dense layer.
            @activation_function (function, optional): Activation function used in the first
            @dense layer. Default is tensorflow.nn.swish.
            @factor_projection (int, optional): Factor by which the dimensionality is increased
            @in the first dense layer. Default is 4.
            **kwargs: Additional arguments passed to the base `Layer` class.
        """
        # Calling the parent class (Layer) constructor to initialize the base layer
        super(FeedForwardModule, self).__init__(**kwargs)

        # Storing the parameters for later use in the layers
        self.factor_projection = factor_projection
        self.embedding_dimension = embedding_dimension
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # First dense layer (with dimensionality increase using the factor_projection)
        self.first_dense_layer = Dense(embedding_dimension * factor_projection, activation=self.activation_function)

        # Dropout applied after the first dense layer
        self.first_dropout = Dropout(self.dropout_rate)

        # Second dense layer (reducing the dimensionality back to the original embedding dimension)
        self.second_dense_layer = Dense(embedding_dimension)

        # Dropout applied after the second dense layer
        self.second_dropout = Dropout(self.dropout_rate)

    def call(self, neural_network_flow):
        """
        Applies the feed-forward operations: dense layer, dropout, another dense
         layer, and dropout.

        Args:
            neural_network_flow (Tensor): Input tensor to be processed through
             the dense layers and dropout.

        Returns:
            Tensor: Output tensor after applying the first dense layer, dropout,
             second dense layer, and dropout.
        """

        # Apply the first dense layer to the input tensor (projection and activation)
        neural_network_flow = self.first_dense_layer(neural_network_flow)

        # Apply the first dropout layer after the first dense layer
        neural_network_flow = self.first_dropout(neural_network_flow)

        # Apply the second dense layer to the data (reducing dimensionality back to embedding dimension)
        neural_network_flow = self.second_dense_layer(neural_network_flow)

        # Apply the second dropout layer after the second dense layer
        neural_network_flow = self.second_dropout(neural_network_flow)

        # Return the final output
        return neural_network_flow