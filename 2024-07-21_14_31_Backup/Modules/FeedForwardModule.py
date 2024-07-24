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
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense

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

    This module consists of two dense layers with an activation function and dropout applied between them.
    It is designed to process the flow of data through a feed-forward neural network in a neural network architecture.

    Attributes:
        embedding_dimension (int): Dimensionality of the output embeddings.
        dropout_rate (float): Dropout rate applied after each dense layer.
        activation_function (function): Activation function used in the first dense layer.
        factor_projection (int): Factor by which the dimensionality is increased in the first dense layer.
        first_dense_layer (Dense): First dense layer with an increased dimensionality and specified activation function.
        dropout1 (Dropout): Dropout layer applied after the first dense layer.
        second_dense_layer (Dense): Second dense layer that projects the data back to the original embedding dimension.
        dropout2 (Dropout): Dropout layer applied after the second dense layer.
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
            embedding_dimension (int, optional): Dimensionality of the output embeddings. Default is 64.
            dropout_rate (float, optional): Dropout rate applied after each dense layer. Default is 0.1.
            activation_function (function, optional): Activation function used in the first dense layer. Default is tensorflow.nn.swish.
            factor_projection (int, optional): Factor by which the dimensionality is increased in the first dense layer. Default is 4.
            **kwargs: Additional arguments passed to the base `Layer` class.
        """
        super(FeedForwardModule, self).__init__(**kwargs)
        self.factor_projection = factor_projection
        self.embedding_dimension = embedding_dimension
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.first_dense_layer = Dense(embedding_dimension * factor_projection, activation=self.activation_function)
        self.dropout1 = Dropout(self.dropout_rate)
        self.second_dense_layer = Dense(embedding_dimension)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, neural_network_flow):
        """
        Applies the feed-forward operations: dense layer, dropout, another dense layer, and dropout.

        Args:
            neural_network_flow (Tensor): Input tensor to be processed through the dense layers and dropout.

        Returns:
            Tensor: Output tensor after applying the first dense layer, dropout, second dense layer, and dropout.
        """
        neural_network_flow = self.first_dense_layer(neural_network_flow)
        neural_network_flow = self.dropout1(neural_network_flow)
        neural_network_flow = self.second_dense_layer(neural_network_flow)
        neural_network_flow = self.dropout2(neural_network_flow)
        return neural_network_flow