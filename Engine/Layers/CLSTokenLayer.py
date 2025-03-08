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
    sys.exit(-1)

# Default value for the projection dimension
DEFAULT_PROJECTION_DIMENSION = 64


class CLSTokenLayer(Layer):
    """
    Implements a layer to add a [CLS] (classification) token to the input sequences.

    The [CLS] token is a special token often used in transformer-based models like BERT to aggregate
    information from the entire input sequence, typically for classification tasks. This token is
    appended to the beginning of each input sequence, enabling the model to learn an aggregate
    representation of the sequence for downstream tasks, such as sentence classification.

    The [CLS] token is typically initialized as a learnable parameter and is added to the input tensor
    during the model's forward pass. It is returned as part of the output, which can then be further
    processed by subsequent layers.

    Attributes:
    -----------
    cls_token : tf.Variable
        The [CLS] token, represented as a learnable embedding. It is shared across all input sequences.

    projection_dimension : int
        The dimensionality of the [CLS] token embedding, which determines the size of the token's
        representation in the model's embedding space.

    Example
    -------
    >>> # Initialize the CLSTokenLayer with a projection dimension of 64
    ...     cls_layer = CLSTokenLayer(projection_dimension=64)
    ...     # Example input tensor of shape (batch_size, sequence_length, embedding_dim)
    ...     input_tensor = tf.random.normal([32, 100, 128])  # Batch of 32, sequence length of 100, 128 features
    ...     # Apply the CLSTokenLayer to the input tensor
    ...     cls_tokens = cls_layer(input_tensor)
    ...     # Print the shape of the output tensor
    >>>     print("Output tensor shape with [CLS] token:", cls_tokens.shape)

    Output:
    -------
    Output tensor shape with [CLS] token: (32, 1, 64)
    """

    def __init__(self, projection_dimension: int = DEFAULT_PROJECTION_DIMENSION, **kwargs):
        """
        Initializes the CLSTokenLayer with the given projection dimension.

        Args:
            projection_dimension (int, optional): Dimensionality of the [CLS] token embeddings.
                                                   Default is 64.
            **kwargs: Additional arguments passed to the `Layer` superclass.

        """
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.cls_token = None
        self.projection_dimension = projection_dimension

    def build(self, input_shape):
        """
        Creates the [CLS] token as a trainable weight variable.

        This method is called once the layer is added to the model and before any computations are
        performed. It creates the [CLS] token as a learnable parameter of shape (1, 1, projection_dimension),
        where `projection_dimension` is the size of the embedding for the [CLS] token.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor. Used to determine the batch size.
        """
        # Initialize the [CLS] token as a trainable variable with the given projection dimension
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dimension),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Adds the [CLS] token to the input sequences.

        This method takes the input tensor, extracts the batch size, and tiles the [CLS] token to
        match the batch size. The [CLS] token is added to each sample in the batch, producing a tensor
        with the shape `(batch_size, 1, projection_dimension)` for the [CLS] token. This token is used
        for downstream classification tasks, where it serves as an aggregate representation of the
        input sequence.

        Args:
            inputs (tf.Tensor): The input tensor (e.g., input sequences) of shape `(batch_size, sequence_length, embedding_dim)`.

        Returns:
            tf.Tensor: A tensor of shape `(batch_size, 1, projection_dimension)`, containing the [CLS] token
                       for each sample in the batch.
        """

        # Get the batch size from the input tensor shape
        batch_size = tensorflow.shape(inputs)[0]

        # Tile the [CLS] token to match the batch size
        # Create a tensor of shape (batch_size, 1, projection_dimension)
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])

        return cls_tokens
