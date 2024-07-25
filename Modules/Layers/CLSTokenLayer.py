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

# Default value for the projection dimension
DEFAULT_PROJECTION_DIMENSION = 64


class CLSTokenLayer(Layer):
    """
    Implements a layer to add a [CLS] token to the input sequences.

    The [CLS] token is a special token used in various models (e.g., BERT) to aggregate
    information from the entire input sequence for classification tasks.

    Attributes
    ----------
    cls_token : tf.Variable
        The [CLS] token variable to be added to the input sequences.
    projection_dimension : int
        The dimensionality of the [CLS] token embeddings.
    """

    def __init__(self, projection_dimension: int = DEFAULT_PROJECTION_DIMENSION, **kwargs):
        """
        Initializes the CLSTokenLayer with the given projection dimension.

        Parameters
        ----------
        projection_dimension : int, optional
            Dimensionality of the [CLS] token embeddings (default is 64).
        **kwargs
            Additional keyword arguments for the Layer superclass.
        """
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.cls_token = None
        self.projection_dimension = projection_dimension

    def build(self, input_shape):
        """
        Creates the [CLS] token as a trainable weight variable.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor. This is used to determine the batch size.
        """
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dimension),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Adds the [CLS] token to the input sequences.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor (e.g., input sequence) to determine the batch size.

        Returns
        -------
        tf.Tensor
            Tensor containing the [CLS] tokens for each sample in the batch,
            with shape (batch_size, 1, projection_dimension).
        """
        batch_size = tensorflow.shape(inputs)[0]
        # Tile the [CLS] token to match the batch size
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])
        return cls_tokens
