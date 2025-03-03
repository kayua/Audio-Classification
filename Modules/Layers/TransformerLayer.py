#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Modules.Layers.PositionalEncoding import PositionalEncoding
from Modules.Layers.TransformerDecoder import TransformerDecoder
from Modules.Layers.TransformerEncoder import TransformerEncoder

try:
    import sys
    import tensorflow
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class Transformer(Layer):
    """
    Custom TensorFlow layer implementing a complete Transformer architecture,
    including positional encoding, multiple encoder and decoder layers, and
    a final linear transformation.

    Args:
        embedding_dimension (int): The dimensionality of the input embeddings.
        number_heads (int): The number of attention heads in the multi-head attention mechanism.
        feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
        number_layers (int): The number of encoder and decoder layers in the Transformer.
        max_sequence_length (int): The maximum length of the input and target sequences.
        dropout_rate (float): The dropout rate applied in the encoder and decoder layers. Default is 0.1.

    Attributes:
        embedding_dimension (int): The dimensionality of the input embeddings.
        number_layers (int): The number of encoder and decoder layers in the Transformer.
        max_sequence_length (int): The maximum length of the input and target sequences.
        positional_encoding (PositionalEncoding): The positional encoding layer that adds positional information to the embeddings.
        encoder_layers (list of TransformerEncoder): The list of encoder layers in the Transformer.
        decoder_layers (list of TransformerDecoder): The list of decoder layers in the Transformer.
        final_layer (Dense): The final dense layer that projects the decoder output to the desired embedding dimension.
    """

    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, number_layers, max_sequence_length,
                 dropout_rate=0.1):
        """
        Initializes the Transformer with specified dimensions, number of layers, and dropout rate.

        Args:
            embedding_dimension (int): The dimensionality of the input embeddings.
            number_heads (int): The number of attention heads in the multi-head attention mechanism.
            feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
            number_layers (int): The number of encoder and decoder layers in the Transformer.
            max_sequence_length (int): The maximum length of the input and target sequences.
            dropout_rate (float): The dropout rate applied in the encoder and decoder layers. Default is 0.1.
        """
        super(Transformer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_layers = number_layers
        self.max_sequence_length = max_sequence_length

        # Positional encoding layer to add positional information to the embeddings
        self.positional_encoding = PositionalEncoding(max_sequence_length, embedding_dimension)

        # List of encoder layers, each being an instance of TransformerEncoder
        self.encoder_layers = [TransformerEncoder(embedding_dimension, number_heads, feedforward_dimension,
                                                  dropout_rate) for _ in range(number_layers)]

        # List of decoder layers, each being an instance of TransformerDecoder
        self.decoder_layers = [TransformerDecoder(embedding_dimension, number_heads, feedforward_dimension,
                                                  dropout_rate) for _ in range(number_layers)]

        # Final dense layer that projects the decoder output to the embedding dimension
        self.final_layer = Dense(embedding_dimension)

    def call(self, inputs, targets, training):
        """
        Performs the forward pass of the Transformer model, including encoding the inputs,
        decoding the targets, and applying the final linear transformation.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, input_sequence_length, embedding_dimension).
            targets (tf.Tensor): The target tensor of shape (batch_size, target_sequence_length, embedding_dimension).
            training (bool): Whether the layer is in training mode (applies dropout) or inference mode.

        Returns:
            tf.Tensor: The output tensor of shape (batch_size, target_sequence_length, embedding_dimension).
        """
        # Apply positional encoding to the input sequence
        model_flow = self.positional_encoding(inputs)

        # Pass the encoded input through each of the encoder layers
        for encoder_layer in self.encoder_layers:
            model_flow = encoder_layer(model_flow, training)

        # Apply positional encoding to the target sequence
        y = self.positional_encoding(targets)

        # Pass the encoded target and encoder output through each of the decoder layers
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, model_flow, training)

        # Apply the final dense layer to the decoder output to project to the desired embedding dimension
        output = self.final_layer(y)

        return output
