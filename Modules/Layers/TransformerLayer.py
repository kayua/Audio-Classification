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
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class PositionalEncoding(Layer):
    """
    Custom TensorFlow layer that adds positional encoding to input embeddings.
    This layer is typically used in Transformer models to inject information
    about the position of each token in a sequence.

    Args:
        max_sequence_length (int): The maximum length of the input sequences.
        embedding_dimension (int): The dimensionality of the input embeddings.

    Attributes:
        positional_encodings (tf.Tensor): A tensor containing the precomputed positional encodings.
        max_sequence_length (int): The maximum sequence length for which positional encodings are computed.
        embedding_dimension (int): The dimensionality of the embeddings.
    """

    def __init__(self, max_sequence_length, embedding_dimension):
        """
        Initializes the PositionalEncoding layer with the specified maximum sequence length
        and embedding dimension.

        Args:
            max_sequence_length (int): The maximum length of input sequences.
            embedding_dimension (int): The dimensionality of input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = None
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        """
        Builds the PositionalEncoding layer by computing the positional encodings.
        This method is called automatically before the layer is used for the first time.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        # Precompute the positional encodings for the maximum sequence length and embedding dimension
        self.positional_encodings = self._get_positional_encodings(self.max_sequence_length, self.embedding_dimension)

    @staticmethod
    def _get_positional_encodings(max_seq_length, embedding_dimension):
        """
        Computes the positional encodings for the given sequence length and embedding dimension.
        The encodings are based on sine and cosine functions of different frequencies.

        Args:
            max_seq_length (int): The maximum length of input sequences.
            embedding_dimension (int): The dimensionality of input embeddings.

        Returns:
            tf.Tensor: A tensor of shape (1, max_seq_length, embedding_dimension) containing
            the positional encodings.
        """
        # Create a range of positions for the sequence
        positional_array = tensorflow.range(max_seq_length, dtype=tensorflow.float32)[:, tensorflow.newaxis]

        # Create an index array for the embedding dimensions
        index = tensorflow.range(embedding_dimension, dtype=tensorflow.float32)[tensorflow.newaxis, :]

        # Compute the angles for the sine and cosine functions
        angles = positional_array / tensorflow.pow(10000.0, (2 * (index // 2)) / tensorflow.cast(embedding_dimension,
                                                                                                 tensorflow.float32))

        # Apply sine to even indices in the embedding dimensions
        angles_sin = tensorflow.math.sin(angles[:, 0::2])

        # Apply cosine to odd indices in the embedding dimensions
        angles_cos = tensorflow.math.cos(angles[:, 1::2])

        # Concatenate the sine and cosine encodings along the last axis
        positional_encodings = tensorflow.concat([angles_sin, angles_cos], axis=-1)

        # Add a batch dimension to the positional encodings
        return positional_encodings[tensorflow.newaxis, ...]

    def call(self, x):
        """
        Adds the precomputed positional encodings to the input embeddings.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            tf.Tensor: A tensor of the same shape as the input, with positional encodings added
            to the embeddings.
        """
        # Get the sequence length of the input tensor
        sequence_length = tensorflow.shape(x)[1]

        # Retrieve the positional encodings corresponding to the current sequence length
        positional_encodings = self.positional_encodings[:, :sequence_length, :]

        # Add the positional encodings to the input embeddings
        return x + positional_encodings


class TransformerDecoder(Layer):
    """
    Custom TensorFlow layer implementing a Transformer decoder block.
    The Transformer decoder is composed of two multi-head attention layers,
    followed by a feedforward neural network, with layer normalization
    and dropout applied after each step.

    Args:
        embedding_dimension (int): The dimensionality of the input embeddings.
        number_heads (int): The number of attention heads in the multi-head attention mechanism.
        feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
        dropout_rate (float): The dropout rate applied after attention and feedforward layers. Default is 0.1.

    Attributes:
        embedding_dimension (int): The dimensionality of the input embeddings.
        number_heads (int): The number of attention heads in the multi-head attention mechanism.
        feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
        dropout_rate (float): The dropout rate applied after attention and feedforward layers.
        first_mult_head_attention (MultiHeadAttention): The first multi-head self-attention layer.
        second_mult_head_attention (MultiHeadAttention): The second multi-head attention layer for attending to encoder output.
        feedforward (Sequential): A sequential model consisting of two Dense layers.
        first_layer_normalization (LayerNormalization): The first layer normalization applied after the first attention layer.
        second_layer_normalization (LayerNormalization): The second layer normalization applied after the second attention layer.
        third_layer_normalization (LayerNormalization): The third layer normalization applied after the feedforward layer.
        first_dropout (Dropout): The dropout layer applied after the first attention output.
        second_dropout (Dropout): The dropout layer applied after the second attention output.
        third_dropout (Dropout): The dropout layer applied after the feedforward output.
    """

    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        """
        Initializes the TransformerDecoder with specified dimensions for embeddings, attention heads,
        and feedforward network, along with the dropout rate.

        Args:
            embedding_dimension (int): The dimensionality of the input embeddings.
            number_heads (int): The number of attention heads in the multi-head attention mechanism.
            feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
            dropout_rate (float): The dropout rate applied after attention and feedforward layers. Default is 0.1.
        """
        super(TransformerDecoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.feedforward_dimension = feedforward_dimension
        self.dropout_rate = dropout_rate

        # First multi-head self-attention layer for the decoder input
        self.first_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Second multi-head attention layer for attending to the encoder output
        self.second_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Feedforward network consisting of two Dense layers
        self.feedforward = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),  # First layer with ReLU activation
            Dense(embedding_dimension)  # Output layer projecting back to embedding dimension
        ])

        # Layer normalization applied after each step
        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.third_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Dropout layers applied after each step for regularization
        self.first_dropout = Dropout(dropout_rate)
        self.second_dropout = Dropout(dropout_rate)
        self.third_dropout = Dropout(dropout_rate)

    def call(self, x, encoder_output, training):
        """
        Performs the forward pass of the Transformer decoder. It applies the first multi-head
        self-attention on the decoder input, then the second multi-head attention on the
        encoder output, followed by a feedforward network.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dimension) to the decoder.
            encoder_output (tf.Tensor): The output tensor from the encoder, used as context in the decoder's attention mechanism.
            training (bool): Whether the layer is in training mode (applies dropout) or inference mode.

        Returns:
            tf.Tensor: The output tensor of the same shape as the input, after processing by the Transformer decoder block.
        """
        # Apply the first multi-head self-attention to the input
        first_attention_output = self.first_mult_head_attention(x, x)

        # Apply dropout to the first attention output if in training mode
        first_attention_output = self.first_dropout(first_attention_output, training=training)

        # Add the first attention output to the input and apply the first layer normalization
        output_normalization = self.first_layer_normalization(x + first_attention_output)

        # Apply the second multi-head attention using the encoder output as context
        second_attention_output = self.second_mult_head_attention(output_normalization, encoder_output)

        # Apply dropout to the second attention output if in training mode
        second_attention_output = self.second_dropout(second_attention_output, training=training)

        # Add the second attention output to the normalized output and apply the second layer normalization
        second_output_normalization = self.second_layer_normalization(output_normalization + second_attention_output)

        # Pass the normalized output through the feedforward network
        feedforward_output = self.feedforward(second_output_normalization)

        # Apply dropout to the feedforward output if in training mode
        feedforward_output = self.third_dropout(feedforward_output, training=training)

        # Add the feedforward output to the normalized output and apply the third layer normalization
        third_output_normalization = self.third_layer_normalization(second_output_normalization + feedforward_output)

        return third_output_normalization


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
