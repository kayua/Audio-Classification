class TransformerEncoder(Layer):
    """
    Custom TensorFlow layer implementing a Transformer encoder block.
    The Transformer encoder is composed of multi-head self-attention,
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
        mult_head_attention (MultiHeadAttention): The multi-head self-attention layer.
        feedforward_layer (Sequential): A sequential model consisting of two Dense layers.
        first_layer_normalization (LayerNormalization): The first layer normalization applied after attention.
        second_layer_normalization (LayerNormalization): The second layer normalization applied after the feedforward layer.
        first_dropout (Dropout): The dropout layer applied after the attention output.
        second_dropout (Dropout): The dropout layer applied after the feedforward output.
    """

    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        """
        Initializes the TransformerEncoder with specified dimensions for embeddings, attention heads,
        and feedforward network, along with the dropout rate.

        Args:
            embedding_dimension (int): The dimensionality of the input embeddings.
            number_heads (int): The number of attention heads in the multi-head attention mechanism.
            feedforward_dimension (int): The dimensionality of the feedforward network hidden layer.
            dropout_rate (float): The dropout rate applied after attention and feedforward layers. Default is 0.1.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.feedforward_dimension = feedforward_dimension
        self.dropout_rate = dropout_rate

        # Multi-head self-attention layer
        self.mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Feedforward network consisting of two Dense layers
        self.feedforward_layer = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),  # First layer with ReLU activation
            Dense(embedding_dimension)  # Output layer projecting back to embedding dimension
        ])

        # First layer normalization, applied after the attention layer
        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Second layer normalization, applied after the feedforward network
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Dropout layers applied after attention and feedforward network
        self.first_dropout = Dropout(dropout_rate)
        self.second_dropout = Dropout(dropout_rate)

    def call(self, x, training):
        """
        Performs the forward pass of the Transformer encoder. Applies multi-head self-attention
        followed by layer normalization, then a feedforward network followed by another layer normalization.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dimension).
            training (bool): Whether the layer is in training mode (applies dropout) or inference mode.

        Returns:
            tf.Tensor: The output tensor of the same shape as the input, after processing by the Transformer encoder block.
        """
        # Apply multi-head self-attention to the input
        attention_output = self.mult_head_attention(x, x)

        # Apply dropout to the attention output if in training mode
        attention_output = self.first_dropout(attention_output, training=training)

        # Add the attention output to the input and apply the first layer normalization
        output_normalization = self.first_layer_normalization(x + attention_output)

        # Pass the normalized output through the feedforward network
        feedforward_output = self.feedforward_layer(output_normalization)

        # Apply dropout to the feedforward output if in training mode
        feedforward_output = self.second_dropout(feedforward_output, training=training)

        # Add the feedforward output to the normalized output and apply the second layer normalization
        output_second_normalization = self.second_layer_normalization(output_normalization + feedforward_output)

        return output_second_normalization
