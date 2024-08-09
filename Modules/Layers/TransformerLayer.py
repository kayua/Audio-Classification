import tensorflow
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, Dropout, LayerNormalization


class PositionalEncoding(Layer):
    def __init__(self, max_sequence_length, embedding_dimension):
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = None
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        self.positional_encodings = self._get_positional_encodings(self.max_sequence_length, self.embedding_dimension)

    @staticmethod
    def _get_positional_encodings(max_seq_length, embedding_dimension):
        positional_array = tensorflow.range(max_seq_length, dtype=tensorflow.float32)[:, tensorflow.newaxis]
        index = tensorflow.range(embedding_dimension, dtype=tensorflow.float32)[tensorflow.newaxis, :]
        angles = positional_array / tensorflow.pow(10000.0, (2 * (index // 2)) / tensorflow.cast(embedding_dimension,
                                                                                                 tensorflow.float32))
        angles_sin = tensorflow.math.sin(angles[:, 0::2])
        angles_cos = tensorflow.math.cos(angles[:, 1::2])
        positional_encodings = tensorflow.concat([angles_sin, angles_cos], axis=-1)

        return positional_encodings[tensorflow.newaxis, ...]

    def call(self, x):
        sequence_length = tensorflow.shape(x)[1]
        positional_encodings = self.positional_encodings[:, :sequence_length, :]
        return x + positional_encodings


class TransformerEncoder(Layer):
    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.feedforward_dimension = feedforward_dimension
        self.dropout_rate = dropout_rate

        self.mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)
        self.feedforward_layer = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),
            Dense(embedding_dimension)
        ])

        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.first_dropout = Dropout(dropout_rate)
        self.second_dropout = Dropout(dropout_rate)

    def call(self, x, training):
        attention_output = self.mult_head_attention(x, x)
        attention_output = self.first_dropout(attention_output, training=training)
        output_normalization = self.first_layer_normalization(x + attention_output)

        feedforward_output = self.feedforward_layer(output_normalization)
        feedforward_output = self.second_dropout(feedforward_output, training=training)
        output_second_normalization = self.second_layer_normalization(output_normalization + feedforward_output)

        return output_second_normalization


class TransformerDecoder(Layer):
    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.ff_dim = feedforward_dimension
        self.dropout_rate = dropout_rate

        self.first_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)
        self.second_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)
        self.feedforward = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),
            Dense(embedding_dimension)
        ])

        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.third_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, encoder_output, training):
        attn1 = self.first_mult_head_attention(x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.first_layer_normalization(x + attn1)

        attn2 = self.second_mult_head_attention(out1, encoder_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.second_layer_normalization(out1 + attn2)

        ffn_output = self.feedforward(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.third_layer_normalization(out2 + ffn_output)

        return out3


class Transformer(Layer):
    def __init__(self, embedding_dimension, num_heads, ff_dim, num_layers, max_seq_length, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.d_model = embedding_dimension
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        self.positional_encoding = PositionalEncoding(max_seq_length, embedding_dimension)
        self.encoder_layers = [TransformerEncoder(embedding_dimension, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoder(embedding_dimension, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.final_layer = Dense(embedding_dimension)  # Modifique conforme a sua necessidade

    def call(self, inputs, targets, training):
        x = self.positional_encoding(inputs)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)

        y = self.positional_encoding(targets)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, x, training)

        output = self.final_layer(y)
        return output