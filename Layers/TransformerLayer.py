import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization, Input
from tensorflow.keras.models import Model


class PositionalEncoding(Layer):
    def __init__(self, max_seq_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model

    def build(self, input_shape):
        self.positional_encodings = self._get_positional_encodings(self.max_seq_length, self.d_model)

    def _get_positional_encodings(self, max_seq_length, d_model):
        pos = tf.range(max_seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angles_sin = tf.math.sin(angles[:, 0::2])
        angles_cos = tf.math.cos(angles[:, 1::2])
        positional_encodings = tf.concat([angles_sin, angles_cos], axis=-1)
        return positional_encodings[tf.newaxis, ...]

    def call(self, x):
        seq_length = tf.shape(x)[1]
        positional_encodings = self.positional_encodings[:, :seq_length, :]
        return x + positional_encodings


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerDecoder(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, encoder_output, training):
        attn1 = self.mha1(x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, encoder_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

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