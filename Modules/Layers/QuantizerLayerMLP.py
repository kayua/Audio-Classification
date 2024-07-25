import tensorflow as tf
from tensorflow.keras.layers import Layer


class QuantizeLayer(Layer):
    def __init__(self, num_bits=8, **kwargs):
        super(QuantizeLayer, self).__init__(**kwargs)
        self.num_bits = num_bits
        self.scale = 2 ** num_bits - 1

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.0)
        inputs = tf.round(inputs * self.scale) / self.scale
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class KNNLayer(Layer):
    def __init__(self, k=5, **kwargs):
        super(KNNLayer, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_points = tf.shape(self.reference_points)[0]

        inputs_expanded = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
        ref_expanded = tf.expand_dims(self.reference_points, axis=0)  # (1, num_points, features)

        distances = tf.norm(inputs_expanded - ref_expanded, axis=-1)  # (batch_size, num_points)
        _, indices = tf.math.top_k(-distances, k=self.k, sorted=False)  # (batch_size, k)

        return indices

    def build(self, input_shape):
        self.reference_points = self.add_weight(
            shape=(100, input_shape[-1]),  # Exemplo: 100 pontos de referência com a mesma dimensão do input
            initializer='random_normal',
            trainable=False,
            name='reference_points'
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k)  # (batch_size, k)


class QuantizationLayer(Layer):
    def __init__(self, k=5, **kwargs):
        super(QuantizationLayer, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # Usar a camada KNN para obter os índices dos k vizinhos mais próximos
        indices = self.knn_layer(inputs)

        # Obter os pontos de referência correspondentes aos índices
        reference_points = tf.gather(self.reference_points, indices, batch_dims=1)  # (batch_size, k, features)

        # Calcular o ponto de referência médio para cada entrada
        quantized = tf.reduce_mean(reference_points, axis=1)  # (batch_size, features)

        return quantized

    def build(self, input_shape):
        self.knn_layer = KNNLayer(k=self.k)
        self.reference_points = self.knn_layer.add_weight(
            shape=(100, input_shape[-1]),  # Exemplo: 100 pontos de referência com a mesma dimensão do input
            initializer='random_normal',
            trainable=False,
            name='reference_points'
        )
        self.knn_layer.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape
