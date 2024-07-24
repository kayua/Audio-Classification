import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential


class QuantizeLayer(Layer):
    def __init__(self, num_bits=8, **kwargs):
        super(QuantizeLayer, self).__init__(**kwargs)
        self.num_bits = num_bits
        self.scale = 2 ** num_bits - 1

    def call(self, inputs):
        # Quantize activations
        inputs = tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.0)  # Clamp between 0 and 1
        inputs = tf.round(inputs * self.scale) / self.scale
        return inputs

