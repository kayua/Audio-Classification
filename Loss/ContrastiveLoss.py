import tensorflow as tf
from tensorflow.keras.losses import Loss


class ContrastiveLoss(Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_predicted):
        y_true = tf.cast(y_true, tf.float32)
        y_predicted = tf.cast(y_predicted, tf.float32)

        d = tf.reduce_sum(tf.square(y_predicted[0] - y_predicted[1]), axis=1)

        d = tf.maximum(d, 1e-10)

        sqrt_d = tf.sqrt(d)

        margin_term = tf.maximum(0.0, self.margin - sqrt_d)

        loss = tf.reduce_mean(y_true * d + (1 - y_true) * tf.square(margin_term))

        return loss