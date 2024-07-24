import tensorflow as tf
from tensorflow.keras.losses import Loss


class ContrastiveLoss(Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        d = tf.reduce_sum(tf.square(y_pred[0] - y_pred[1]), axis=1)

        loss = tf.reduce_mean(y_true * d + (1 - y_true) * tf.maximum(0.0, self.margin - tf.sqrt(d)) ** 2)
        return loss