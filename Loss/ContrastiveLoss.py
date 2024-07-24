import tensorflow
from tensorflow.keras.losses import Loss


class ContrastiveLoss(Loss):
    """
    Custom implementation of Contrastive Loss for training models with the
    contrastive loss function, which is commonly used in tasks involving
    similarity learning.

    Attributes:
        margin (float): The margin value for the loss function.
    """

    def __init__(self, margin=1.0, **kwargs):
        """
        Initializes the ContrastiveLoss class with a specified margin.

        Args:
            margin (float): The margin value for the contrastive loss. Default is 1.0.
            **kwargs: Additional keyword arguments passed to the base Loss class.
        """
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_predicted):
        """
        Computes the contrastive loss.

        Args:
            y_true (tf.Tensor): Tensor of true labels with shape (batch_size,).
            y_predicted (tf.Tensor): Tensor of predicted embeddings with shape (2, batch_size, embedding_dim).

        Returns:
            tf.Tensor: The computed contrastive loss.
        """
        y_true = tensorflow.cast(y_true, tensorflow.float32)
        y_predicted = tensorflow.cast(y_predicted, tensorflow.float32)

        # Calculate the Euclidean distance between the two sets of embeddings
        distance = tensorflow.reduce_sum(tensorflow.square(y_predicted[0] - y_predicted[1]), axis=1)

        # Ensure the distance is non-zero to avoid division by zero errors
        distance = tensorflow.maximum(distance, 1e-10)

        # Compute the square root of the distance
        sqrt_distance = tensorflow.sqrt(distance)

        # Calculate the margin term for the contrastive loss
        margin_term = tensorflow.maximum(0.0, self.margin - sqrt_distance)

        # Compute the final contrastive loss
        contrastive_loss = tensorflow.reduce_mean(y_true * distance + (1 - y_true) * tensorflow.square(margin_term))

        return contrastive_loss
