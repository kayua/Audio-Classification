#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    import sys

    import tensorflow

    from tensorflow.keras.losses import Loss

except ImportError as error:
    print(error)
    sys.exit(-1)



class ContrastiveLoss(Loss):
    """
    Custom implementation of Contrastive Loss for training models with the
    contrastive loss function, commonly used in similarity learning tasks, such as
    metric learning and Siamese networks.

    The contrastive loss function is designed to minimize the distance between similar pairs
    and maximize the distance between dissimilar pairs, where the similarity of a pair is
    typically indicated by a binary label (1 for similar, 0 for dissimilar).

    References:
    -----------
        - Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction by Learning an Invariant Mapping.
          Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
        - Bengio, Y., & LeCun, Y. (2007). Learning Deep Architectures for AI. Foundations and
          Trends in Machine Learning, 2(1), 1-127.

    Mathematical Formula:
    ---------------------
    The contrastive loss for a pair of embeddings y1 and y2 is computed as:

        L = (1/N) * sum_i [ y_i * D^2 + (1 - y_i) * max(0, m - D)^2 ]

    Where:
        - y_i is the binary label indicating whether the pair is similar (y_i = 1) or dissimilar (y_i = 0).
        - D is the Euclidean distance between the two embeddings: D = ||y1 - y2||.
        - m is the margin, a hyperparameter that defines the minimum distance between dissimilar pairs. Default is 1.0.
        - N is the number of pairs in the batch.

    Args:
        margin (float): The margin value for the contrastive loss function. Default is 1.0. This margin is used
                        to separate dissimilar pairs, ensuring they are at least this distance apart.
        **kwargs: Additional keyword arguments passed to the base `Loss` class.

    Attributes:
        margin (float): The margin value for the contrastive loss function.

    Example
    -------
        >>> # Create a ContrastiveLoss object with a margin of 1.0
        ...     contrastive_loss_layer = ContrastiveLoss(margin=1.0)
        ...     # Example tensors for true labels and predicted embeddings
        ...     y_true = tensorflow.constant([1, 0, 1])  # Labels: 1 for similar, 0 for dissimilar
        ...     y_pred = tensorflow.random.normal((2, 3, 128))  # Predicted embeddings of shape (2, batch_size, embedding_dim)
        ...     # Compute the contrastive loss
        ...     loss = contrastive_loss_layer(y_true, y_pred)
        >>>     print(loss)

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
