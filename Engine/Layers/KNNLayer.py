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

except ImportError as error:
    print(error)
    sys.exit(-1)


class KNNLayer(Layer):
    """
    Custom TensorFlow layer that performs K-Nearest Neighbors (KNN) clustering on input data.
    This layer computes the indices of the closest K reference points (centroids) to each input sample.

    The KNN layer can be useful in various tasks such as classification, clustering, and anomaly detection
    by finding the closest clusters to the input data based on a distance metric.

    Reference:
        This layer is inspired by the K-Nearest Neighbors algorithm as described in the paper:
        "K-Nearest Neighbors" (Cover and Hart, 1967). The algorithm assigns class labels to input data points
        by considering the class labels of their nearest neighbors.

    Args:
        number_clusters (int): The number of nearest clusters to find for each input. Default is 5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        reference_points (tf.Variable): A variable representing the reference points (or centroids)
            used for distance calculation in the KNN algorithm. These reference points are randomly initialized
            and are not trainable.
        clusters (int): The number of clusters to return for each input. This corresponds to the number of
            nearest neighbors to find.

    Example:
        >>> # Create a simple KNNLayer with 5 clusters
        ...   knn_layer = KNNLayer(number_clusters=5)
        ...   # Sample input tensor (batch_size=3, input_dim=4)
        ...   inputs = tf.random.normal((3, 4))
        ...   # Build the KNNLayer (must be done before calling it)
        ...   knn_layer.build(inputs.shape)
        ...   # Get the indices of the nearest clusters
        ...   indices = knn_layer(inputs)
        >>>   print(indices)
    """

    def __init__(self, number_clusters=5, **kwargs):
        """
        Initializes the KNNLayer with the specified number of clusters.

        Args:
            number_clusters (int): Number of nearest clusters to compute. Default is 5.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(KNNLayer, self).__init__(**kwargs)
        self.reference_points = None
        self.clusters = number_clusters

    def call(self, inputs):
        """
        Performs the forward computation of the layer. Computes the distances between the input
        data and the reference points, and returns the indices of the closest clusters.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            tf.Tensor: A tensor of shape (batch_size, number_clusters) containing the indices
            of the nearest clusters for each input.
        """
        # Expand dimensions of inputs and reference points for broadcasting during subtraction
        inputs_expanded = tensorflow.expand_dims(inputs, axis=1)
        ref_expanded = tensorflow.expand_dims(self.reference_points, axis=0)

        # Calculate L2 distances between inputs and reference points
        distances = tensorflow.norm(inputs_expanded - ref_expanded, axis=-1)

        # Get the indices of the top-k nearest clusters
        _, indices = tensorflow.math.top_k(-distances, k=self.clusters, sorted=False)

        return indices

    def build(self, input_shape):
        """
        Creates and initializes the reference points (or centroids) that the input data will be
        compared against during the forward pass.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        self.reference_points = self.add_weight(
            shape=(100, input_shape[-1]),  # 100 reference points, each with the same dimension as the input
            initializer='random_normal',  # Initialize reference points with a normal distribution
            trainable=False,  # Reference points are not trainable, meaning they won't be updated during backpropagation
            name='reference_points'
        )

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            tuple: The shape of the output tensor, which is (batch_size, number_clusters).
        """
        return (input_shape[0], self.clusters)
