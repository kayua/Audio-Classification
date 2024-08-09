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
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class KNNLayer(Layer):
    """
    Custom TensorFlow layer that performs K-Nearest Neighbors (KNN) clustering on input data.

    Args:
        number_clusters (int): The number of nearest clusters to find for each input. Default is 4.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        reference_points (tf.Variable): A variable representing reference points (or centroids)
            used for distance calculation in the KNN algorithm.
        clusters (int): The number of clusters to return for each input.
    """

    def __init__(self, number_clusters=5, **kwargs):
        """
        Initializes the KNNLayer with the specified number of clusters.

        Args:
            number_clusters (int): Number of nearest clusters to compute.
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


class QuantizationLayer(Layer):
    """
    Custom TensorFlow layer that performs vector quantization on the input data.
    The layer finds the nearest clusters to each input vector and replaces the
    input with the average of the closest reference points (centroids).

    Args:
        number_clusters (int): The number of nearest clusters to consider during quantization. Default is 5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        built (bool): Indicates whether the layer has been built.
        reference_points (tf.Variable): A variable representing the reference points (centroids) used
            in the KNN algorithm for quantization.
        knn_layer (KNNLayer): An instance of the KNNLayer that is used to find the nearest clusters.
        k (int): The number of clusters to use for quantization.
    """

    def __init__(self, number_clusters=5, **kwargs):
        """
        Initializes the QuantizationLayer with the specified number of clusters.

        Args:
            number_clusters (int): The number of nearest clusters to compute during quantization.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(QuantizationLayer, self).__init__(**kwargs)
        self.built = None
        self.reference_points = None
        self.knn_layer = None
        self.k = number_clusters

    def call(self, inputs):
        """
        Performs the forward computation of the layer. Finds the nearest clusters for each input vector,
        retrieves the corresponding reference points, and computes the quantized output by averaging
        these points.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            tf.Tensor: A quantized tensor of shape (batch_size, input_dim), where each input vector
            is replaced by the average of its nearest reference points.
        """
        # Use the KNNLayer to find the indices of the nearest clusters
        indices = self.knn_layer(inputs)

        # Gather the reference points corresponding to the nearest clusters
        reference_points = tensorflow.gather(self.reference_points, indices, batch_dims=1)

        # Compute the quantized output by averaging the reference points
        quantized = tensorflow.reduce_mean(reference_points, axis=1)

        return quantized

    def build(self, input_shape):
        """
        Builds the QuantizationLayer by creating the KNNLayer and initializing the reference points.
        The reference points are shared between the QuantizationLayer and the KNNLayer.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        # Initialize the KNNLayer with the specified number of clusters
        self.knn_layer = KNNLayer(number_clusters=self.k)

        # Create and initialize the reference points (centroids)
        self.reference_points = self.knn_layer.add_weight(
            shape=(100, input_shape[-1]),  # 100 reference points, each with the same dimension as the input
            initializer='random_normal',  # Initialize reference points with a normal distribution
            trainable=False,  # Reference points are not trainable
            name='reference_points'
        )

        # Build the KNNLayer with the input shape
        self.knn_layer.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer, which is the same as the input shape.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            tuple: The shape of the output tensor, which is the same as the input shape.
        """
        return input_shape