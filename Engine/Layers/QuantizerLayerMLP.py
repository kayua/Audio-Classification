#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
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

    from tensorflow.keras.layers import Layer
    from Engine.Layers.KNNLayer import KNNLayer

except ImportError as error:
    print(error)
    sys.exit(-1)

class QuantizationLayer(Layer):
    """
    Custom TensorFlow layer that performs vector quantization on the input data.
    The layer finds the nearest clusters (centroids) to each input vector and replaces
    the input with the average of the closest reference points (centroids). This process
    is commonly used in tasks like data compression, image segmentation, and feature learning.

    Vector Quantization (VQ) involves mapping input vectors to a finite set of representative
    centroids (also known as codebook vectors or cluster centroids). The input is then replaced
    by the centroid closest to it, based on a distance metric (typically Euclidean).

    Reference:
        The concept of vector quantization is widely used in signal processing and data compression.
        "Vector Quantization and Signal Compression" by R. M. Gray and D. L. Neuhoff, 1998.
        This technique is also closely related to K-Means clustering, where the centroids serve
        as the quantized representations of the data.

    Args:
        number_clusters (int): The number of nearest clusters (centroids) to consider during quantization.
                               This is the number of centroids to be considered for each input.
                               Default is 5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        built_point (bool): Indicates whether the layer has been built.
        reference_points (tf.Variable): A variable representing the reference points (centroids) used
            in the quantization process. These centroids are fixed (non-trainable) and are shared
            between the KNNLayer and the QuantizationLayer.
        knn_layer (KNNLayer): An instance of the KNNLayer that is used to find the nearest clusters (centroids).
        number_k_clusters (int): The number of clusters to use for quantization. It determines how many centroids
                 to consider for each input vector.

    Example:
        >>> # Create a QuantizationLayer with 5 clusters
        ...     quant_layer = QuantizationLayer(number_clusters=5)
        ...     # Sample input tensor (batch_size=3, input_dim=4)
        ...     inputs = tf.random.normal((3, 4))
        ...     # Build the QuantizationLayer (must be done before calling it)
        ...     quant_layer.build(inputs.shape)
        ...     # Get the quantized output
        ...     quantized_output = quant_layer(inputs)
        ...     print(quantized_output)
        >>>
    """


    def __init__(self, number_clusters=5, **kwargs):
        """
        Initializes the QuantizationLayer with the specified number of clusters.

        Args:
            number_clusters (int): The number of nearest clusters (centroids) to compute during quantization.
                                    Default is 5.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(QuantizationLayer, self).__init__(**kwargs)
        self.built_point = None
        self.reference_points = None
        self.knn_layer = None
        self.number_k_clusters = number_clusters

    def call(self, inputs):
        """
        Performs the forward computation of the layer. For each input vector, the layer finds the
        nearest clusters (centroids), retrieves the corresponding reference points (centroids), and
        computes the quantized output by averaging these points.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, input_dim). Each row represents
                                 a data point or feature vector that will be quantized.

        Returns:
            tf.Tensor: A quantized tensor of shape (batch_size, input_dim), where each input vector
                       is replaced by the average of its nearest reference points (centroids).
        """
        # Use the KNNLayer to find the indices of the nearest clusters
        indices = self.knn_layer(inputs)

        # Gather the reference points corresponding to the nearest clusters
        reference_points = tensorflow.gather(self.reference_points, indices, batch_dims=1)

        # Compute the quantized output by averaging the reference points
        quantized_weights = tensorflow.reduce_mean(reference_points, axis=1)

        return quantized_weights

    def build(self, input_shape):
        """
        Builds the QuantizationLayer by creating the KNNLayer and initializing the reference points.
        The reference points are shared between the QuantizationLayer and the KNNLayer.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        # Initialize the KNNLayer with the specified number of clusters
        self.knn_layer = KNNLayer(number_clusters=self.number_k_clusters)

        # Create and initialize the reference points (centroids)
        self.reference_points = self.knn_layer.add_weight(
            shape=(100, input_shape[-1]),  # 100 reference points, each with the same dimension as the input
            initializer='random_normal',  # Initialize reference points with a normal distribution
            trainable=False,  # Reference points are not trainable
            name='reference_points'
        )

        # Build the KNNLayer with the input shape
        self.knn_layer.build(input_shape)
        self.built_point = True

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer, which is the same as the input shape.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            tuple: The shape of the output tensor, which is the same as the input shape.
        """
        return input_shape