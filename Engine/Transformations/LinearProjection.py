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
    import numpy

    from tensorflow.python.keras.layers import Dense

except ImportError as error:
    print(error)
    sys.exit(-1)

class LinearProjection:
    """
    This class implements a linear projection layer that flattens input patches
    and applies a Dense layer to project them into a target dimension.

    Attributes:
        projection_dimension (int): The target dimensionality of the projection.
        dense_layer (Dense): The dense layer used to perform the projection.

    Methods:
        project(tensor_patches: np.ndarray) -> np.ndarray
            Projects the input tensor patches to the specified dimension.
    """

    def __init__(self, projection_dimension: int):
        """
        Initializes the LinearProjection instance with the desired projection dimension.

        Args:
            projection_dimension (int): The target dimension for the projection.
        """
        self.projection_dimension = projection_dimension
        self.dense_layer = Dense(projection_dimension)

    def linear_projection(self, tensor_patches: numpy.ndarray) -> numpy.ndarray:
        """
        Projects the input tensor patches into the configured projection dimension.

        Args:
            tensor_patches (np.ndarray): Input array of tensor patches with shape
                                          (batch_size, patch_count, patch_size).

        Returns:
            np.ndarray: The projected tensor with shape (batch_size, projection_dimension).
        """
        if not isinstance(tensor_patches, numpy.ndarray):
            raise ValueError("Input tensor_patches must be a numpy.ndarray")

        # Flatten each patch (batch_size, patch_count * patch_size)
        patches_flat = tensor_patches.reshape(tensor_patches.shape[0], -1)

        # Apply the Dense layer to project to the desired dimension
        return self.dense_layer(patches_flat)