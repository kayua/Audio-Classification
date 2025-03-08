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
    import numpy

    from tensorflow.python.keras.layers import Dense

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
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