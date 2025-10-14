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

except ImportError as error:
    print(error)
    sys.exit(-1)

class SpectrogramPatcher:
    """
    This class provides functionality to split spectrograms into non-overlapping patches
    of a fixed size, with optional zero-padding to ensure complete coverage.

    Attributes:
        patch_size (tuple): A tuple (patch_height, patch_width) specifying the size of each patch.

    Methods:
        split_into_patches(spectrogram: np.ndarray) -> np.ndarray
            Splits a 2D spectrogram into patches of the specified size, with padding if needed.
    """

    def __init__(self, patch_size: tuple[int, int]):
        """
        Initializes the SpectrogramPatcher with the desired patch size.

        Args:
            patch_size (tuple): A tuple (patch_height, patch_width) defining the size of each patch.
        """
        self.patch_size = patch_size

    def split_spectrogram_into_patches(self, spectrogram: numpy.ndarray) -> numpy.ndarray:
        """
        Splits a 2D spectrogram into non-overlapping patches of the configured patch size.
        Pads the spectrogram with zeros if its dimensions are not perfectly divisible by the patch size.

        Args:
            spectrogram (np.ndarray): A 2D array representing the spectrogram.

        Returns:
            np.ndarray: A 3D array containing the patches, with shape (num_patches, patch_height, patch_width).
        """
        if not isinstance(spectrogram, numpy.ndarray):
            raise ValueError("The input spectrogram must be a numpy.ndarray.")

        # Calculate the padding needed to make the spectrogram dimensions divisible by the patch size
        padding_height = (self.patch_size[0] - (spectrogram.shape[0] % self.patch_size[0])) % self.patch_size[0]
        padding_width = (self.patch_size[1] - (spectrogram.shape[1] % self.patch_size[1])) % self.patch_size[1]

        # Apply zero padding to the spectrogram
        padded_spectrogram = numpy.pad(spectrogram, ((0, padding_height), (0, padding_width)),
                                       mode='constant', constant_values=0)

        # Calculate the number of patches in each dimension
        number_patches_x = padded_spectrogram.shape[0] // self.patch_size[0]
        number_patches_y = padded_spectrogram.shape[1] // self.patch_size[1]

        patches = []

        # Extract patches by iterating over the padded spectrogram
        for axis_x in range(number_patches_x):
            for axis_y in range(number_patches_y):
                patch = padded_spectrogram[
                        axis_x * self.patch_size[0]:(axis_x + 1) * self.patch_size[0],
                        axis_y * self.patch_size[1]:(axis_y + 1) * self.patch_size[1]
                ]

                patches.append(patch)

        # Convert list of patches to numpy array
        return numpy.array(patches)