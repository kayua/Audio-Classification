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

except ImportError as error:
    print(error)
    sys.exit(-1)

class MaskCreator:
    """
    A utility class for creating masks for sequence data.

    Methods
    -------
    create(seq_len: int) -> tf.Tensor
        Creates a lower triangular mask (causal mask) for sequences.
    """

    @staticmethod
    def create_mask(seq_len: int) -> tensorflow.Tensor:
        """
        Creates a lower triangular mask for causal attention.

        This mask is commonly used in transformer models to prevent a position
        from attending to future positions.

        Parameters
        ----------
        seq_len : int
            The length of the sequence.

        Returns
        -------
        mask : tf.Tensor
            A tensor of shape (seq_len, seq_len) with ones in the lower triangle and zeros elsewhere.
        """
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive.")

        mask = tensorflow.linalg.band_part(tensorflow.ones((seq_len, seq_len)), -1, 0)
        return mask