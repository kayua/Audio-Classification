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

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
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