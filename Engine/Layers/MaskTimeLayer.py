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

    from keras import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)

class TimeMasking(Layer):
    """
    TensorFlow layer for applying time masking to sequential data.

    This layer masks a percentage of time steps in the input sequence by setting their values to zero.

    Attributes:
        mask_time_probabilities (float): Probability of masking time steps.
        number_mask_time_steps (int): Number of consecutive time steps to mask.

    Reference:
        - Inspired by SpecAugment: https://arxiv.org/abs/1904.08779

    Example:
        >>> # Create a simple KNNLayer with 5 clusters
        ...   time_masking = TimeMasking(mask_time_prob=0.2, number_mask_time_steps=5)
        ...   hidden_states = tensorflow.random.normal([2, 10, 32])  # (batch_size=2, sequence_length=10, feature_dim=32)
        ...   lengths = tensorflow.constant([10, 8])  # Valid sequence lengths
        ...   masked_states, mask = time_masking(hidden_states, lengths)
        ...   print(masked_states.shape)  # (2, 10, 32)
        >>>   print(mask.shape)  # (2, 10)
    """

    def __init__(self, mask_time_prob: float, number_mask_time_steps: int, **kwargs):
        """
        Initializes the TimeMasking layer.

        Args:
            mask_time_prob (float): Probability of masking time steps.
            number_mask_time_steps (int): Number of consecutive time steps to mask.
        """
        super(TimeMasking, self).__init__(**kwargs)
        self.mask_time_probabilities = mask_time_prob
        self.number_mask_time_steps = number_mask_time_steps

    def call(self, hidden_states, lengths):
        """
        Applies time masking to the input hidden states.

        Args:
            hidden_states (tf.Tensor): Input tensor of shape `(B, L, D)` where B is batch size,
                                       L is sequence length, and D is feature dimension.
            lengths (tf.Tensor): Tensor of shape `(B,)` containing the valid sequence lengths for each batch element.

        Returns:
            tuple:
                - Masked hidden states (tf.Tensor of shape `(B, L, D)`).
                - Time mask (tf.Tensor of shape `(B, L)`, dtype=tf.bool).
        """
        batch_size, number_steps, hidden_size = (tensorflow.shape(hidden_states)[0],
                                                 tensorflow.shape(hidden_states)[1],
                                                 tensorflow.shape(hidden_states)[2])

        # Generate mask probabilities per batch
        mask_lengths = tensorflow.cast(
            tensorflow.math.ceil(self.mask_time_probabilities * tensorflow.cast(lengths, tensorflow.float32)),
            tensorflow.int32)

        # Generate random start indices for masking
        random_starts = tensorflow.sort(
            tensorflow.random.uniform(shape=(batch_size, tensorflow.reduce_max(mask_lengths)),
                                      maxval=number_steps, dtype=tensorflow.int32), axis=-1)

        # Expand indices for num_mask_time_steps
        offsets = tensorflow.range(self.number_mask_time_steps)
        expanded_offsets = tensorflow.expand_dims(offsets, 0)  # Shape (1, num_mask_time_steps)
        expanded_random_starts = tensorflow.expand_dims(random_starts, -1)  # Shape (B, K, 1)
        mask_positions = expanded_random_starts + expanded_offsets  # Shape (B, K, num_mask_time_steps)

        # Clip indices to stay within sequence length
        mask_positions = tensorflow.clip_by_value(mask_positions, 0, number_steps - 1)

        # Create boolean mask
        time_mask_indices = tensorflow.reduce_any(tensorflow.one_hot(mask_positions,
                                                                     depth=number_steps, dtype=tensorflow.bool), axis=1)

        # Apply mask to hidden states
        mask_values = tensorflow.zeros_like(hidden_states)
        hidden_states = tensorflow.where(tensorflow.expand_dims(time_mask_indices, -1), mask_values, hidden_states)

        return hidden_states, time_mask_indices