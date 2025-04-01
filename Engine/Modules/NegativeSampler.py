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


import random
import tensorflow

class NegativeSampler:
    """
    The NegativeSampler class is designed to sample negative examples for a given batch of targets.
    Negative samples are drawn by excluding self-sampling, ensuring that the target is not sampled as its own negative.

    The class operates on a batch of data and samples K negative examples for each target in the batch.

    Attributes:
        K (int): The number of negative samples per target.

    Methods:
        negative_sampler: Samples negative examples for a given batch of targets.
        _create_negative_sample_candidate_indices: Creates candidate indices for negative sampling, excluding self-sampling.
        _generate_negative_sample_indices: Generates the indices for negative samples ensuring no self-sampling.

    Example:
        >>> python
        ...     # Initialize the NegativeSampler with the number of negative samples per target
        ...     sampler = NegativeSampler(K=5)
        ...
        ...     # Label tensor of shape (N, D) where N is the number of targets and D is the feature dimension
        ...     labels = tf.random.normal([100, 64])  # Example tensor with 100 targets and 64 features
        ...
        ...     # List of the number of targets per batch
        ...     num_targets_per_batch = [50, 50]  # Two batches with 50 targets each
        ...
        ...     # Generate negative samples
        ...     negative_samples = sampler.negative_sampler(labels, num_targets_per_batch)
        ...
        ...     # Negative samples shape will be (100, 5, 64)
        ...     print(negative_samples.shape)  # Output: (100, 5, 64)
        >>>
    """

    def __init__(self, K):
        """
        Initializes the NegativeSampler instance with the number of negative samples per target.

        Args:
            K (int): The number of negative samples per target.
        """
        self.K = K

    def negative_sampler(self, label: tensorflow.Tensor, number_targets_per_batch: list[int]) -> tensorflow.Tensor:
        """
        Samples negative examples for a given batch of targets by selecting negative samples for each target.
        This method iterates over each batch of targets, creates candidate indices for negative sampling,
        ensures no self-sampling, and collects the negative samples.

        Args:
            label (tf.Tensor): A tensor of shape (N, D), where N is the total number of targets and D is the feature dimension.
            number_targets_per_batch (list[int]): A list of integers representing the number of targets per batch.

        Returns:
            tf.Tensor: A tensor of shape (N, K, D) containing the negative samples, where K is the number of negative samples per target.

        Example:
            ```python
            negative_samples = sampler.negative_sampler(labels, [50, 50])
            ```
        """
        negative_samples = []
        start_idx = 0

        for number_targets in number_targets_per_batch:
            negative_sample_candidate_indices = self._create_negative_sample_candidate_indices(number_targets, start_idx)
            where_negative_sample = self._generate_negative_sample_indices(number_targets)
            negative_sample_indices = tensorflow.gather(negative_sample_candidate_indices, where_negative_sample[1])
            negative_samples.append(tensorflow.gather(label, negative_sample_indices))

            start_idx += number_targets

        # Concatenate all samples and reshape to the desired shape
        negative_samples = tensorflow.concat(negative_samples, axis=0)
        negative_samples = tensorflow.reshape(negative_samples, (label.shape[0], self.K, -1))

        return negative_samples

    @staticmethod
    def _create_negative_sample_candidate_indices(number_targets: int, start_idx: int) -> tensorflow.Tensor:
        """
        Creates candidate indices for negative sampling by excluding self-sampling (diagonal).

        Args:
            number_targets (int): The number of targets in the current batch.
            start_idx (int): The starting index for the current batch.

        Returns:
            tf.Tensor: A tensor containing the candidate indices for negative samples, excluding self-sampling.

        Example:
            ```python
            negative_sample_candidate_indices = self._create_negative_sample_candidate_indices(50, 0)
            ```
        """
        # Create indices for candidates
        negative_sample_candidate_indices = tensorflow.repeat(
            tensorflow.range(number_targets), number_targets
        )

        # Create a diagonal matrix and remove the diagonal (self-sampling)
        diagonal = tensorflow.eye(number_targets, dtype=tensorflow.bool)
        negative_sample_candidate_indices = tensorflow.boolean_mask(negative_sample_candidate_indices, ~diagonal)
        negative_sample_candidate_indices += start_idx

        return negative_sample_candidate_indices

    def _generate_negative_sample_indices(self, number_targets: int) -> tuple:
        """
        Generates the indices for negative samples by ensuring no self-sampling occurs.

        Args:
            number_targets (int): The number of targets in the current batch.

        Returns:
            tuple: A tuple containing the row and column indices for the negative samples.

        Example:
            ```python
            where_negative_sample = self._generate_negative_sample_indices(50)
            ```
        """
        # Sample negative indices, ensuring no self-sampling (using random.sample for the row)
        where_negative_sample = (
            tensorflow.tile(tensorflow.range(number_targets)[:, None], [1, self.K]),
            tensorflow.stack([random.sample(list(range(number_targets - 1)), k=self.K) for _ in range(number_targets)],
                             axis=0).flatten())

        return where_negative_sample
