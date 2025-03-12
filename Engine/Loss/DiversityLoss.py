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
    from tensorflow.keras.losses import Loss

except ImportError as error:
    print(error)
    sys.exit(-1)

class DiversityLoss(Loss):
    """
    Custom loss function to compute the diversity loss, which encourages diversity in the
    predictions by penalizing low entropy in the predicted distribution.

    The diversity loss is defined as:

        L_diversity = (1 / (G * V)) * sum(sum(p * log(p)))

    Where:
        - p is the predicted perplexity for each group and vocabulary dimension.
        - G is the number of groups.
        - V is the vocabulary size.

    The loss is based on the idea of encouraging higher entropy in the predictions, leading
    to more diverse outputs.

    Reference:
    ---------
        Rombach, Robin, et al. "High-Resolution Image Synthesis with Latent Diffusion Models."
        arXiv preprint arXiv:2112.10752 (2021). https://arxiv.org/abs/2112.10752

    Example:
    -------
    >>> python
    ...     # Example usage of the DiversityLoss class
    ...     G = 5  # Number of groups
    ...     V = 100  # Vocabulary size
    ...     loss_function = DiversityLoss(G, V)
    ...
    ...     # Assume 'predicted_perplexity' is a tensor with predicted perplexity values
    ...     # for each group and vocabulary dimension.
    ...     predicted_perplexity = tensorflow.random.uniform(shape=(G, V), minval=0.1, maxval=10.0)
    ...
    ...     # Compute the diversity loss
    ...     diversity_loss_value = loss_function(predicted_perplexity)
    ...
    ...     print(f"Diversity Loss: {diversity_loss_value.numpy()}")
    >>>
    In the above example, the `predicted_perplexity` tensor is randomly generated,
    but in a real scenario, this would be the model's output for perplexity values.

    """

    def __init__(self, number_groups: int, vocubulary_size: int, name="diversity_loss"):
        """
        Initializes the DiversityLoss class.

        Args:
            number_groups (int): The number of groups.
            vocubulary_size (int): The vocabulary size.
            name (str, optional): The name of the loss function. Defaults to "diversity_loss".
        """
        super().__init__(name=name)
        self.number_groups = number_groups  # Number of groups
        self.vocabulary_size = vocubulary_size  # Vocabulary size

    def call(self, perplexity):
        """
        Computes the diversity loss based on the predicted perplexity.

        Args:
            perplexity (tensor): A tensor containing the predicted perplexity values for each group
                                  and vocabulary dimension.

        Returns:
            diversity_loss (tensor): The computed diversity loss value.
        """
        # Calculate the logarithm of the perplexity
        log_perplexity = tensorflow.math.log(perplexity)

        # Compute the entropy for each group (summation over the vocabulary dimension)
        entropy = tensorflow.reduce_sum(perplexity * log_perplexity, axis=-1)

        # Compute the final diversity loss (summation over all groups and normalization)
        diversity_loss = tensorflow.reduce_sum(entropy) / (self.number_groups * self.vocabulary_size)

        return diversity_loss