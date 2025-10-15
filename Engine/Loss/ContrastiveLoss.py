"""
===================================================================================
ARQUIVO 4: Wav2Vec2 Main Model - VERSÃO COMPLETA CORRIGIDA
===================================================================================
"""

try:
    import sys
    import logging
    import numpy as np
    import tensorflow as tf
    import tensorflow
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    from tensorflow.keras import Model
    from Engine.Layers.GELU import GELU

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Activation
    from Engine.Layers.MaskLayer import MaskCreator
    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D

except ImportError as error:
    print(error)
    sys.exit(-1)


class Wav2Vec2ContrastiveLoss(tf.keras.losses.Loss):
    """
    True Wav2Vec2 Contrastive Loss (InfoNCE Loss) with Diversity Loss.

    Implements:
    - InfoNCE loss with negative sampling
    - Loss computed only on masked positions
    - Diversity loss to encourage codebook usage
    """

    def __init__(self, temperature=0.1, num_negatives=100,
                 diversity_weight=0.1, name='wav2vec2_contrastive_loss'):
        super().__init__(name=name)
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.diversity_weight = diversity_weight

    def sample_negatives(self, quantized, batch_size, seq_length, num_negatives):
        """
        Sample negative examples from other time steps and other examples in batch.

        Args:
            quantized: (batch_size, seq_length, hidden_dim)
            batch_size: int
            seq_length: int
            num_negatives: int

        Returns:
            negatives: (batch_size, seq_length, num_negatives, hidden_dim)
        """
        hidden_dim = tf.shape(quantized)[-1]

        # Flatten to (batch_size * seq_length, hidden_dim)
        quantized_flat = tf.reshape(quantized, [-1, hidden_dim])
        total_positions = batch_size * seq_length

        # Sample random indices for each position
        negative_indices = tf.random.uniform(
            shape=(batch_size, seq_length, num_negatives),
            minval=0,
            maxval=total_positions,
            dtype=tf.int32
        )

        # Flatten and gather
        negative_indices_flat = tf.reshape(negative_indices, [-1])
        negatives_flat = tf.gather(quantized_flat, negative_indices_flat)

        # Reshape back
        negatives = tf.reshape(
            negatives_flat,
            [batch_size, seq_length, num_negatives, hidden_dim]
        )

        return negatives

    def compute_diversity_loss(self, perplexity):
        """
        Diversity loss to encourage using different codebook entries.

        Args:
            perplexity: Scalar tensor representing codebook perplexity

        Returns:
            diversity_loss: Scalar tensor
        """
        target_perplexity = 100.0
        diversity_loss = tf.math.squared_difference(
            perplexity, target_perplexity
        ) / target_perplexity
        return diversity_loss

    def call(self, y_true, y_pred):
        """
        Compute InfoNCE loss with diversity loss.

        Args:
            y_true: Tuple of (quantized, mask_indices, perplexity)
            y_pred: contextualized representations (batch_size, seq_length, hidden_dim)

        Returns:
            total_loss: Combined contrastive + diversity loss
        """
        quantized, mask_indices, perplexity = y_true
        contextualized = y_pred

        # Normalize representations
        contextualized = tf.nn.l2_normalize(contextualized, axis=-1)
        quantized = tf.nn.l2_normalize(quantized, axis=-1)

        batch_size = tf.shape(contextualized)[0]
        seq_length = tf.shape(contextualized)[1]

        # Sample negatives
        negatives = self.sample_negatives(
            quantized, batch_size, seq_length, self.num_negatives
        )
        negatives = tf.nn.l2_normalize(negatives, axis=-1)

        # Positive similarity
        positive_similarity = tf.reduce_sum(
            contextualized * quantized, axis=-1
        ) / self.temperature

        # Negative similarities
        contextualized_expanded = tf.expand_dims(contextualized, axis=2)
        negative_similarities = tf.reduce_sum(
            contextualized_expanded * negatives, axis=-1
        ) / self.temperature

        # InfoNCE loss (log-sum-exp for stability)
        positive_exp = tf.exp(positive_similarity)
        negative_exp_sum = tf.reduce_sum(tf.exp(negative_similarities), axis=-1)
        log_prob = positive_similarity - tf.math.log(positive_exp + negative_exp_sum + 1e-7)

        # Apply mask
        mask_indices_float = tf.cast(mask_indices, tf.float32)
        masked_log_prob = log_prob * mask_indices_float
        num_masked = tf.reduce_sum(mask_indices_float) + 1e-7
        contrastive_loss = -tf.reduce_sum(masked_log_prob) / num_masked

        # Diversity loss
        diversity_loss = self.compute_diversity_loss(perplexity)

        # Combined
        total_loss = contrastive_loss + self.diversity_weight * diversity_loss

        return total_loss
