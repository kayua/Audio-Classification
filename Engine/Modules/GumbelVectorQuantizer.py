#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{7}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/15'
__credits__ = ['unknown']

# MIT License
# Copyright (c) 2025 unknown

"""
===================================================================================
ARQUIVO 1: GumbelVectorQuantizer.py - CORRIGIDO
===================================================================================
Correções:
- Perplexity agora retorna escalar (média sobre grupos e vetores)
- Cálculo correto de perplexity para diversity loss
- Melhor estimativa da diversidade do codebook
"""

try:
    import sys
    import tensorflow
    from keras import Model, Layer
    from tensorflow.keras.layers import Dense
except ImportError as error:
    print(error)
    sys.exit(-1)


class GumbelVectorQuantizer(Layer):
    """
    A Gumbel-Softmax based Vector Quantizer for neural networks.

    CORRECTED VERSION:
    - Returns scalar perplexity for diversity loss computation
    - Proper perplexity calculation across all codebook entries

    Reference:
        - "Neural Discrete Representation Learning" by van den Oord et al.
          https://arxiv.org/abs/1711.00937
        - Wav2Vec 2.0 paper by Baevski et al.
    """

    def __init__(self):
        """Initialize the GumbelVectorQuantizer with default config."""
        super().__init__()
        self.number_groups = 4
        self.number_vectors = 16
        self.code_vector_size = 16 // 4
        self.temperature = 0.2

        self.linear = Dense(self.number_groups * self.number_vectors)
        self.code_book = self.add_weight(
            shape=(1, self.number_groups, self.number_vectors, self.code_vector_size),
            initializer="random_normal",
            trainable=True,
            name='codebook'
        )

    @staticmethod
    def _compute_perplexity(probs, lengths):
        """
        Computes the perplexity of the codebook usage distribution.

        Perplexity measures diversity: higher perplexity = more diverse code usage.
        Perplexity = exp(entropy)

        Args:
            probs (tf.Tensor): Probability distribution over codebook vectors.
                              Shape: (B, L, G, V)
            lengths (tf.Tensor): Sequence lengths in the batch. Shape: (B,)

        Returns:
            tf.Tensor: Scalar perplexity value (averaged over groups and vectors)
        """
        # Create mask for valid positions
        mask = tensorflow.sequence_mask(
            lengths,
            maxlen=tensorflow.shape(probs)[1],
            dtype=probs.dtype
        )
        mask = tensorflow.reshape(mask, (-1, tensorflow.shape(mask)[1], 1, 1))

        # Mask and compute average probability per codebook entry
        masked_probs = probs * mask
        num_values = tensorflow.reduce_sum(mask)

        # Average probability across all timesteps: shape (G, V)
        avg_probs = tensorflow.reduce_sum(masked_probs, axis=[0, 1]) / num_values

        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -tensorflow.reduce_sum(
            avg_probs * tensorflow.math.log(avg_probs + epsilon)
        )

        # Perplexity = exp(entropy)
        # Average over groups for final scalar
        perplexity = tensorflow.exp(entropy / tensorflow.cast(
            self.number_groups * self.number_vectors,
            tensorflow.float32
        ))

        return perplexity

    def call(self, hidden_state):
        """
        Performs forward pass of Gumbel-Softmax vector quantization.

        Args:
            hidden_state: Tuple of (hidden_states, lengths)
                - hidden_states (tf.Tensor): Input to quantize. Shape: (B, L, D1)
                - lengths (tf.Tensor): Sequence lengths. Shape: (B,)

        Returns:
            Tuple:
                - code_vectors (tf.Tensor): Quantized vectors. Shape: (B, L, D2)
                - perplexity (tf.Tensor): Scalar perplexity of codebook usage
        """
        hidden_states, lengths = hidden_state
        batch_size = tensorflow.shape(hidden_states)[0]
        length = tensorflow.shape(hidden_states)[1]

        # Project hidden states to codebook space
        hidden_states = self.linear(hidden_states)
        hidden_states = tensorflow.reshape(
            hidden_states,
            (batch_size * length, self.number_groups, self.number_vectors)
        )

        # Sample codebook vector probabilities using Gumbel-Softmax
        # Gumbel noise for exploration
        gumbel_noise = -tensorflow.math.log(
            -tensorflow.math.log(
                tensorflow.random.uniform(tensorflow.shape(hidden_states)) + 1e-10
            ) + 1e-10
        )

        code_vector_probs = tensorflow.nn.softmax(
            (hidden_states + gumbel_noise) / self.temperature,
            axis=-1
        )

        # Apply hard quantization (straight-through estimator)
        code_vector_probs_hard = tensorflow.one_hot(
            tensorflow.argmax(code_vector_probs, axis=-1),
            depth=self.number_vectors,
            dtype=code_vector_probs.dtype
        )

        # Straight-through: forward uses hard, backward uses soft
        code_vector_probs = tensorflow.stop_gradient(
            code_vector_probs_hard - code_vector_probs
        ) + code_vector_probs

        # Reshape for codebook lookup
        code_vector_probs = tensorflow.reshape(
            code_vector_probs,
            (batch_size * length, self.number_groups, self.number_vectors, 1)
        )

        # Apply codebook lookup
        code_vectors = code_vector_probs * self.code_book
        code_vectors = tensorflow.reduce_sum(code_vectors, axis=-2)
        code_vectors = tensorflow.reshape(
            code_vectors,
            (batch_size, length, self.number_groups * self.code_vector_size)
        )

        # Compute soft distribution for perplexity (no Gumbel noise)
        hidden_states_reshaped = tensorflow.reshape(
            self.linear(hidden_state[0]),
            (batch_size, length, self.number_groups, self.number_vectors)
        )
        code_vector_soft_distance = tensorflow.nn.softmax(
            hidden_states_reshaped,
            axis=-1
        )

        # Compute perplexity as scalar
        perplexity = self._compute_perplexity(code_vector_soft_distance, lengths)

        return code_vectors, perplexity


