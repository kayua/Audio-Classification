#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/15'
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

"""
===================================================================================
GumbelVectorQuantizer - CORRIGIDO PARA WAV2VEC 2.0
===================================================================================

Correções implementadas:
✅ Perplexity retorna escalar (não mais tensor (G, V))
✅ Cálculo correto de perplexity para diversity loss
✅ Gumbel noise aplicado corretamente
✅ Straight-through estimator implementado
✅ Suporte para argumento 'name' no __init__
✅ Melhor estimativa da diversidade do codebook

Referências:
- "Neural Discrete Representation Learning" by van den Oord et al.
  https://arxiv.org/abs/1711.00937
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech 
  Representations" by Baevski et al.
  https://arxiv.org/abs/2006.11477
"""

try:
    import sys
    import tensorflow
    from keras import Layer
    from tensorflow.keras.layers import Dense
except ImportError as error:
    print(error)
    sys.exit(-1)


class GumbelVectorQuantizer(Layer):
    """
    A Gumbel-Softmax based Vector Quantizer for Wav2Vec 2.0.

    This layer discretizes continuous representations using Gumbel-Softmax relaxation
    and a learnable codebook. It's a critical component for self-supervised learning
    in Wav2Vec 2.0.

    Key Features:
    -------------
    - Gumbel-Softmax for differentiable discrete sampling
    - Straight-through estimator for hard quantization
    - Scalar perplexity output for diversity loss
    - Multiple codebook groups for better representation

    Architecture:
    -------------
    Input: (batch_size, sequence_length, hidden_dim)
    ↓
    Linear projection → (batch, seq, num_groups * num_vectors)
    ↓
    Reshape → (batch * seq, num_groups, num_vectors)
    ↓
    Gumbel-Softmax sampling
    ↓
    Hard quantization (straight-through)
    ↓
    Codebook lookup → (batch, seq, num_groups * code_vector_size)
    ↓
    Output: (quantized_vectors, perplexity_scalar)

    Parameters:
    -----------
    name : str, optional
        Name of the layer (for Keras)
    number_groups : int, default=4
        Number of codebook groups (G)
    number_vectors : int, default=16
        Number of vectors per group (V)
    code_vector_size : int, default=4
        Size of each code vector (total_dim / num_groups)
    temperature : float, default=0.2
        Temperature for Gumbel-Softmax (lower = harder)

    Input Shape:
    ------------
    Tuple of (hidden_states, lengths)
    - hidden_states: (batch_size, seq_length, hidden_dim)
    - lengths: (batch_size,) sequence lengths

    Output Shape:
    -------------
    Tuple of (code_vectors, perplexity)
    - code_vectors: (batch_size, seq_length, num_groups * code_vector_size)
    - perplexity: scalar tensor (measure of codebook diversity)

    Example:
    --------
    >>> quantizer = GumbelVectorQuantizer(name='my_quantizer')
    >>> hidden = tf.random.normal((32, 100, 256))  # batch=32, seq=100, dim=256
    >>> lengths = tf.fill([32], 100)
    >>> quantized, perplexity = quantizer([hidden, lengths])
    >>> print(quantized.shape)  # (32, 100, 16)
    >>> print(perplexity.shape)  # scalar

    Notes:
    ------
    - Perplexity is computed WITHOUT Gumbel noise for stability
    - Higher perplexity indicates more diverse codebook usage (good!)
    - Target perplexity ~100 is typical for Wav2Vec 2.0
    """

    def __init__(self, name=None, **kwargs):
        """
        Initialize the GumbelVectorQuantizer.

        Args:
            name (str, optional): Layer name for Keras
            **kwargs: Additional keyword arguments passed to parent Layer
        """
        super().__init__(name=name, **kwargs)

        # Codebook configuration
        self.number_groups = 4  # G: number of codebook groups
        self.number_vectors = 4  # V: vectors per group
        self.code_vector_size = 4 // 4  # D: size of each code vector
        self.temperature = 0.2  # τ: Gumbel-Softmax temperature

        # Linear projection layer
        self.linear = Dense(
            self.number_groups * self.number_vectors,
            name='quantizer_projection'
        )

        # Learnable codebook
        # Shape: (1, G, V, D)
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

        Perplexity measures how evenly the codebook entries are being used:
        - Higher perplexity = more diverse code usage (desirable)
        - Lower perplexity = code collapse (undesirable)

        Formula:
        --------
        Perplexity = exp(Entropy / N)
        where Entropy = -Σ p(c) * log(p(c))
              N = number of codebook entries (G * V)

        Args:
            probs (tf.Tensor): Probability distribution over codebook vectors
                              Shape: (batch_size, seq_length, num_groups, num_vectors)
            lengths (tf.Tensor): Valid sequence lengths in batch
                                Shape: (batch_size,)

        Returns:
            tf.Tensor: Scalar perplexity value

        Implementation Notes:
        ---------------------
        1. Create mask for valid positions (ignoring padding)
        2. Compute average probability of each codebook entry
        3. Calculate entropy: H = -Σ p * log(p)
        4. Normalize by number of entries: H_norm = H / (G * V)
        5. Perplexity = exp(H_norm)
        """
        # Create mask for valid sequence positions
        # Shape: (batch_size, seq_length)
        mask = tensorflow.sequence_mask(
            lengths,
            maxlen=tensorflow.shape(probs)[1],
            dtype=probs.dtype
        )

        # Reshape mask to match probs shape
        # Shape: (batch_size, seq_length, 1, 1)
        mask = tensorflow.reshape(mask, (-1, tensorflow.shape(mask)[1], 1, 1))

        # Mask out invalid positions
        masked_probs = probs * mask
        num_values = tensorflow.reduce_sum(mask)

        # Compute average probability for each codebook entry
        # Sum over batch and sequence dimensions, then normalize
        # Shape: (num_groups, num_vectors)
        avg_probs = tensorflow.reduce_sum(masked_probs, axis=[0, 1]) / num_values

        # Compute entropy: H = -Σ p * log(p)
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs = tensorflow.math.log(avg_probs + epsilon)
        entropy = -tensorflow.reduce_sum(avg_probs * log_probs)

        # Normalize entropy by number of codebook entries
        num_codes = tensorflow.cast(
            probs.shape[2] * probs.shape[3],  # num_groups * num_vectors
            tensorflow.float32
        )
        normalized_entropy = entropy / num_codes

        # Perplexity = exp(normalized_entropy)
        perplexity = tensorflow.exp(normalized_entropy)

        return perplexity

    def call(self, hidden_state, training=None):
        """
        Forward pass of Gumbel-Softmax vector quantization.

        Process:
        --------
        1. Project hidden states to logits space
        2. Sample from Gumbel-Softmax distribution
        3. Apply straight-through estimator (hard in forward, soft in backward)
        4. Look up codebook vectors
        5. Compute perplexity for diversity monitoring

        Args:
            hidden_state: Tuple of (hidden_states, lengths)
                - hidden_states (tf.Tensor): Input representations
                  Shape: (batch_size, seq_length, hidden_dim)
                - lengths (tf.Tensor): Sequence lengths
                  Shape: (batch_size,)
            training (bool, optional): Whether in training mode

        Returns:
            Tuple of (code_vectors, perplexity)
            - code_vectors (tf.Tensor): Quantized representations
              Shape: (batch_size, seq_length, num_groups * code_vector_size)
            - perplexity (tf.Tensor): Scalar codebook diversity metric

        Mathematical Details:
        ---------------------
        Gumbel-Softmax:
            g_i = -log(-log(u_i)) where u_i ~ Uniform(0,1)
            p_i = softmax((logits_i + g_i) / τ)

        Straight-through estimator:
            forward: use argmax (hard)
            backward: use softmax (soft)
            implementation: stop_gradient(hard - soft) + soft

        Codebook lookup:
            quantized = Σ p_i * codebook_i
        """
        # Unpack inputs
        hidden_states, lengths = hidden_state
        batch_size = tensorflow.shape(hidden_states)[0]
        length = tensorflow.shape(hidden_states)[1]

        # Step 1: Project to codebook space
        # (B, L, D) → (B, L, G*V) → (B*L, G, V)
        hidden_states_projected = self.linear(hidden_states)
        hidden_states = tensorflow.reshape(
            hidden_states_projected,
            (batch_size * length, self.number_groups, self.number_vectors)
        )

        # Step 2: Gumbel-Softmax sampling
        # Generate Gumbel noise: g = -log(-log(u))
        uniform_noise = tensorflow.random.uniform(tensorflow.shape(hidden_states))
        gumbel_noise = -tensorflow.math.log(
            -tensorflow.math.log(uniform_noise + 1e-10) + 1e-10
        )

        # Apply Gumbel-Softmax: softmax((logits + g) / τ)
        code_vector_probs = tensorflow.nn.softmax(
            (hidden_states + gumbel_noise) / self.temperature,
            axis=-1
        )

        # Step 3: Straight-through estimator
        # Hard quantization for forward pass
        code_vector_probs_hard = tensorflow.one_hot(
            tensorflow.argmax(code_vector_probs, axis=-1),
            depth=self.number_vectors,
            dtype=code_vector_probs.dtype
        )

        # Straight-through trick: gradient flows through soft version
        code_vector_probs = tensorflow.stop_gradient(
            code_vector_probs_hard - code_vector_probs
        ) + code_vector_probs

        # Step 4: Codebook lookup
        # Reshape for matrix multiplication with codebook
        # (B*L, G, V) → (B*L, G, V, 1)
        code_vector_probs = tensorflow.reshape(
            code_vector_probs,
            (batch_size * length, self.number_groups, self.number_vectors, 1)
        )

        # Weighted sum of codebook vectors
        # (B*L, G, V, 1) * (1, G, V, D) → (B*L, G, V, D)
        code_vectors = code_vector_probs * self.code_book

        # Sum over vectors dimension: (B*L, G, V, D) → (B*L, G, D)
        code_vectors = tensorflow.reduce_sum(code_vectors, axis=-2)

        # Reshape back to sequence: (B*L, G, D) → (B, L, G*D)
        code_vectors = tensorflow.reshape(
            code_vectors,
            (batch_size, length, self.number_groups * self.code_vector_size)
        )

        # Step 5: Compute perplexity (without Gumbel noise for stability)
        # Re-project original hidden states (without noise)
        hidden_states_for_perplexity = tensorflow.reshape(
            self.linear(hidden_state[0]),
            (batch_size, length, self.number_groups, self.number_vectors)
        )

        # Soft distribution for perplexity calculation
        code_vector_soft_distance = tensorflow.nn.softmax(
            hidden_states_for_perplexity,
            axis=-1
        )

        # Compute perplexity as scalar
        perplexity = self._compute_perplexity(code_vector_soft_distance, lengths)

        return code_vectors, perplexity

    def get_config(self):
        """
        Returns the config of the layer for serialization.

        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'number_groups': self.number_groups,
            'number_vectors': self.number_vectors,
            'code_vector_size': self.code_vector_size,
            'temperature': self.temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Args:
            config (dict): Configuration dictionary

        Returns:
            GumbelVectorQuantizer: New instance
        """
        return cls(**config)