#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/14'
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
    import tensorflow as tf

    from keras import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class TimeMasking(Layer):
    """
    TensorFlow layer for applying time masking to sequential data (vectorized implementation).

    This layer masks a percentage of time steps in the input sequence by setting their values to zero.
    This is a fully vectorized implementation without Python loops for better performance.

    Attributes:
        mask_time_probabilities (float): Probability of masking time steps.
        number_mask_time_steps (int): Number of consecutive time steps to mask.

    Reference:
        - Inspired by SpecAugment: https://arxiv.org/abs/1904.08779
        - Wav2Vec 2.0: https://arxiv.org/abs/2006.11477

    Example:
        >>> # Create a TimeMasking layer
        ...   time_masking = TimeMasking(mask_time_prob=0.2, number_mask_time_steps=5)
        ...   hidden_states = tf.random.normal([2, 10, 32])  # (batch_size=2, sequence_length=10, feature_dim=32)
        ...   lengths = tf.constant([10, 8], dtype=tf.int32)  # Valid sequence lengths
        ...   masked_states, mask = time_masking([hidden_states, lengths])
        ...   print(masked_states.shape)  # (2, 10, 32)
        >>>   print(mask.shape)  # (2, 10)
    """

    def __init__(self, mask_time_prob: float, number_mask_time_steps: int, **kwargs):
        """
        Initializes the TimeMasking layer.

        Args:
            mask_time_prob (float): Probability of masking time steps (0.0 to 1.0).
            number_mask_time_steps (int): Number of consecutive time steps to mask in each span.
        """
        super(TimeMasking, self).__init__(**kwargs)
        self.mask_time_probabilities = mask_time_prob
        self.number_mask_time_steps = number_mask_time_steps

    def call(self, inputs):
        """
        Applies time masking to the input hidden states (vectorized).

        Args:
            inputs: Either a list/tuple of [hidden_states, lengths] or just hidden_states
                - hidden_states (tf.Tensor): Input tensor of shape `(B, L, D)` where B is batch size,
                                           L is sequence length, and D is feature dimension.
                - lengths (tf.Tensor): Tensor of shape `(B,)` containing the valid sequence lengths
                                      for each batch element.

        Returns:
            tuple:
                - Masked hidden states (tf.Tensor of shape `(B, L, D)`).
                - Time mask (tf.Tensor of shape `(B, L)`, dtype=tf.bool).
        """
        # Unpack inputs - accept both list/tuple format and separate arguments
        if isinstance(inputs, (list, tuple)):
            hidden_states, lengths = inputs
        else:
            hidden_states = inputs
            # If lengths not provided, assume full sequence length for all samples
            lengths = tf.fill([tf.shape(hidden_states)[0]], tf.shape(hidden_states)[1])

        batch_size = tf.shape(hidden_states)[0]
        number_steps = tf.shape(hidden_states)[1]
        hidden_size = tf.shape(hidden_states)[2]

        # Calculate number of time steps to mask per sequence
        # mask_lengths: (B,)
        mask_lengths = tf.cast(
            tf.math.ceil(self.mask_time_probabilities * tf.cast(lengths, tf.float32)),
            tf.int32
        )
        mask_lengths = tf.maximum(mask_lengths, 1)  # At least one mask per sequence

        # Get maximum mask length across batch
        max_mask_length = tf.reduce_max(mask_lengths)

        # Generate random start indices for each mask span
        # Shape: (B, max_mask_length)
        random_starts = tf.random.uniform(
            shape=(batch_size, max_mask_length),
            minval=0,
            maxval=number_steps - self.number_mask_time_steps + 1,
            dtype=tf.int32
        )
        random_starts = tf.maximum(random_starts, 0)

        # Create range for consecutive positions
        # Shape: (number_mask_time_steps,)
        offsets = tf.range(self.number_mask_time_steps, dtype=tf.int32)

        # Expand dimensions for broadcasting
        # random_starts: (B, max_mask_length, 1)
        # offsets: (1, 1, number_mask_time_steps)
        random_starts_expanded = tf.expand_dims(random_starts, axis=-1)
        offsets_expanded = tf.reshape(offsets, [1, 1, self.number_mask_time_steps])

        # Compute all mask positions
        # Shape: (B, max_mask_length, number_mask_time_steps)
        mask_positions = random_starts_expanded + offsets_expanded

        # Clip to valid range
        mask_positions = tf.clip_by_value(mask_positions, 0, number_steps - 1)

        # Reshape to (B, max_mask_length * number_mask_time_steps)
        mask_positions_flat = tf.reshape(mask_positions, [batch_size, -1])

        # Create a mask to filter out positions beyond the mask_length for each batch
        # Shape: (B, max_mask_length)
        sequence_mask = tf.sequence_mask(mask_lengths, max_mask_length, dtype=tf.int32)

        # Expand to match mask_positions shape
        # Shape: (B, max_mask_length, number_mask_time_steps)
        sequence_mask_expanded = tf.expand_dims(sequence_mask, axis=-1)
        sequence_mask_expanded = tf.tile(sequence_mask_expanded, [1, 1, self.number_mask_time_steps])
        sequence_mask_flat = tf.reshape(sequence_mask_expanded, [batch_size, -1])

        # Create batch indices
        # Shape: (B, max_mask_length * number_mask_time_steps)
        batch_indices = tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1),
            [1, max_mask_length * self.number_mask_time_steps]
        )

        # Stack to create scatter indices
        # Shape: (B * max_mask_length * number_mask_time_steps, 2)
        scatter_indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(mask_positions_flat, [-1])
        ], axis=1)

        # Create updates (True for positions to mask)
        # Filter by sequence_mask to only update valid positions
        updates = tf.cast(tf.reshape(sequence_mask_flat, [-1]), tf.bool)

        # Initialize mask as all False
        time_mask_indices = tf.zeros([batch_size, number_steps], dtype=tf.bool)

        # Scatter update to set masked positions to True
        time_mask_indices = tf.tensor_scatter_nd_update(
            time_mask_indices,
            scatter_indices,
            updates
        )

        # Apply mask to hidden states
        # Expand mask to match hidden_states dimensions
        # Shape: (B, L, 1)
        mask_expanded = tf.expand_dims(time_mask_indices, axis=-1)

        # Create masked hidden states (set masked positions to zero)
        masked_hidden_states = tf.where(
            mask_expanded,
            tf.zeros_like(hidden_states),
            hidden_states
        )

        return masked_hidden_states, time_mask_indices

    @staticmethod
    def compute_output_shape(input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input (or list of shape tuples for multiple inputs)

        Returns:
            Tuple of output shapes for (masked_hidden_states, mask)
        """
        if isinstance(input_shape, list):
            hidden_shape = input_shape[0]
        else:
            hidden_shape = input_shape

        return (hidden_shape, (hidden_shape[0], hidden_shape[1]))

    def get_config(self):
        """
        Returns the config of the layer for serialization.

        Returns:
            Dictionary with layer configuration
        """
        config = super(TimeMasking, self).get_config()
        config.update({
            'mask_time_prob': self.mask_time_probabilities,
            'number_mask_time_steps': self.number_mask_time_steps
        })
        return config


# !/usr/bin/python3
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
TimeMaskingWithStorage - CLASSE AUXILIAR PARA WAV2VEC 2.0
===================================================================================

Esta classe estende TimeMasking para armazenar os índices de máscara gerados,
que são necessários para calcular a loss InfoNCE apenas nas posições mascaradas.

Correções:
✅ Não passa argumento 'training' para super().call() (compatibilidade)
✅ Armazena mask_indices em atributo acessível
✅ Suporta argumentos nomeados (name, **kwargs)
✅ Fornece método get_last_mask_indices() para acesso externo

Uso no Wav2Vec 2.0:
-------------------
1. Durante forward pass, armazena os índices mascarados
2. Durante cálculo da loss, recupera os índices via get_last_mask_indices()
3. Loss InfoNCE usa os índices para computar loss apenas em posições mascaradas
"""

try:
    import sys
    import tensorflow as tf
    from Engine.Layers.MaskTimeLayer import TimeMasking
except ImportError as error:
    print(f"Import error: {error}")
    print("Make sure TimeMasking is available in Engine.Layers.MaskTimeLayer")
    sys.exit(-1)


class TimeMaskingWithStorage(TimeMasking):
    """
    Extended TimeMasking layer that stores mask indices for loss computation.

    This wrapper class extends the original TimeMasking layer to store the
    mask indices generated during the forward pass. These indices are required
    by the corrected InfoNCE loss function, which only computes loss on
    masked positions (as per the original Wav2Vec 2.0 paper).

    Key Features:
    -------------
    - Stores last computed mask indices in _last_mask_indices
    - Compatible with Keras training loop (accepts training argument)
    - Provides get_last_mask_indices() method for external access
    - Thread-safe for single model instance

    Architecture:
    -------------
    Input: (hidden_states, lengths)
    ↓
    TimeMasking.call(inputs) → Apply temporal masking
    ↓
    Store mask_indices → _last_mask_indices
    ↓
    Output: (masked_output, mask_indices)

    Usage Example:
    --------------
    >>> masking_layer = TimeMaskingWithStorage(
    ...     mask_time_prob=0.065,
    ...     number_mask_time_steps=10,
    ...     name='time_masking'
    ... )
    >>>
    >>> # Forward pass
    >>> masked_output, mask_indices = masking_layer([features, lengths])
    >>>
    >>> # Later, during loss computation
    >>> stored_mask = masking_layer.get_last_mask_indices()
    >>> loss = compute_loss_only_on_masked(contextualized, quantized, stored_mask)

    Integration with Wav2Vec2DynamicTrainingModel:
    -----------------------------------------------
    The train_step() method accesses mask indices like this:

    >>> def train_step(self, data):
    ...     x = data
    ...     with tf.GradientTape() as tape:
    ...         contextualized, (quantized, perplexity) = self(x, training=True)
    ...
    ...         # Get mask layer
    ...         mask_layer = None
    ...         for layer in self.encoder_model.layers:
    ...             if isinstance(layer, TimeMaskingWithStorage):
    ...                 mask_layer = layer
    ...                 break
    ...
    ...         # Get stored mask indices
    ...         if mask_layer is not None:
    ...             mask_indices = mask_layer.get_last_mask_indices()
    ...
    ...         # Compute loss only on masked positions
    ...         y_true = (quantized, mask_indices, perplexity)
    ...         loss = self.compiled_loss(y_true, contextualized)
    ...
    ...     # Update weights...

    Notes:
    ------
    - The parent TimeMasking class does NOT accept 'training' argument in call()
    - We accept it here for Keras compatibility but don't pass it to parent
    - Mask indices are stored per forward pass (not accumulated)
    - Thread-safe for single model, but not for multiple concurrent models

    Attributes:
    -----------
    _last_mask_indices : tf.Tensor or None
        Most recently computed mask indices, shape (batch_size, seq_length)
    """

    def __init__(self, mask_time_prob=0.065, number_mask_time_steps=10,
                 name=None, **kwargs):
        """
        Initialize TimeMaskingWithStorage layer.

        Args:
            mask_time_prob (float, optional): Probability of masking a time step.
                Default: 0.065 (6.5% masking rate as in Wav2Vec 2.0)
            number_mask_time_steps (int, optional): Span length of consecutive
                time steps to mask. Default: 10
            name (str, optional): Layer name for Keras
            **kwargs: Additional keyword arguments passed to parent TimeMasking

        Example:
            >>> layer = TimeMaskingWithStorage(
            ...     mask_time_prob=0.065,
            ...     number_mask_time_steps=10,
            ...     name='my_masking_layer'
            ... )
        """
        super().__init__(
            mask_time_prob=mask_time_prob,
            number_mask_time_steps=number_mask_time_steps,
            name=name,
            **kwargs
        )
        self._last_mask_indices = None

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass that stores mask indices.

        This method calls the parent TimeMasking.call() method to perform
        the actual masking, then stores the mask indices for later retrieval.

        IMPORTANT: The parent TimeMasking.call() does NOT accept 'training'
        argument, so we accept it here for Keras compatibility but don't
        pass it to the parent.

        Args:
            inputs: Tuple of (hidden_states, lengths)
                - hidden_states (tf.Tensor): Input features to mask
                  Shape: (batch_size, seq_length, feature_dim)
                - lengths (tf.Tensor): Valid sequence lengths
                  Shape: (batch_size,)
            training (bool, optional): Whether in training mode (not passed to parent)
            **kwargs: Additional arguments (not passed to parent)

        Returns:
            If parent returns tuple:
                Tuple of (masked_output, mask_indices)
                - masked_output: Features with temporal masking applied
                - mask_indices: Boolean mask indicating masked positions
            Otherwise:
                Just the masked output

        Side Effects:
            Sets self._last_mask_indices to the computed mask indices

        Example:
            >>> features = tf.random.normal((32, 128, 256))
            >>> lengths = tf.fill([32], 128)
            >>>
            >>> masked_output, mask_indices = layer([features, lengths], training=True)
            >>>
            >>> # Verify mask was stored
            >>> assert layer.get_last_mask_indices() is not None
            >>> assert tf.reduce_all(mask_indices == layer.get_last_mask_indices())
        """
        # Call parent WITHOUT training argument (TimeMasking doesn't accept it)
        result = super().call(inputs)

        # Check if result is a tuple containing mask indices
        if isinstance(result, tuple) and len(result) == 2:
            masked_output, mask_indices = result

            # Store mask indices for later access during loss computation
            self._last_mask_indices = mask_indices

            return result

        # If parent doesn't return mask indices, just return the result
        # (This shouldn't happen with standard TimeMasking)
        return result

    def get_last_mask_indices(self):
        """
        Returns the most recently computed mask indices.

        This method is called by the training loop to retrieve the mask
        indices that were stored during the most recent forward pass.

        Returns:
            tf.Tensor or None: Boolean mask tensor of shape (batch_size, seq_length)
                indicating which positions were masked (True = masked).
                Returns None if no forward pass has been performed yet.

        Example:
            >>> # After forward pass
            >>> masked_output, _ = layer([features, lengths])
            >>>
            >>> # Get stored mask
            >>> mask = layer.get_last_mask_indices()
            >>> print(mask.shape)  # (batch_size, seq_length)
            >>> print(mask.dtype)  # bool

        Notes:
            - Returns None before the first forward pass
            - Overwritten on each forward pass (not accumulated)
            - Not thread-safe across multiple models
        """
        return self._last_mask_indices

    def reset_mask_indices(self):
        """
        Resets the stored mask indices to None.

        This can be useful for debugging or when you want to ensure
        a clean state between different forward passes.

        Example:
            >>> layer.reset_mask_indices()
            >>> assert layer.get_last_mask_indices() is None
        """
        self._last_mask_indices = None

    def get_config(self):
        """
        Returns the config of the layer for serialization.

        Returns:
            dict: Configuration dictionary including parent config

        Example:
            >>> config = layer.get_config()
            >>> new_layer = TimeMaskingWithStorage.from_config(config)
        """
        config = super().get_config()
        # No additional config needed beyond parent
        return config