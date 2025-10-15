"""
===================================================================================
ARQUIVO 3: TimeMaskingWithStorage - NOVA CLASSE AUXILIAR
===================================================================================
Wrapper para TimeMasking que armazena os índices de máscara
"""

try:
    from Engine.Layers.MaskTimeLayer import TimeMasking
    import tensorflow as tf
except ImportError:
    # Fallback if import fails
    pass


class TimeMaskingWithStorage(TimeMasking):
    """
    Extended TimeMasking layer that stores mask indices for loss computation.

    This is required for the corrected InfoNCE loss which only computes
    loss on masked positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_mask_indices = None

    def call(self, inputs, training=None):
        """
        Forward pass that stores mask indices.

        Args:
            inputs: Tuple of (hidden_states, lengths)
            training: Boolean, whether in training mode

        Returns:
            Tuple of (masked_output, mask_indices) if training
            Otherwise just masked_output
        """
        result = super().call(inputs, training=training)

        if isinstance(result, tuple) and len(result) == 2:
            masked_output, mask_indices = result
            # Store mask indices for access during loss computation
            self._last_mask_indices = mask_indices
            return result

        return result

    def get_last_mask_indices(self):
        """Returns the most recently computed mask indices."""
        return self._last_mask_indices
