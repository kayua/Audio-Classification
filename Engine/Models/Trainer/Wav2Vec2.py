#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
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
    import logging
    import numpy as np
    import tensorflow
    import seaborn as sns

    from tensorflow.keras import Model


    from Engine.Layers.MaskTimeLayer import TimeMaskingWithStorage


except ImportError as error:
    print(error)
    sys.exit(-1)


class Wav2Vec2DynamicTrainingModel(tensorflow.keras.Model):
    """
    Custom training model for dynamic target computation.
    """

    def __init__(self, encoder_model, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_model = encoder_model
        self.loss_fn = loss_fn  # Store loss function for train_step

    def call(self, inputs, training=False):
        return self.encoder_model(inputs, training=training)

    def train_step(self, data):
        """Dynamic training step that computes targets per batch."""
        x = data

        with tensorflow.GradientTape() as tape:
            # Forward pass - returns a dictionary
            outputs = self(x, training=True)

            # Extract values from dictionary
            contextualized = outputs['contextualized']
            quantized = outputs['quantized']
            perplexity = outputs['perplexity']
            mask_indices = outputs['mask_indices']

            # Prepare inputs for loss function
            y_true = (quantized, mask_indices, perplexity)

            # Call loss function directly (stored in self.loss_fn)
            loss = self.loss_fn(y_true, contextualized)

        # Update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return metrics
        return {'loss': loss}