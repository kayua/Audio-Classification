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
    import logging
    import numpy as np
    import tensorflow as tf
    import tensorflow

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
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Activation
    from Engine.Layers.MaskLayer import MaskCreator
    from tensorflow.keras.layers import TimeDistributed
    from Engine.Layers.MaskTimeLayer import TimeMasking
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Engine.Layers.QuantizerLayerMLP import QuantizationLayer
    from Engine.Models.Process.Wav2Vec2_Process import Wav2Vec2Process
    from Engine.Modules.GumbelVectorQuantizer import GumbelVectorQuantizer

except ImportError as error:
    print(error)
    sys.exit(-1)


class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
    """
    @AudioWav2Vec2
        AudioWav2Vec2 is a class that implements a deep learning model based on the Wav2Vec2
        architecture for audio feature extraction and classification tasks. This model
        utilizes a convolutional encoder followed by transformer-based attention layers,
        with quantization layers applied for feature compression.
        The architecture is particularly suited for speech recognition or other audio-related tasks.

        The model consists of:
            - Convolutional layers for feature extraction from audio waveforms.
            - Transformer blocks with multi-head attention to capture long-range dependencies.
            - Quantization layers to compress the feature representations.
            - Dense layers for the final classification.

    Reference:
        Baevski, A., Zhou, Y., & Mohamed, A. R. (2020). Wav2Vec 2.0: A Framework for Self-Supervised
        Learning of Speech Representations. *Proceedings of NeurIPS 2020*. https://arxiv.org/abs/2006.11477
    """

    def __init__(self, arguments):
        """
        Initialize the AudioWav2Vec2 model with specified hyperparameters.

        Args:
            @arguments: Object containing all model hyperparameters.
        """
        Wav2Vec2Process.__init__(self, arguments)

        # Initialize model parameters
        self.neural_network_model = None
        self.list_filters_encoder = arguments.wav_to_vec_list_filters_encoder
        self.loss_function = arguments.wav_to_vec_loss_function
        self.optimizer_function = arguments.wav_to_vec_optimizer_function
        self.kernel_size = arguments.wav_to_vec_kernel_size
        self.quantization_units = arguments.wav_to_vec_quantization_bits
        self.key_dimension = arguments.wav_to_vec_key_dimension
        self.intermediary_layer_activation = arguments.wav_to_vec_intermediary_layer_activation
        self.number_heads = arguments.wav_to_vec_number_heads
        self.input_dimension = arguments.wav_to_vec_input_dimension
        self.number_classes = arguments.number_classes
        self.dropout_rate = arguments.wav_to_vec_dropout_rate
        self.last_layer_activation = arguments.wav_to_vec_last_layer_activation
        self.model_name = "Wav2Vec2"

    def build_model(self) -> None:
        """
        Build the Wav2Vec2 model architecture using Keras.

        The model consists of the following components:
            - Convolutional encoder layers with the specified filters.
            - Transformer block with multi-head attention for capturing long-range dependencies.
            - Quantization layer for feature compression.
            - Dense layers for classification.

        The model is designed for audio processing tasks, particularly feature extraction.
        """

        # Input layer: reshape the input to match the expected format
        inputs = Input(shape=self.input_dimension, name='audio_input')
        neural_network_flow = Reshape((128, 80, 1))(inputs)

        # Apply convolutional layers to extract features from the input audio
        for number_filters in self.list_filters_encoder:
            neural_network_flow = TimeDistributed(Conv1D(
                number_filters,
                self.kernel_size,
                strides=(2,),
                use_bias=True,
                kernel_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None
            ))(neural_network_flow)

            neural_network_flow = TimeDistributed(GELU())(neural_network_flow)
            neural_network_flow = TimeDistributed(LayerNormalization())(neural_network_flow)

        # Flatten the convolutional output to feed into the dense layers
        flatten_flow = TimeDistributed(Flatten())(neural_network_flow)

        # Apply a dense layer with specified activation
        dense_layer = TimeDistributed(Dense(
            self.number_classes,
            activation=self.intermediary_layer_activation
        ))(flatten_flow)

        # Create lengths tensor - all sequences have the same length in this case
        sequence_length = self.input_dimension[0]  # 128
        lengths = Lambda(
            lambda x: tf.fill([tf.shape(x)[0]], tf.constant(sequence_length, dtype=tf.int32)),
            dtype=tf.int32,
            name='sequence_lengths'
        )(dense_layer)

        # Apply time masking using the corrected layer
        # Note: TimeMasking now returns a tuple (masked_output, mask_indices)
        masking_layer = TimeMasking(
            mask_time_prob=0.2,
            number_mask_time_steps=5,
            name='time_masking'
        )

        # Call the layer with both inputs
        time_masking, mask_indices = masking_layer([dense_layer, lengths])

        # Transformer block with multi-head attention
        transformer_attention = MultiHeadAttention(
            num_heads=self.number_heads,
            key_dim=self.key_dimension,
            name='transformer_attention'
        )(time_masking, time_masking)

        # Add residual connection and normalize using LayerNormalization
        transformer_attention = Add(name='residual_add_1')([time_masking, transformer_attention])
        transformer_attention = LayerNormalization(name='layer_norm_1')(transformer_attention)

        # Feed-forward network (fully connected layers)
        feedforward_network = Dense(
            self.number_classes * 4,
            name='feedforward_1'
        )(transformer_attention)
        feedforward_network = GELU(name='gelu_feedforward')(feedforward_network)
        feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_1')(feedforward_network)

        feedforward_network = Dense(
            self.number_classes,
            name='feedforward_2'
        )(feedforward_network)
        feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_2')(feedforward_network)

        # Add the output of the feed-forward network and normalize
        transformer_output = Add(name='residual_add_2')([transformer_attention, feedforward_network])
        transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)

        # Gumbel Vector Quantization layer for feature compression
        # Note: Quantization is applied to unmasked representations (dense_layer, not time_masking)
        quantization_layer = GumbelVectorQuantizer(name='gumbel_quantizer')
        quantized_output, perplexity = quantization_layer([dense_layer, lengths])

        # Create the Keras model with multiple outputs
        # Output 1: Contextualized representations from transformer
        # Output 2: Quantized representations
        self.neural_network_model = Model(
            inputs=inputs,
            outputs=[transformer_output, quantized_output],
            name=self.model_name
        )

        # DO NOT compile here - compilation is done in compile_and_train method
        # Display model summary
        self.neural_network_model.summary()
        logging.info("Model architecture built successfully. Compilation will be done in compile_and_train().")

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int,
                          validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the neural network model using a simplified Wav2Vec2 approach.

        This version trains the full model end-to-end for the classification task without
        the contrastive pretraining phase. The model architecture (encoder + transformer +
        quantizer) learns useful representations directly from the labeled data.

        This approach is more stable and often works well in practice, especially when you
        have sufficient labeled data for your task.

        Args:
            train_data (tensorflow.Tensor): The input training data (raw audio).
            train_labels (tensorflow.Tensor): The corresponding labels for the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of the batches for each training step.
            validation_data (tuple, optional): A tuple containing validation data and labels.

        Returns:
            tensorflow.keras.callbacks.History: The history object containing training metrics.
        """
        logging.info("=" * 70)
        logging.info("Starting Wav2Vec2-inspired end-to-end training")
        logging.info("Training full model directly for classification (no pretraining)")
        logging.info("=" * 70)

        # Add classification head on top of transformer output
        # Get the transformer output (first output of the model)
        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(
            self.neural_network_model.output[0]  # Use transformer output (contextualized)
        )

        # Classification head with intermediate layer
        neural_network_flow = Dense(
            self.number_classes * 2,
            name='classification_hidden'
        )(neural_network_flow)
        neural_network_flow = GELU(name='classification_gelu')(neural_network_flow)
        neural_network_flow = Dropout(
            self.dropout_rate,
            name='classification_dropout'
        )(neural_network_flow)

        # Output layer for classification
        neural_network_flow = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='classification_output'
        )(neural_network_flow)

        logging.info(
            f"Added classification head: Dense({self.number_classes * 2}) -> "
            f"Dense({self.number_classes}, activation='{self.last_layer_activation}')"
        )

        # Create new model with classification output
        self.neural_network_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=neural_network_flow,
            name=f"{self.model_name}_Classifier"
        )

        # Compile for supervised learning (end-to-end)
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )
        logging.info(f"Model compiled for end-to-end supervised learning.")
        logging.info(f"Optimizer: {self.optimizer_function}, Loss: {self.loss_function}")
        logging.info(f"All layers are trainable (encoder + transformer + classifier)")

        # Display final model summary
        logging.info("Final model architecture:")
        self.neural_network_model.summary()

        # Train on labeled data (end-to-end)
        logging.info(f"Training for {epochs} epochs with batch size {batch_size}.")
        training_history = self.neural_network_model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )

        logging.info("Training completed successfully.")
        logging.info("=" * 70)

        return training_history

    # Property definitions
    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_filters_encoder(self):
        return self._list_filters_encoder

    @list_filters_encoder.setter
    def list_filters_encoder(self, value):
        self._list_filters_encoder = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, value):
        self._optimizer_function = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

    @property
    def quantization_units(self):
        return self._quantization_units

    @quantization_units.setter
    def quantization_units(self, value):
        self._quantization_units = value

    @property
    def key_dimension(self):
        return self._key_dimension

    @key_dimension.setter
    def key_dimension(self, value):
        self._key_dimension = value

    @property
    def intermediary_layer_activation(self):
        return self._intermediary_layer_activation

    @intermediary_layer_activation.setter
    def intermediary_layer_activation(self, value):
        self._intermediary_layer_activation = value

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, value):
        self._number_heads = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value):
        self._number_classes = value

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value

    @property
    def last_layer_activation(self):
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value):
        self._last_layer_activation = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value