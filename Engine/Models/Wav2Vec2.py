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
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage import zoom
    from scipy.ndimage import gaussian_filter
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

    from Engine.Layers.MaskTimeLayer import TimeMaskingWithStorage
    from Engine.Loss.ContrastiveLoss import Wav2Vec2ContrastiveLoss
    from Engine.Models.Process.Wav2Vec2_Process import Wav2Vec2Process
    from Engine.Modules.GumbelVectorQuantizer import GumbelVectorQuantizer
    from Engine.Models.Trainer.Wav2Vec2 import Wav2Vec2DynamicTrainingModel

except ImportError as error:
    print(error)
    sys.exit(-1)



class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
    """
    Wav2Vec2 implementation for self-supervised audio representation learning.

    This class implements the complete Wav2Vec2 architecture including:
    - Convolutional feature encoder
    - Contextualized transformer representations
    - Vector quantization with Gumbel softmax
    - Contrastive learning with diversity loss
    - Two-phase training procedure

    Attributes:
        neural_network_model (tf.keras.Model): The main Wav2Vec2 model
        gradcam_model (tf.keras.Model): Model for Grad-CAM visualization
        list_filters_encoder (list): List of filters for convolutional encoder
        loss_function: Loss function for supervised training
        optimizer_function: Optimizer for model training
        kernel_size (int): Kernel size for convolutional layers
        quantization_units (int): Number of quantization units
        key_dimension (int): Dimension for attention keys
        intermediary_layer_activation (str): Activation for intermediate layers
        number_heads (int): Number of attention heads
        input_dimension (tuple): Input shape for audio data
        number_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        last_layer_activation (str): Activation for final layer
        model_name (str): Name identifier for the model
        contrastive_temperature (float): Temperature for contrastive loss
        num_negatives (int): Number of negative samples
        diversity_weight (float): Weight for diversity loss term
    """
    def __init__(self, arguments):
        """
        Initialize Wav2Vec2 model with configuration parameters.

        Args:
            arguments: Configuration object containing all model parameters:
                - wav_to_vec_list_filters_encoder: List of filters for conv encoder
                - wav_to_vec_loss_function: Loss function for supervised training
                - wav_to_vec_optimizer_function: Optimizer for training
                - wav_to_vec_kernel_size: Kernel size for conv layers
                - wav_to_vec_quantization_bits: Quantization units
                - wav_to_vec_key_dimension: Attention key dimension
                - wav_to_vec_intermediary_layer_activation: Intermediate activation
                - wav_to_vec_number_heads: Number of attention heads
                - wav_to_vec_input_dimension: Input shape
                - number_classes: Number of output classes
                - wav_to_vec_dropout_rate: Dropout rate
                - wav_to_vec_last_layer_activation: Final layer activation
        """
        Wav2Vec2Process.__init__(self, arguments)

        self.neural_network_model = None
        self.gradcam_model = None
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

        self.contrastive_temperature = 0.1
        self.num_negatives = 100
        self.diversity_weight = 0.1

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """
        Build the complete Wav2Vec2 architecture.

        The architecture consists of:
        1. Convolutional feature encoder
        2. Time masking for self-supervised learning
        3. Transformer contextual network
        4. Vector quantization with Gumbel softmax
        5. Multi-head self-attention

        Returns:
            None: The model is stored in self.neural_network_model
        """

        inputs = Input(shape=self.input_dimension, name='audio_input')
        neural_network_flow = Reshape((128, 80, 1), name='reshape_input')(inputs)

        # Convolutional encoder
        for idx, number_filters in enumerate(self.list_filters_encoder):
            neural_network_flow = TimeDistributed(Conv1D(number_filters,
                                                         self.kernel_size,
                                                         strides=(2,),
                                                         use_bias=True,
                                                         name=f'conv1d_encoder_{idx}'),
                                                  name=f'time_dist_conv_{idx}')(neural_network_flow)

            neural_network_flow = TimeDistributed(GELU(), name=f'time_dist_gelu_{idx}')(neural_network_flow)
            neural_network_flow = TimeDistributed(LayerNormalization(), name=f'time_dist_ln_{idx}')(neural_network_flow)

        # Flatten and project
        flatten_flow = TimeDistributed(Flatten(), name='time_dist_flatten')(neural_network_flow)
        dense_layer = TimeDistributed(Dense(self.number_classes,
                                            activation=self.intermediary_layer_activation,
                                            name='dense_projection'), name='time_dist_dense')(flatten_flow)

        # Sequence lengths
        sequence_length = self.input_dimension[0]
        lengths = Lambda(lambda x: tensorflow.fill([tensorflow.shape(x)[0]],
                                                   tensorflow.constant(sequence_length, dtype=tensorflow.int32)),
                         dtype=tensorflow.int32, name='sequence_lengths')(dense_layer)

        # Time masking with storage
        masking_layer = TimeMaskingWithStorage(mask_time_prob=0.065, number_mask_time_steps=10, name='time_masking')
        time_masking, mask_indices = masking_layer([dense_layer, lengths])

        # Transformer encoder
        transformer_attention = MultiHeadAttention(num_heads=self.number_heads,
                                                   key_dim=self.key_dimension,
                                                   name='transformer_attention')(time_masking, time_masking)

        transformer_attention = Add(name='residual_add_1')([time_masking, transformer_attention])
        transformer_attention = LayerNormalization(name='layer_norm_1')(transformer_attention)

        # Feedforward
        feedforward_network = Dense(self.number_classes * 4, name='feedforward_1')(transformer_attention)
        feedforward_network = GELU(name='gelu_feedforward')(feedforward_network)
        feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_1')(feedforward_network)

        feedforward_network = Dense(self.number_classes, name='feedforward_2')(feedforward_network)
        feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_2')(feedforward_network)

        transformer_output = Add(name='residual_add_2')([transformer_attention, feedforward_network])
        transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)

        # Vector quantization
        quantization_layer = GumbelVectorQuantizer(name='gumbel_quantizer')
        quantized_output, perplexity = quantization_layer([dense_layer, lengths])

        self.neural_network_model = Model(inputs=inputs,
                                          outputs=[transformer_output, (quantized_output, perplexity)],
                                          name=self.model_name)

        self.neural_network_model.summary()
        logging.info("✓ Wav2Vec2 architecture built")

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int,
                          validation_data: tuple = None,
                          freeze_encoder: bool = False) -> tensorflow.keras.callbacks.History:
        """
        Two-phase training procedure for Wav2Vec2.

        Phase 1: Self-supervised pretraining with contrastive loss
        Phase 2: Supervised fine-tuning for classification

        Args:
            train_data (tf.Tensor): Training audio data
            train_labels (tf.Tensor): Training labels
            epochs (int): Number of training epochs per phase
            batch_size (int): Batch size for training
            validation_data (tuple, optional): Validation data (x_val, y_val)
            freeze_encoder (bool): Whether to freeze encoder during fine-tuning

        Returns:
            tf.keras.callbacks.History: Training history from fine-tuning phase

        Example:
            ```python
            # Initialize model
            wav2vec2 = AudioWav2Vec2(config)
            wav2vec2.build_model()

            # Train with two-phase approach
            history = wav2vec2.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                freeze_encoder=False
            )

            # Evaluate model
            test_loss, test_accuracy = wav2vec2.neural_network_model.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_accuracy:.4f}")
            ```
        """

        logging.info("=" * 80)
        logging.info("WAV2VEC2 TRAINING (CORRECTED)")
        logging.info("=" * 80)

        # Phase 1: Pretraining
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: SELF-SUPERVISED PRETRAINING")
        logging.info("=" * 80)

        # Create contrastive loss
        contrastive_loss = Wav2Vec2ContrastiveLoss(temperature=self.contrastive_temperature,
                                                   num_negatives=self.num_negatives,
                                                   diversity_weight=self.diversity_weight)

        # Wrap model with dynamic training model, passing the loss function
        pretrain_model = Wav2Vec2DynamicTrainingModel(self.neural_network_model,
                                                      loss_fn=contrastive_loss,
                                                      name='pretrain_model')

        # Compile model (optimizer only, loss is handled in train_step)
        pretrain_model.compile(optimizer=self.optimizer_function)

        logging.info("✓ Compiled with InfoNCE + diversity loss")
        logging.info(f"⚙ Starting pretraining for {epochs} epochs...")

        pretrain_history = pretrain_model.fit(train_data, epochs=epochs, batch_size=batch_size, verbose=1)

        logging.info("✓ Pretraining completed!")

        # Phase 2: Fine-tuning
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: SUPERVISED FINE-TUNING")
        logging.info("=" * 80)

        if freeze_encoder:
            self.neural_network_model.trainable = True
            logging.info("✓ Froze encoder")
        else:
            self.neural_network_model.trainable = True
            logging.info("✓ Encoder trainable")

        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(self.neural_network_model.output[0])

        neural_network_flow = Dense(self.number_classes * 2, name='classification_hidden')(neural_network_flow)
        neural_network_flow = GELU(name='classification_gelu')(neural_network_flow)
        neural_network_flow = Dropout(self.dropout_rate, name='classification_dropout')(neural_network_flow)

        neural_network_flow = Dense(self.number_classes,
                                    activation=self.last_layer_activation,
                                    name='classification_output')(neural_network_flow)

        self.neural_network_model = Model(inputs=self.neural_network_model.inputs,
                                          outputs=neural_network_flow,
                                          name=f"{self.model_name}_FineTuned")

        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

        logging.info(f"⚙ Starting fine-tuning for {epochs} epochs...")

        finetune_history = self.neural_network_model.fit(train_data,
                                                         train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data,
                                                         verbose=1)

        return finetune_history


    # Properties (unchanged)
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

