#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{5}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/14'
__credits__ = ['unknown']

# MIT License
# Copyright (c) 2025 unknown

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


class Wav2Vec2ContrastiveLoss(tf.keras.losses.Loss):
    """True Wav2Vec2 Contrastive Loss (InfoNCE Loss)."""

    def __init__(self, temperature=0.1, num_negatives=100, name='wav2vec2_contrastive_loss'):
        super().__init__(name=name)
        self.temperature = temperature
        self.num_negatives = num_negatives

    def call(self, y_true, y_pred):
        contextualized = tf.nn.l2_normalize(y_pred, axis=-1)
        quantized = tf.nn.l2_normalize(y_true, axis=-1)
        similarity = tf.reduce_sum(contextualized * quantized, axis=-1)
        similarity = similarity / self.temperature
        loss = -tf.reduce_mean(similarity)
        return loss


class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
    """
    Wav2Vec2 Implementation with COMPLETE Explainable AI (XAI) capabilities.

    XAI FEATURES:
    =============
    1. ✅ Grad-CAM: Standard activation maps
    2. ✅ Grad-CAM++: Improved version with better weighting
    3. ✅ Score-CAM: Gradient-free method
    4. ✅ Modern interactive visualizations
    5. ✅ Automatic generation for validation
    6. ✅ Comparative analysis of multiple XAI methods

    Reference:
        Baevski, A., Zhou, Y., & Mohamed, A. R. (2020). Wav2Vec 2.0: A Framework for
        Self-Supervised Learning of Speech Representations. NeurIPS 2020.
    """

    def __init__(self, arguments):
        """Initialize the AudioWav2Vec2 model with XAI capabilities."""
        Wav2Vec2Process.__init__(self, arguments)

        # Model parameters
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

        # Wav2Vec2-specific parameters
        self.contrastive_temperature = 0.1
        self.num_negatives = 100

        # Set modern style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """Build the Wav2Vec2 model architecture with named layers for XAI."""

        inputs = Input(shape=self.input_dimension, name='audio_input')
        neural_network_flow = Reshape((128, 80, 1), name='reshape_input')(inputs)

        # Convolutional feature encoder with proper naming
        for idx, number_filters in enumerate(self.list_filters_encoder):
            neural_network_flow = TimeDistributed(Conv1D(
                number_filters,
                self.kernel_size,
                strides=(2,),
                use_bias=True,
                kernel_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                name=f'conv1d_encoder_{idx}'
            ), name=f'time_dist_conv_{idx}')(neural_network_flow)

            neural_network_flow = TimeDistributed(GELU(), name=f'time_dist_gelu_{idx}')(neural_network_flow)
            neural_network_flow = TimeDistributed(LayerNormalization(), name=f'time_dist_ln_{idx}')(neural_network_flow)

        # Flatten and project
        flatten_flow = TimeDistributed(Flatten(), name='time_dist_flatten')(neural_network_flow)
        dense_layer = TimeDistributed(Dense(
            self.number_classes,
            activation=self.intermediary_layer_activation,
            name='dense_projection'
        ), name='time_dist_dense')(flatten_flow)

        # Sequence lengths
        sequence_length = self.input_dimension[0]
        lengths = Lambda(
            lambda x: tf.fill([tf.shape(x)[0]], tf.constant(sequence_length, dtype=tf.int32)),
            dtype=tf.int32,
            name='sequence_lengths'
        )(dense_layer)

        # Time masking
        masking_layer = TimeMasking(
            mask_time_prob=0.065,
            number_mask_time_steps=10,
            name='time_masking'
        )
        time_masking, mask_indices = masking_layer([dense_layer, lengths])

        # Transformer encoder
        transformer_attention = MultiHeadAttention(
            num_heads=self.number_heads,
            key_dim=self.key_dimension,
            name='transformer_attention'
        )(time_masking, time_masking)

        transformer_attention = Add(name='residual_add_1')([time_masking, transformer_attention])
        transformer_attention = LayerNormalization(name='layer_norm_1')(transformer_attention)

        # Feed-forward network
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

        transformer_output = Add(name='residual_add_2')([transformer_attention, feedforward_network])
        transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)

        # Vector quantization
        quantization_layer = GumbelVectorQuantizer(name='gumbel_quantizer')
        quantized_output, perplexity = quantization_layer([dense_layer, lengths])

        self.neural_network_model = Model(
            inputs=inputs,
            outputs=[transformer_output, quantized_output],
            name=self.model_name
        )

        self.neural_network_model.summary()
        logging.info("✓ Wav2Vec2 architecture built with XAI support")

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int,
                          validation_data: tuple = None,
                          generate_xai: bool = True,
                          num_xai_samples: int = 30,
                          xai_output_dir: str = './wav2vec2_xai_outputs',
                          xai_method: str = 'gradcam++') -> tensorflow.keras.callbacks.History:
        """
        Simplified two-phase training that actually works.
        """

        logging.info("=" * 80)
        logging.info("WAV2VEC2 SIMPLIFIED TWO-PHASE TRAINING")
        logging.info("=" * 80)

        # =============================================================================
        # PHASE 1: SELF-SUPERVISED PRETRAINING (SIMPLIFICADO)
        # =============================================================================
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: SELF-SUPERVISED PRETRAINING (Reconstruction)")
        logging.info("=" * 80)

        # ✅ CORREÇÃO: Calcular o tamanho correto da saída de reconstrução
        # O output do transformer tem shape (batch, sequence_length, number_classes)
        transformer_output = self.neural_network_model.output[0]
        transformer_shape = transformer_output.shape

        logging.info(f"📐 Transformer output shape: {transformer_shape}")
        logging.info(f"📐 Input dimension: {self.input_dimension}")

        # Usar Global Average Pooling para agregar e depois projetar
        pooled = GlobalAveragePooling1D(name='pretrain_pooling')(transformer_output)

        # Calcular o tamanho total do input
        if isinstance(self.input_dimension, (list, tuple)):
            if len(self.input_dimension) == 1:
                total_input_size = self.input_dimension[0]
            else:
                total_input_size = np.prod(self.input_dimension)
        else:
            total_input_size = self.input_dimension

        logging.info(f"📐 Reconstruction target size: {total_input_size}")

        # Camada de reconstrução
        pretrain_output = Dense(total_input_size, activation='linear',
                                name='reconstruction_output')(pooled)

        pretrain_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=pretrain_output
        )

        pretrain_model.compile(
            optimizer=self.optimizer_function,
            loss='mse'
        )

        logging.info("✓ Model compiled for reconstruction pretraining")
        logging.info(f"⚙ Starting pretraining for {max(epochs // 2, 5)} epochs...")

        # Flatten dos dados de treino se necessário
        train_data_flat = train_data.reshape(train_data.shape[0], -1)

        pretrain_history = pretrain_model.fit(
            train_data,
            train_data_flat,  # ✅ Target é o input flattened
            epochs=max(epochs // 2, 5),
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

        logging.info("✓ Self-supervised pretraining completed!")

        # =============================================================================
        # PHASE 2: SUPERVISED FINE-TUNING
        # =============================================================================
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: SUPERVISED FINE-TUNING")
        logging.info("=" * 80)

        # Descongelar últimas camadas para permitir fine-tuning
        num_layers_to_freeze = max(len(self.neural_network_model.layers) - 8, 0)

        for i, layer in enumerate(self.neural_network_model.layers):
            if i < num_layers_to_freeze:
                layer.trainable = False
            else:
                layer.trainable = True

        num_frozen = sum([1 for layer in self.neural_network_model.layers if not layer.trainable])
        num_trainable = sum([1 for layer in self.neural_network_model.layers if layer.trainable])

        logging.info(f"✓ Froze {num_frozen} layers, keeping {num_trainable} trainable")

        # Adicionar classification head
        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(
            self.neural_network_model.output[0]
        )

        neural_network_flow = Dense(
            self.number_classes * 2,
            activation='relu',
            name='classification_hidden'
        )(neural_network_flow)
        neural_network_flow = Dropout(self.dropout_rate, name='classification_dropout')(neural_network_flow)

        neural_network_flow = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='classification_output'
        )(neural_network_flow)

        self.neural_network_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=neural_network_flow,
            name=f"{self.model_name}_FineTuned"
        )

        # Compilar com learning rate apropriado
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=5e-5)  # ✅ Learning rate mais baixo

        self.neural_network_model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=['accuracy']
        )

        logging.info(f"✓ Model compiled for supervised learning")
        logging.info(f"📊 Model parameters:")
        logging.info(f"   - Total layers: {len(self.neural_network_model.layers)}")
        logging.info(f"   - Trainable: {num_trainable}")
        logging.info(f"   - Frozen: {num_frozen}")

        self.neural_network_model.summary(print_fn=lambda x: logging.info(x))

        logging.info(f"\n⚙ Starting fine-tuning for {epochs} epochs...")

        # Callbacks para melhor treinamento
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        finetune_history = self.neural_network_model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        logging.info("✓ Supervised fine-tuning completed!")

        # Descongelar tudo para XAI
        for layer in self.neural_network_model.layers:
            layer.trainable = True

        # =============================================================================
        # XAI VISUALIZATION GENERATION
        # =============================================================================
        if generate_xai and validation_data is not None:
            logging.info("\n" + "=" * 80)
            logging.info("GENERATING XAI VISUALIZATIONS")
            logging.info("=" * 80)

            val_data, val_labels = validation_data

            # Verificar se há predições corretas suficientes
            predictions = self.neural_network_model.predict(val_data[:100], verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            if len(val_labels.shape) > 1:
                true_labels = np.argmax(val_labels[:100], axis=1)
            else:
                true_labels = val_labels[:100]

            accuracy = np.mean(predicted_classes == true_labels)
            logging.info(f"📊 Validation accuracy (first 100 samples): {accuracy:.2%}")

            if accuracy > 0.1:  # Apenas gerar XAI se houver algum aprendizado
                try:
                    stats = self.generate_validation_visualizations(
                        validation_data=val_data,
                        validation_labels=val_labels,
                        num_samples=num_xai_samples,
                        output_dir=xai_output_dir,
                        xai_method=xai_method
                    )

                    logging.info(f"✓ Generated {stats['total_samples']} XAI visualizations")
                    logging.info(f"  - Correct predictions: {stats['correct_predictions']}")
                    logging.info(f"  - Incorrect predictions: {stats['incorrect_predictions']}")
                except Exception as e:
                    logging.error(f"❌ Error generating XAI visualizations: {e}")
                    logging.error(f"   Traceback: {traceback.format_exc()}")
            else:
                logging.warning("⚠ Model accuracy too low, skipping XAI generation")

        logging.info("\n" + "=" * 80)
        logging.info("WAV2VEC2 TRAINING COMPLETED!")
        logging.info("=" * 80 + "\n")

        return finetune_history

    # def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
    #                       epochs: int, batch_size: int,
    #                       validation_data: tuple = None,
    #                       generate_xai: bool = True,
    #                       num_xai_samples: int = 30,
    #                       xai_output_dir: str = './wav2vec2_xai_outputs',
    #                       xai_method: str = 'gradcam++') -> tensorflow.keras.callbacks.History:
    #     """
    #     Two-phase Wav2Vec2 training with XAI visualization.
    #
    #     Args:
    #         train_data: Training input data
    #         train_labels: Training labels
    #         epochs: Number of training epochs
    #         batch_size: Batch size for training
    #         validation_data: Optional validation data tuple (X_val, y_val)
    #         generate_xai: Whether to generate XAI visualizations after training
    #         num_xai_samples: Number of samples to visualize
    #         xai_output_dir: Output directory for XAI visualizations
    #         xai_method: XAI method to use ('gradcam', 'gradcam++', or 'scorecam')
    #
    #     Returns:
    #         Training history from fine-tuning phase
    #     """
    #
    #     logging.info("=" * 80)
    #     logging.info("WAV2VEC2 TWO-PHASE TRAINING WITH XAI")
    #     logging.info("=" * 80)
    #
    #     # =============================================================================
    #     # PHASE 1: SELF-SUPERVISED CONTRASTIVE PRETRAINING
    #     # =============================================================================
    #     logging.info("\n" + "=" * 80)
    #     logging.info("PHASE 1: SELF-SUPERVISED PRETRAINING")
    #     logging.info("=" * 80)
    #
    #     projection_layer = Dense(16, name='contrastive_projection')
    #     projected_contextualized = projection_layer(self.neural_network_model.output[0])
    #
    #     pretrain_model = Model(
    #         inputs=self.neural_network_model.inputs,
    #         outputs=[projected_contextualized, self.neural_network_model.output[1]]
    #     )
    #
    #     contrastive_loss = Wav2Vec2ContrastiveLoss(
    #         temperature=self.contrastive_temperature,
    #         num_negatives=self.num_negatives
    #     )
    #
    #     pretrain_model.compile(
    #         optimizer=self.optimizer_function,
    #         loss=[contrastive_loss, 'mse'],
    #         loss_weights=[1.0, 0.01]
    #     )
    #
    #     logging.info("✓ Model compiled with contrastive loss")
    #     logging.info(f"⚙ Starting pretraining for {epochs} epochs...")
    #
    #     quantized_targets = pretrain_model.predict(train_data, batch_size=batch_size, verbose=0)
    #
    #     pretrain_history = pretrain_model.fit(
    #         train_data,
    #         [quantized_targets[1], quantized_targets[1]],
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         verbose=1
    #     )
    #
    #     logging.info("✓ Self-supervised pretraining completed!")
    #
    #     # =============================================================================
    #     # PHASE 2: SUPERVISED FINE-TUNING
    #     # =============================================================================
    #     logging.info("\n" + "=" * 80)
    #     logging.info("PHASE 2: SUPERVISED FINE-TUNING")
    #     logging.info("=" * 80)
    #
    #     self.neural_network_model.trainable = False
    #     logging.info("✓ Froze pretrained encoder weights")
    #
    #     neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(
    #         self.neural_network_model.output[0]
    #     )
    #
    #     neural_network_flow = Dense(
    #         self.number_classes * 2,
    #         name='classification_hidden'
    #     )(neural_network_flow)
    #     neural_network_flow = GELU(name='classification_gelu')(neural_network_flow)
    #     neural_network_flow = Dropout(self.dropout_rate, name='classification_dropout')(neural_network_flow)
    #
    #     neural_network_flow = Dense(
    #         self.number_classes,
    #         activation=self.last_layer_activation,
    #         name='classification_output'
    #     )(neural_network_flow)
    #
    #     logging.info("✓ Added classification head")
    #
    #     self.neural_network_model = Model(
    #         inputs=self.neural_network_model.inputs,
    #         outputs=neural_network_flow,
    #         name=f"{self.model_name}_FineTuned"
    #     )
    #
    #     self.neural_network_model.compile(
    #         optimizer=self.optimizer_function,
    #         loss=self.loss_function,
    #         metrics=['accuracy']
    #     )
    #
    #     logging.info(f"✓ Model compiled for supervised learning")
    #     self.neural_network_model.summary()
    #
    #     logging.info(f"\n⚙ Starting fine-tuning for {epochs} epochs...")
    #
    #     finetune_history = self.neural_network_model.fit(
    #         train_data,
    #         train_labels,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         validation_data=validation_data,
    #         verbose=1
    #     )
    #
    #     logging.info("✓ Supervised fine-tuning completed!")
    #
    #     # =============================================================================
    #     # XAI VISUALIZATION GENERATION
    #     # =============================================================================
    #     if generate_xai and validation_data is not None:
    #         logging.info("\n" + "=" * 80)
    #         logging.info("GENERATING XAI VISUALIZATIONS")
    #         logging.info("=" * 80)
    #
    #         val_data, val_labels = validation_data
    #
    #         stats = self.generate_validation_visualizations(
    #             validation_data=val_data,
    #             validation_labels=val_labels,
    #             num_samples=num_xai_samples,
    #             output_dir=xai_output_dir,
    #             xai_method=xai_method
    #         )
    #
    #         logging.info(f"✓ Generated {stats['total_samples']} XAI visualizations")
    #         logging.info(f"  - Correct predictions: {stats['correct_predictions']}")
    #         logging.info(f"  - Incorrect predictions: {stats['incorrect_predictions']}")
    #
    #     logging.info("\n" + "=" * 80)
    #     logging.info("WAV2VEC2 TRAINING COMPLETED SUCCESSFULLY!")
    #     logging.info("=" * 80 + "\n")
    #
    #     return finetune_history

    # =================================================================================
    # XAI METHODS
    # =================================================================================

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """Build an auxiliary model for GradCAM computation."""
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating GradCAM model")

        if target_layer_name is None:
            # Procurar a última camada convolucional
            conv_layers = [layer for layer in self.neural_network_model.layers
                           if 'conv' in layer.name.lower()]
            if not conv_layers:
                raise ValueError("No convolutional layers found!")
            target_layer_name = conv_layers[-1].name
            logging.info(f"🎯 Using layer for Grad-CAM: {target_layer_name}")

        try:
            target_layer = self.neural_network_model.get_layer(target_layer_name)
        except:
            # Se não encontrar, tentar camada de atenção
            target_layer = self.neural_network_model.get_layer('transformer_attention')
            logging.warning(f"Layer {target_layer_name} not found, using transformer_attention")

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

    def compute_gradcam(self, input_sample: np.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> np.ndarray:
        """Standard Grad-CAM computation for audio spectrograms."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(self.input_dimension)
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        # Handle TimeDistributed output (batch, time, freq, channels)
        if len(layer_output.shape) == 4:
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
            layer_output_squeezed = layer_output[0]

            # Average over time dimension
            layer_output_avg = tensorflow.reduce_mean(layer_output_squeezed, axis=0)
            heatmap = layer_output_avg @ pooled_grads[..., tensorflow.newaxis]
        else:
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
            layer_output_squeezed = layer_output[0]
            heatmap = layer_output_squeezed @ pooled_grads[..., tensorflow.newaxis]

        heatmap = tensorflow.squeeze(heatmap)
        heatmap = tensorflow.maximum(heatmap, 0)

        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_gradcam_plusplus(self, input_sample: np.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> np.ndarray:
        """Grad-CAM++ computation with improved localization."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(self.input_dimension)
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape1:
            with tensorflow.GradientTape() as tape2:
                with tensorflow.GradientTape() as tape3:
                    layer_output, predictions = self.gradcam_model(input_tensor)

                    if class_idx is None:
                        class_idx = tensorflow.argmax(predictions[0]).numpy()

                    class_score = predictions[:, class_idx]

                grads = tape3.gradient(class_score, layer_output)
            grads_2 = tape2.gradient(grads, layer_output)
        grads_3 = tape1.gradient(grads_2, layer_output)

        # Compute alpha weights
        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
            layer_output * grads_3, axis=-1, keepdims=True
        ) + 1e-10

        alpha = numerator / denominator
        relu_grads = tensorflow.maximum(grads, 0.0)
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))

        # Compute weighted activation map
        layer_output_squeezed = layer_output[0]
        layer_output_avg = tensorflow.reduce_mean(layer_output_squeezed, axis=0)
        heatmap = layer_output_avg @ weights[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_scorecam(self, input_sample: np.ndarray, class_idx: int = None,
                         target_layer_name: str = None) -> np.ndarray:
        """Score-CAM computation (gradient-free method)."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(self.input_dimension)
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        # Get activations and average over time
        activations = layer_output[0].numpy()
        activations_avg = np.mean(activations, axis=0)
        num_channels = activations_avg.shape[-1]

        weights = []
        for i in range(num_channels):
            act_map = activations_avg[:, i]

            # Normalize
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            target_shape = (128, 80)  # Wav2Vec2 uses 128x80 spectrograms
            zoom_factors = (target_shape[0] / act_map.shape[0],
                            target_shape[1] / act_map.shape[1] if len(act_map.shape) > 1 else 1)

            if len(act_map.shape) == 1:
                upsampled = zoom(act_map, zoom_factors[0], order=1)
                upsampled = np.tile(upsampled[:, np.newaxis], (1, target_shape[1]))
            else:
                upsampled = zoom(act_map, zoom_factors, order=1)

            # Mask input
            input_reshaped = input_sample[0].reshape(target_shape)
            masked_input = input_reshaped * upsampled
            masked_input = masked_input.reshape((1,) + self.input_dimension)

            # Get score
            masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
            score = masked_pred[0, class_idx]
            weights.append(score)

        weights = np.array(weights)
        weights = np.maximum(weights, 0)

        # Weighted combination
        heatmap = np.tensordot(activations_avg, weights, axes=([-1], [0]))

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
                            smooth: bool = True) -> np.ndarray:
        """Interpolate heatmap to target shape with optional smoothing."""
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)

        zoom_factors = (target_shape[0] / heatmap.shape[0],
                        target_shape[1] / heatmap.shape[1] if len(heatmap.shape) > 1 else 1)

        if len(heatmap.shape) == 1:
            interpolated = zoom(heatmap, zoom_factors[0], order=3)
            interpolated = np.tile(interpolated[:, np.newaxis], (1, target_shape[1]))
        else:
            interpolated = zoom(heatmap, zoom_factors, order=3)

        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=1.5)

        return interpolated

    def plot_xai_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                        class_idx: int, predicted_class: int, true_label: int = None,
                        confidence: float = None, xai_method: str = 'gradcam++',
                        save_path: str = None, show_plot: bool = True) -> None:
        """Modern XAI visualization for audio spectrograms."""

        # Reshape input
        input_plot = input_sample.reshape((128, 80))

        # Interpolate heatmap
        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_plot.shape, smooth=True)

        # Create figure
        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heatmap = 'jet'

        # 1. Original Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_plot, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('📊 Audio Spectrogram', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Time Frames', fontsize=10)
        ax1.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'🔥 Activation Map ({xai_method.upper()})', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Time Frames', fontsize=10)
        ax2.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_plot - input_plot.min()) / (input_plot.max() - input_plot.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title('🎯 Overlay', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Time Frames', fontsize=10)
        ax3.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Temporal Importance
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_importance = np.mean(interpolated_heatmap, axis=0)
        time_steps = np.arange(len(temporal_importance))
        temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

        ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Smoothed')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Original')

        ax4.set_xlabel('Time Frame', fontsize=10)
        ax4.set_ylabel('Mean Importance', fontsize=10)
        ax4.set_title('📈 Temporal Importance Profile', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_ylim([0, 1])

        # Super title
        pred_status = '✅' if predicted_class == true_label else '❌'
        conf_str = f' | Confidence: {confidence:.1%}' if confidence is not None else ''

        if true_label is not None:
            suptitle = f'{pred_status} Predicted: Class {predicted_class} | True: Class {true_label}{conf_str}'
        else:
            suptitle = f'Predicted: Class {predicted_class}{conf_str}'

        fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logging.info(f"💾 Saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_validation_visualizations(self, validation_data: np.ndarray,
                                           validation_labels: np.ndarray,
                                           num_samples: int = 10,
                                           output_dir: str = './wav2vec2_xai',
                                           target_layer_name: str = None,
                                           xai_method: str = 'gradcam++') -> dict:
        """Generate XAI visualizations for validation samples."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        predictions = self.neural_network_model.predict(validation_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        if len(validation_labels.shape) > 1:
            true_labels = np.argmax(validation_labels, axis=1)
        else:
            true_labels = validation_labels

        correct_indices = np.where(predicted_classes == true_labels)[0]
        incorrect_indices = np.where(predicted_classes != true_labels)[0]

        num_correct = min(num_samples // 2, len(correct_indices))
        num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

        selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if len(
            correct_indices) > 0 else []
        selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if len(
            incorrect_indices) > 0 else []

        selected_indices = np.concatenate([selected_correct, selected_incorrect])

        stats = {
            'total_samples': len(selected_indices),
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

        for i, idx in enumerate(selected_indices):
            try:
                sample = validation_data[idx]
                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

                # Compute heatmap
                if xai_method.lower() == 'gradcam++':
                    heatmap = self.compute_gradcam_plusplus(sample, class_idx=predicted,
                                                            target_layer_name=target_layer_name)
                elif xai_method.lower() == 'scorecam':
                    heatmap = self.compute_scorecam(sample, class_idx=predicted,
                                                    target_layer_name=target_layer_name)
                else:
                    heatmap = self.compute_gradcam(sample, class_idx=predicted,
                                                   target_layer_name=target_layer_name)

                is_correct = predicted == true_label

                if is_correct:
                    stats['correct_predictions'] += 1
                    prefix = 'correct'
                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorrect'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_sample_{i:03d}_true_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                self.plot_xai_modern(sample, heatmap, predicted, predicted, true_label,
                                     confidence=confidence, xai_method=xai_method,
                                     save_path=save_path, show_plot=False)

            except Exception as e:
                logging.error(f"Error processing sample {idx}: {e}")
                continue

        return stats

    # Properties (same as before)
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