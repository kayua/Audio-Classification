"""
===================================================================================
ARQUIVO 4: Wav2Vec2 Main Model - VERSÃO COMPLETA CORRIGIDA
===================================================================================
"""
from Engine.Layers.MaskTimeLayer import TimeMaskingWithStorage
from Engine.Models.Process.Wav2Vec2_Process import Wav2Vec2Process
from Engine.Modules.GumbelVectorQuantizer import GumbelVectorQuantizer

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


class Wav2Vec2DynamicTrainingModel(tf.keras.Model):
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

        with tf.GradientTape() as tape:
            # Forward pass
            contextualized, quantized_tuple = self(x, training=True)
            quantized, perplexity = quantized_tuple

            # Get mask indices
            mask_layer = None
            for layer in self.encoder_model.layers:
                if isinstance(layer, TimeMaskingWithStorage):
                    mask_layer = layer
                    break

            if mask_layer is not None and hasattr(mask_layer, '_last_mask_indices'):
                mask_indices = mask_layer._last_mask_indices
            else:
                mask_indices = tf.ones(tf.shape(contextualized)[:2], dtype=tf.bool)

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


class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
    """
    Complete Wav2Vec2 Implementation with all corrections applied.

    CORRECTED FEATURES:
    ✅ InfoNCE contrastive loss with negative sampling
    ✅ Diversity loss for codebook usage
    ✅ Dynamic training (targets computed per batch)
    ✅ Loss computed only on masked positions
    ✅ Encoder adjustable during fine-tuning
    ✅ Proper perplexity calculation (scalar)

    XAI FEATURES:
    ✅ Grad-CAM, Grad-CAM++, Score-CAM
    ✅ Modern visualizations
    ✅ Automatic validation generation
    """

    def __init__(self, arguments):
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
        """Build Wav2Vec2 architecture."""

        inputs = Input(shape=self.input_dimension, name='audio_input')
        neural_network_flow = Reshape((128, 80, 1), name='reshape_input')(inputs)

        # Convolutional encoder
        for idx, number_filters in enumerate(self.list_filters_encoder):
            neural_network_flow = TimeDistributed(Conv1D(
                number_filters,
                self.kernel_size,
                strides=(2,),
                use_bias=True,
                name=f'conv1d_encoder_{idx}'
            ), name=f'time_dist_conv_{idx}')(neural_network_flow)

            neural_network_flow = TimeDistributed(
                GELU(), name=f'time_dist_gelu_{idx}'
            )(neural_network_flow)
            neural_network_flow = TimeDistributed(
                LayerNormalization(), name=f'time_dist_ln_{idx}'
            )(neural_network_flow)

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

        # Time masking with storage
        masking_layer = TimeMaskingWithStorage(
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

        # Feedforward
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
            outputs=[transformer_output, (quantized_output, perplexity)],
            name=self.model_name
        )

        self.neural_network_model.summary()
        logging.info("✓ Wav2Vec2 architecture built")

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int,
                          validation_data: tuple = None,
                          generate_xai: bool = True,
                          num_xai_samples: int = 30,
                          xai_output_dir: str = './wav2vec2_xai_outputs',
                          xai_method: str = 'gradcam++',
                          freeze_encoder: bool = False) -> tensorflow.keras.callbacks.History:
        """Two-phase training with all corrections."""

        logging.info("=" * 80)
        logging.info("WAV2VEC2 TRAINING (CORRECTED)")
        logging.info("=" * 80)

        # Phase 1: Pretraining
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: SELF-SUPERVISED PRETRAINING")
        logging.info("=" * 80)

        # Create contrastive loss
        contrastive_loss = Wav2Vec2ContrastiveLoss(
            temperature=self.contrastive_temperature,
            num_negatives=self.num_negatives,
            diversity_weight=self.diversity_weight
        )

        # Wrap model with dynamic training model, passing the loss function
        pretrain_model = Wav2Vec2DynamicTrainingModel(
            self.neural_network_model,
            loss_fn=contrastive_loss,  # Pass loss function
            name='pretrain_model'
        )

        # Compile model (optimizer only, loss is handled in train_step)
        pretrain_model.compile(optimizer=self.optimizer_function)

        logging.info("✓ Compiled with InfoNCE + diversity loss")
        logging.info(f"⚙ Starting pretraining for {epochs} epochs...")

        pretrain_history = pretrain_model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        logging.info("✓ Pretraining completed!")

        # Phase 2: Fine-tuning
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: SUPERVISED FINE-TUNING")
        logging.info("=" * 80)

        if freeze_encoder:
            self.neural_network_model.trainable = False
            logging.info("✓ Froze encoder")
        else:
            self.neural_network_model.trainable = True
            logging.info("✓ Encoder trainable")

        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(
            self.neural_network_model.output[0]
        )

        neural_network_flow = Dense(
            self.number_classes * 2,
            name='classification_hidden'
        )(neural_network_flow)
        neural_network_flow = GELU(name='classification_gelu')(neural_network_flow)
        neural_network_flow = Dropout(self.dropout_rate, name='classification_dropout')(neural_network_flow)

        neural_network_flow = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='classification_output'
        )(neural_network_flow)

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

        logging.info("✓ Fine-tuning completed!")

        # XAI Generation
        if generate_xai and validation_data is not None:
            logging.info("\n" + "=" * 80)
            logging.info("GENERATING XAI VISUALIZATIONS")
            logging.info("=" * 80)

            val_data, val_labels = validation_data

            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=num_xai_samples,
                output_dir=xai_output_dir,
                xai_method=xai_method
            )

            logging.info(f"✓ Generated {stats['total_samples']} visualizations")

        logging.info("\n" + "=" * 80)
        logging.info("TRAINING COMPLETED")
        logging.info("=" * 80 + "\n")

        return finetune_history

    # XAI methods remain the same...
    # (Include all the XAI methods from the corrected version)

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        if self.neural_network_model is None:
            raise ValueError("Model must be built first")

        if target_layer_name is None:
            conv_layers = [l for l in self.neural_network_model.layers if 'conv' in l.name.lower()]
            if not conv_layers:
                raise ValueError("No conv layers found")
            target_layer_name = conv_layers[-1].name

        try:
            target_layer = self.neural_network_model.get_layer(target_layer_name)
        except:
            target_layer = self.neural_network_model.get_layer('transformer_attention')

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

    def compute_gradcam(self, input_sample: np.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> np.ndarray:
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

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

        if len(layer_output.shape) == 4:
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
            layer_output_squeezed = layer_output[0]
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
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

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

        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
            layer_output * grads_3, axis=-1, keepdims=True
        ) + 1e-10

        alpha = numerator / denominator
        relu_grads = tensorflow.maximum(grads, 0.0)
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))

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
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(self.input_dimension)
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        activations = layer_output[0].numpy()
        activations_avg = np.mean(activations, axis=0)
        num_channels = activations_avg.shape[-1]

        weights = []
        for i in range(num_channels):
            act_map = activations_avg[:, i]

            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            target_shape = (128, 80)
            zoom_factors = (target_shape[0] / act_map.shape[0],
                            target_shape[1] / act_map.shape[1] if len(act_map.shape) > 1 else 1)

            if len(act_map.shape) == 1:
                upsampled = zoom(act_map, zoom_factors[0], order=1)
                upsampled = np.tile(upsampled[:, np.newaxis], (1, target_shape[1]))
            else:
                upsampled = zoom(act_map, zoom_factors, order=1)

            input_reshaped = input_sample[0].reshape(target_shape)
            masked_input = input_reshaped * upsampled
            masked_input = masked_input.reshape((1,) + self.input_dimension)

            masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
            score = masked_pred[0, class_idx]
            weights.append(score)

        weights = np.array(weights)
        weights = np.maximum(weights, 0)

        heatmap = np.tensordot(activations_avg, weights, axes=([-1], [0]))

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
                            smooth: bool = True) -> np.ndarray:
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

        input_plot = input_sample.reshape((128, 80))
        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_plot.shape, smooth=True)

        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_plot, cmap='viridis', aspect='auto', interpolation='bilinear')
        ax1.set_title('📊 Audio Spectrogram', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Time Frames', fontsize=10)
        ax1.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap='jet',
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'🔥 Activation Map ({xai_method.upper()})', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Time Frames', fontsize=10)
        ax2.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_plot - input_plot.min()) / (input_plot.max() - input_plot.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap='jet',
                         alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title('🎯 Overlay', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Time Frames', fontsize=10)
        ax3.set_ylabel('Frequency Bins', fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

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

    # Properties
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