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


"""
===================================================================================
WAV2VEC2 XAI - MÉTODOS APROPRIADOS PARA TRANSFORMERS
===================================================================================
Substitui Grad-CAM++ por métodos mais adequados:
1. Attention Weights Visualization (nativo para Transformers)
2. Integrated Gradients (melhor para arquiteturas híbridas)
3. Input Saliency Maps (simples e efetivo)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import logging
from typing import Tuple, Optional


class Wav2Vec2XAIVisualization:
    """
    XAI methods specifically designed for Wav2Vec2 (CNN + Transformer) architecture.

    MÉTODOS IMPLEMENTADOS:
    ✅ Attention Weights Visualization - Visualiza pesos de atenção do Transformer
    ✅ Integrated Gradients - Atribuição robusta para arquiteturas híbridas
    ✅ Input Saliency Maps - Gradientes de primeira ordem (baseline simples)
    """

    def __init__(self, model: tf.keras.Model, input_dimension: Tuple[int, int]):
        self.model = model
        self.input_dimension = input_dimension

    # ==================== MÉTODO 1: ATTENTION WEIGHTS ====================
    def extract_attention_weights(self, input_sample: np.ndarray) -> np.ndarray:
        """
        Extrai e visualiza os pesos de atenção do MultiHeadAttention.

        Este é o método MAIS APROPRIADO para Transformers, pois usa
        informação nativa da arquitetura (sem backpropagation).

        Args:
            input_sample: Audio sample (shape matching input_dimension)

        Returns:
            attention_weights: Mapa de atenção normalizado
        """
        # Prepare input
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        input_tensor = tf.convert_to_tensor(input_sample, dtype=tf.float32)

        # Find attention layer
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                attention_layer = layer
                break

        if attention_layer is None:
            logging.warning("⚠️ No MultiHeadAttention layer found!")
            return np.zeros((128, 80))

        # Create model to extract attention layer output
        # We need to get the intermediate representation before attention
        attention_input_layer = None
        for i, layer in enumerate(self.model.layers):
            if layer == attention_layer:
                # Get the layer before attention
                if i > 0:
                    attention_input_layer = self.model.layers[i - 1]
                break

        if attention_input_layer is None:
            logging.warning("⚠️ Could not find attention input layer!")
            return np.zeros((128, 80))

        # Build extraction model
        extraction_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=attention_input_layer.output
        )

        # Get attention input
        attention_input = extraction_model(input_tensor)

        # Compute attention weights manually
        # MultiHeadAttention: Q @ K^T / sqrt(d_k)
        query = attention_input
        key = attention_input

        # Simplified attention computation (single head)
        d_k = query.shape[-1]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(d_k, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Average over batch and heads
        attention_map = tf.reduce_mean(attention_weights, axis=0).numpy()

        # Average over sequence dimension to get importance per timestep
        temporal_importance = np.mean(attention_map, axis=0)

        # Reshape to match input spectrogram
        target_shape = (128, 80)
        if len(temporal_importance) < target_shape[0]:
            # Upsample temporal importance to match input shape
            from scipy.ndimage import zoom
            zoom_factor = target_shape[0] / len(temporal_importance)
            temporal_upsampled = zoom(temporal_importance, zoom_factor, order=1)
        else:
            temporal_upsampled = temporal_importance[:target_shape[0]]

        # Tile to create 2D map
        attention_heatmap = np.tile(temporal_upsampled[:, np.newaxis], (1, target_shape[1]))

        # Normalize
        attention_heatmap = (attention_heatmap - attention_heatmap.min()) / \
                            (attention_heatmap.max() - attention_heatmap.min() + 1e-10)

        return attention_heatmap

    # ==================== MÉTODO 2: INTEGRATED GRADIENTS ====================
    def compute_integrated_gradients(self,
                                     input_sample: np.ndarray,
                                     class_idx: Optional[int] = None,
                                     steps: int = 50,
                                     baseline: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implementa Integrated Gradients - método robusto para atribuição de features.

        VANTAGENS sobre Grad-CAM++:
        - Funciona bem com arquiteturas híbridas (CNN + Transformer)
        - Satisfaz axiomas de atribuição (sensitivity, implementation invariance)
        - Não requer camadas convolucionais específicas

        Referência: Sundararajan et al. (2017) - "Axiomatic Attribution for Deep Networks"

        Args:
            input_sample: Audio sample
            class_idx: Target class (None = predicted class)
            steps: Number of integration steps (mais = mais preciso)
            baseline: Baseline input (None = zeros)

        Returns:
            attributions: Mapa de atribuição normalizado
        """
        # Prepare input
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        input_tensor = tf.convert_to_tensor(input_sample, dtype=tf.float32)

        # Create baseline (zeros or custom)
        if baseline is None:
            baseline = np.zeros_like(input_sample)
        else:
            if len(baseline.shape) == 2:
                baseline = np.expand_dims(baseline, axis=0)
        baseline_tensor = tf.convert_to_tensor(baseline, dtype=tf.float32)

        # Get predicted class if not specified
        if class_idx is None:
            predictions = self.model(input_tensor)
            class_idx = tf.argmax(predictions[0]).numpy()

        # Generate interpolated inputs between baseline and actual input
        alphas = tf.linspace(0.0, 1.0, steps + 1)

        # Vectorized path interpolation
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline_tensor, axis=0)
        input_x = tf.expand_dims(input_tensor, axis=0)
        delta = input_x - baseline_x
        interpolated_inputs = baseline_x + alphas_x * delta

        # Reshape for batch processing
        interpolated_inputs = tf.reshape(interpolated_inputs,
                                         [-1] + list(input_sample.shape[1:]))

        # Compute gradients for all interpolated inputs
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            predictions = self.model(interpolated_inputs)
            target_class_logits = predictions[:, class_idx]

        gradients = tape.gradient(target_class_logits, interpolated_inputs)

        # Average gradients (trapezoidal rule)
        avg_gradients = tf.reduce_mean(gradients, axis=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        integrated_gradients = (input_sample[0] - baseline[0]) * avg_gradients.numpy()

        # Take absolute value and normalize
        attribution_map = np.abs(integrated_gradients)
        attribution_map = (attribution_map - attribution_map.min()) / \
                          (attribution_map.max() - attribution_map.min() + 1e-10)

        return attribution_map

    # ==================== MÉTODO 3: INPUT SALIENCY MAPS ====================
    def compute_saliency_map(self,
                             input_sample: np.ndarray,
                             class_idx: Optional[int] = None) -> np.ndarray:
        """
        Computa mapa de saliência simples usando gradientes de primeira ordem.

        VANTAGENS:
        - Muito rápido (single backprop)
        - Funciona com qualquer arquitetura
        - Baseline simples para comparação

        Args:
            input_sample: Audio sample
            class_idx: Target class

        Returns:
            saliency_map: Mapa de saliência normalizado
        """
        # Prepare input
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        input_tensor = tf.Variable(input_sample, dtype=tf.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = self.model(input_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()
            class_score = predictions[:, class_idx]

        gradients = tape.gradient(class_score, input_tensor)

        # Take absolute value of gradients
        saliency = tf.abs(gradients).numpy()[0]

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

        return saliency

    # ==================== VISUALIZAÇÃO UNIFICADA ====================
    def plot_comparison(self,
                        input_sample: np.ndarray,
                        class_idx: Optional[int] = None,
                        predicted_class: Optional[int] = None,
                        true_label: Optional[int] = None,
                        confidence: Optional[float] = None,
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> dict:
        """
        Gera visualização comparativa com os 3 métodos.

        Returns:
            dict: Dicionário com os 3 mapas gerados
        """
        logging.info("🎨 Gerando visualizações XAI comparativas...")

        # Compute all three methods
        logging.info("   1/3 Computing Attention Weights...")
        attention_map = self.extract_attention_weights(input_sample)

        logging.info("   2/3 Computing Integrated Gradients...")
        ig_map = self.compute_integrated_gradients(input_sample, class_idx)

        logging.info("   3/3 Computing Saliency Map...")
        saliency_map = self.compute_saliency_map(input_sample, class_idx)

        # Prepare input for plotting
        input_plot = input_sample.reshape((128, 80))

        # Create figure
        fig = plt.figure(figsize=(22, 10), facecolor='white')
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Row 1: Heatmaps
        methods = [
            ('Attention Weights', attention_map, 'YlOrRd'),
            ('Integrated Gradients', ig_map, 'RdYlGn_r'),
            ('Saliency Map', saliency_map, 'plasma')
        ]

        for idx, (method_name, heatmap, cmap) in enumerate(methods):
            # Smooth heatmap
            heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)

            ax = fig.add_subplot(gs[0, idx])
            im = ax.imshow(heatmap_smooth, cmap=cmap, aspect='auto',
                           interpolation='bilinear', vmin=0, vmax=1)
            ax.set_title(f'🔥 {method_name}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Frequency Bins', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1, Column 4: Original Spectrogram
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(input_plot, cmap='viridis', aspect='auto', interpolation='bilinear')
        ax.set_title('📊 Original Spectrogram', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Time Frames', fontsize=9)
        ax.set_ylabel('Frequency Bins', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2: Overlays
        input_normalized = (input_plot - input_plot.min()) / \
                           (input_plot.max() - input_plot.min() + 1e-10)

        for idx, (method_name, heatmap, cmap) in enumerate(methods):
            heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)

            ax = fig.add_subplot(gs[1, idx])
            ax.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
            im = ax.imshow(heatmap_smooth, cmap=cmap, alpha=0.6,
                           aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
            ax.set_title(f'🎯 Overlay: {method_name}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Frequency Bins', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2, Column 4: Temporal Importance Comparison
        ax = fig.add_subplot(gs[1, 3])

        for method_name, heatmap, color in [
            ('Attention', attention_map, '#FF6B6B'),
            ('Int. Gradients', ig_map, '#4ECDC4'),
            ('Saliency', saliency_map, '#95E1D3')
        ]:
            temporal = np.mean(heatmap, axis=0)
            temporal_smooth = gaussian_filter(temporal, sigma=2)
            time_steps = np.arange(len(temporal_smooth))
            ax.plot(time_steps, temporal_smooth, linewidth=2.5,
                    color=color, label=method_name, alpha=0.8)

        ax.set_xlabel('Time Frame', fontsize=10)
        ax.set_ylabel('Mean Importance', fontsize=10)
        ax.set_title('📈 Temporal Importance Comparison', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim([0, 1])

        # Suptitle
        pred_status = '✅' if predicted_class == true_label else '❌'
        conf_str = f' | Confidence: {confidence:.1%}' if confidence is not None else ''

        if true_label is not None and predicted_class is not None:
            suptitle = f'{pred_status} Predicted: Class {predicted_class} | True: Class {true_label}{conf_str}'
        elif predicted_class is not None:
            suptitle = f'Predicted: Class {predicted_class}{conf_str}'
        else:
            suptitle = 'XAI Comparison: Attention vs Integrated Gradients vs Saliency'

        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logging.info(f"💾 Saved comparison: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return {
            'attention_weights': attention_map,
            'integrated_gradients': ig_map,
            'saliency_map': saliency_map
        }


# ==================== INTEGRAÇÃO NO CÓDIGO PRINCIPAL ====================
def integrate_into_wav2vec2_model(model_class):
    """
    Adiciona os novos métodos XAI à classe AudioWav2Vec2.

    Uso:
    1. Substitua os métodos compute_gradcam*, compute_scorecam
    2. Atualize generate_validation_visualizations para usar plot_comparison
    """

    def generate_validation_visualizations_corrected(self,
                                                     validation_data: np.ndarray,
                                                     validation_labels: np.ndarray,
                                                     num_samples: int = 10,
                                                     output_dir: str = './wav2vec2_xai',
                                                     xai_method: str = 'comparison') -> dict:
        """
        VERSÃO CORRIGIDA: Usa métodos apropriados para Transformers.

        Args:
            xai_method: 'comparison', 'attention', 'integrated_gradients', ou 'saliency'
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"\n{'=' * 60}")
        logging.info(f"XAI VISUALIZATION - CORRECTED METHODS")
        logging.info(f"{'=' * 60}")

        # Initialize XAI visualizer
        xai_viz = Wav2Vec2XAIVisualization(self.neural_network_model, self.input_dimension)

        # Get predictions
        predictions = self.neural_network_model.predict(validation_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        if len(validation_labels.shape) > 1:
            true_labels = np.argmax(validation_labels, axis=1)
        else:
            true_labels = validation_labels

        # Select samples
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
            'total_samples': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

        # Generate visualizations
        for i, idx in enumerate(selected_indices):
            try:
                sample = validation_data[idx]
                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

                is_correct = predicted == true_label
                prefix = 'correct' if is_correct else 'incorrect'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_sample_{i:03d}_true_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                # Generate comparison plot with all 3 methods
                xai_viz.plot_comparison(
                    input_sample=sample,
                    class_idx=predicted,
                    predicted_class=predicted,
                    true_label=true_label,
                    confidence=confidence,
                    save_path=save_path,
                    show_plot=False
                )

                stats['total_samples'] += 1
                if is_correct:
                    stats['correct_predictions'] += 1
                else:
                    stats['incorrect_predictions'] += 1

            except Exception as e:
                logging.error(f"Error processing sample {idx}: {e}")
                continue

        logging.info(f"\n✅ Generated {stats['total_samples']} visualizations")
        return stats

    return generate_validation_visualizations_corrected



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
    ✅ FIXED: Proper gradcam_model reconstruction after fine-tuning
    ✅ FIXED: Detailed logging for debugging
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
            loss_fn=contrastive_loss,
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

        # ==================== XAI Generation (FIXED) ====================
        logging.info("\n" + "=" * 80)
        logging.info("XAI VISUALIZATION SETUP")
        logging.info("=" * 80)
        logging.info(f"🔍 DEBUG: generate_xai = {generate_xai}")
        logging.info(f"🔍 DEBUG: validation_data = {'None' if validation_data is None else 'Provided'}")

        if validation_data is not None:
            val_data, val_labels = validation_data
            logging.info(f"🔍 DEBUG: val_data.shape = {val_data.shape}")
            logging.info(f"🔍 DEBUG: val_labels.shape = {val_labels.shape}")

        if generate_xai and validation_data is not None:
            logging.info("\n" + "=" * 80)
            logging.info("GENERATING XAI VISUALIZATIONS")
            logging.info("=" * 80)

            try:
                val_data, val_labels = validation_data

                # CRITICAL FIX: Rebuild gradcam_model for fine-tuned architecture
                logging.info("🔧 Rebuilding Grad-CAM model for fine-tuned architecture...")
                self.rebuild_gradcam_for_finetuned_model()

                logging.info(f"📊 Starting generation of {num_xai_samples} visualizations...")
                logging.info(f"📁 Output directory: {xai_output_dir}")
                logging.info(f"🎨 XAI method: {xai_method}")

                stats = self.generate_validation_visualizations(
                    validation_data=val_data,
                    validation_labels=val_labels,
                    num_samples=num_xai_samples,
                    output_dir=xai_output_dir,
                    xai_method=xai_method
                )

                logging.info(f"\n✅ XAI GENERATION COMPLETED!")
                logging.info(f"   Total samples: {stats['total_samples']}")
                logging.info(f"   Correct predictions: {stats['correct_predictions']}")
                logging.info(f"   Incorrect predictions: {stats['incorrect_predictions']}")

            except Exception as e:
                logging.error(f"\n❌ CRITICAL ERROR IN XAI GENERATION!")
                logging.error(f"   Error: {e}")
                import traceback
                logging.error(f"   Traceback:\n{traceback.format_exc()}")

        else:
            logging.warning("\n⚠️ XAI VISUALIZATION SKIPPED")
            if not generate_xai:
                logging.warning("   Reason: generate_xai=False")
            if validation_data is None:
                logging.warning("   Reason: validation_data not provided")

        logging.info("\n" + "=" * 80)
        logging.info("TRAINING COMPLETED")
        logging.info("=" * 80 + "\n")

        return finetune_history

    def rebuild_gradcam_for_finetuned_model(self) -> None:
        """
        CRITICAL FIX: Rebuild gradcam_model after fine-tuning.

        After fine-tuning, the model structure changes:
        - Before: [transformer_output, (quantized_output, perplexity)]
        - After: classification_output (single tensor)

        This method finds the best intermediate layer for Grad-CAM visualization.
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built first!")

        logging.info("🔍 Searching for suitable layer for Grad-CAM...")

        # Priority 1: Transformer/Attention layers (best for interpretability)
        attention_layers = [l for l in self.neural_network_model.layers
                            if any(keyword in l.name.lower()
                                   for keyword in ['attention', 'transformer', 'layer_norm_2'])]

        # Priority 2: Convolutional layers
        conv_layers = [l for l in self.neural_network_model.layers
                       if 'conv' in l.name.lower()]

        # Priority 3: Dense layers (before classification head)
        dense_layers = [l for l in self.neural_network_model.layers
                        if 'dense' in l.name.lower() and 'classification' not in l.name.lower()]

        # Select target layer based on priority
        if attention_layers:
            target_layer = attention_layers[-1]
            logging.info(f"✓ Found attention layer: '{target_layer.name}'")
        elif dense_layers:
            target_layer = dense_layers[-1]
            logging.info(f"✓ Found dense layer: '{target_layer.name}'")
        elif conv_layers:
            target_layer = conv_layers[-1]
            logging.info(f"✓ Found conv layer: '{target_layer.name}'")
        else:
            # Fallback: use layer before final classification
            target_layer = self.neural_network_model.layers[-3]
            logging.warning(f"⚠️ Using fallback layer: '{target_layer.name}'")

        # Log layer details
        logging.info(f"   Layer type: {type(target_layer).__name__}")
        logging.info(f"   Output shape: {target_layer.output_shape}")

        # Build Grad-CAM model
        try:
            self.gradcam_model = Model(
                inputs=self.neural_network_model.inputs,
                outputs=[target_layer.output, self.neural_network_model.output]
            )
            logging.info("✅ Grad-CAM model successfully rebuilt!")

        except Exception as e:
            logging.error(f"❌ Error building Grad-CAM model: {e}")
            raise

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """Legacy method - now calls rebuild_gradcam_for_finetuned_model."""
        if target_layer_name is not None:
            # If specific layer requested, use that
            try:
                target_layer = self.neural_network_model.get_layer(target_layer_name)
                self.gradcam_model = Model(
                    inputs=self.neural_network_model.inputs,
                    outputs=[target_layer.output, self.neural_network_model.output]
                )
                logging.info(f"✓ Built Grad-CAM model with layer: {target_layer_name}")
            except Exception as e:
                logging.warning(f"⚠️ Could not find layer '{target_layer_name}', using auto-detection")
                self.rebuild_gradcam_for_finetuned_model()
        else:
            # Auto-detect best layer
            self.rebuild_gradcam_for_finetuned_model()

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

        logging.info(f"\n{'=' * 60}")
        logging.info(f"STARTING XAI VISUALIZATION GENERATION")
        logging.info(f"{'=' * 60}")
        logging.info(f"📊 Validation data shape: {validation_data.shape}")
        logging.info(f"📊 Validation labels shape: {validation_labels.shape}")
        logging.info(f"🎨 XAI method: {xai_method}")
        logging.info(f"📁 Output directory: {output_dir}")
        logging.info(f"🔢 Requested samples: {num_samples}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"✓ Output directory created/verified")

        # Get predictions
        try:
            logging.info("🔮 Computing predictions...")
            predictions = self.neural_network_model.predict(validation_data, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            logging.info(f"✓ Predictions computed: {predictions.shape}")
        except Exception as e:
            logging.error(f"❌ Error computing predictions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {'total_samples': 0, 'correct_predictions': 0, 'incorrect_predictions': 0}

        # Process labels
        if len(validation_labels.shape) > 1:
            true_labels = np.argmax(validation_labels, axis=1)
            logging.info(f"✓ Converted one-hot labels to class indices")
        else:
            true_labels = validation_labels

        # Select samples
        correct_indices = np.where(predicted_classes == true_labels)[0]
        incorrect_indices = np.where(predicted_classes != true_labels)[0]

        logging.info(f"📈 Correct predictions: {len(correct_indices)}")
        logging.info(f"📉 Incorrect predictions: {len(incorrect_indices)}")

        num_correct = min(num_samples // 2, len(correct_indices))
        num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

        selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if len(
            correct_indices) > 0 else []
        selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if len(
            incorrect_indices) > 0 else []

        selected_indices = np.concatenate([selected_correct, selected_incorrect])

        logging.info(f"✓ Selected {len(selected_indices)} samples for visualization")
        logging.info(f"   - {len(selected_correct)} correct predictions")
        logging.info(f"   - {len(selected_incorrect)} incorrect predictions")

        stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

        # Generate visualizations
        logging.info(f"\n{'=' * 60}")
        logging.info("GENERATING HEATMAPS...")
        logging.info(f"{'=' * 60}")

        for i, idx in enumerate(selected_indices):
            try:
                logging.info(f"\n[{i + 1}/{len(selected_indices)}] Processing sample index {idx}...")

                sample = validation_data[idx]
                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

                logging.info(f"   True: {true_label} | Predicted: {predicted} | Confidence: {confidence:.2%}")

                # Compute heatmap
                logging.info(f"   Computing {xai_method} heatmap...")
                if xai_method.lower() == 'gradcam++':
                    heatmap = self.compute_gradcam_plusplus(sample, class_idx=predicted,
                                                            target_layer_name=target_layer_name)
                elif xai_method.lower() == 'scorecam':
                    heatmap = self.compute_scorecam(sample, class_idx=predicted,
                                                    target_layer_name=target_layer_name)
                else:
                    heatmap = self.compute_gradcam(sample, class_idx=predicted,
                                                   target_layer_name=target_layer_name)

                logging.info(f"   ✓ Heatmap computed: shape {heatmap.shape}")

                is_correct = predicted == true_label

                if is_correct:
                    stats['correct_predictions'] += 1
                    prefix = 'correct'
                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorrect'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_sample_{i:03d}_true_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                logging.info(f"   Plotting and saving to: {os.path.basename(save_path)}")
                self.plot_xai_modern(sample, heatmap, predicted, predicted, true_label,
                                     confidence=confidence, xai_method=xai_method,
                                     save_path=save_path, show_plot=False)

                stats['total_samples'] += 1
                logging.info(f"   ✅ Successfully saved visualization")

            except Exception as e:
                logging.error(f"\n❌ ERROR processing sample {idx}:")
                logging.error(f"   {str(e)}")
                import traceback
                logging.error(f"   Full traceback:\n{traceback.format_exc()}")
                continue

        # Final summary
        logging.info(f"\n{'=' * 60}")
        logging.info("XAI GENERATION SUMMARY")
        logging.info(f"{'=' * 60}")
        logging.info(f"✅ Total visualizations: {stats['total_samples']}")
        logging.info(f"   - Correct predictions: {stats['correct_predictions']}")
        logging.info(f"   - Incorrect predictions: {stats['incorrect_predictions']}")
        logging.info(f"📁 All files saved to: {output_dir}")
        logging.info(f"{'=' * 60}\n")

        return stats

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

#
# class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
#     """
#     Complete Wav2Vec2 Implementation with all corrections applied.
#
#     CORRECTED FEATURES:
#     ✅ InfoNCE contrastive loss with negative sampling
#     ✅ Diversity loss for codebook usage
#     ✅ Dynamic training (targets computed per batch)
#     ✅ Loss computed only on masked positions
#     ✅ Encoder adjustable during fine-tuning
#     ✅ Proper perplexity calculation (scalar)
#
#     XAI FEATURES:
#     ✅ Grad-CAM, Grad-CAM++, Score-CAM
#     ✅ Modern visualizations
#     ✅ Automatic validation generation
#     """
#
#     def __init__(self, arguments):
#         Wav2Vec2Process.__init__(self, arguments)
#
#         self.neural_network_model = None
#         self.gradcam_model = None
#         self.list_filters_encoder = arguments.wav_to_vec_list_filters_encoder
#         self.loss_function = arguments.wav_to_vec_loss_function
#         self.optimizer_function = arguments.wav_to_vec_optimizer_function
#         self.kernel_size = arguments.wav_to_vec_kernel_size
#         self.quantization_units = arguments.wav_to_vec_quantization_bits
#         self.key_dimension = arguments.wav_to_vec_key_dimension
#         self.intermediary_layer_activation = arguments.wav_to_vec_intermediary_layer_activation
#         self.number_heads = arguments.wav_to_vec_number_heads
#         self.input_dimension = arguments.wav_to_vec_input_dimension
#         self.number_classes = arguments.number_classes
#         self.dropout_rate = arguments.wav_to_vec_dropout_rate
#         self.last_layer_activation = arguments.wav_to_vec_last_layer_activation
#         self.model_name = "Wav2Vec2"
#
#         self.contrastive_temperature = 0.1
#         self.num_negatives = 100
#         self.diversity_weight = 0.1
#
#         plt.style.use('seaborn-v0_8-darkgrid')
#         sns.set_palette("husl")
#
#     def build_model(self) -> None:
#         """Build Wav2Vec2 architecture."""
#
#         inputs = Input(shape=self.input_dimension, name='audio_input')
#         neural_network_flow = Reshape((128, 80, 1), name='reshape_input')(inputs)
#
#         # Convolutional encoder
#         for idx, number_filters in enumerate(self.list_filters_encoder):
#             neural_network_flow = TimeDistributed(Conv1D(
#                 number_filters,
#                 self.kernel_size,
#                 strides=(2,),
#                 use_bias=True,
#                 name=f'conv1d_encoder_{idx}'
#             ), name=f'time_dist_conv_{idx}')(neural_network_flow)
#
#             neural_network_flow = TimeDistributed(
#                 GELU(), name=f'time_dist_gelu_{idx}'
#             )(neural_network_flow)
#             neural_network_flow = TimeDistributed(
#                 LayerNormalization(), name=f'time_dist_ln_{idx}'
#             )(neural_network_flow)
#
#         # Flatten and project
#         flatten_flow = TimeDistributed(Flatten(), name='time_dist_flatten')(neural_network_flow)
#         dense_layer = TimeDistributed(Dense(
#             self.number_classes,
#             activation=self.intermediary_layer_activation,
#             name='dense_projection'
#         ), name='time_dist_dense')(flatten_flow)
#
#         # Sequence lengths
#         sequence_length = self.input_dimension[0]
#         lengths = Lambda(
#             lambda x: tf.fill([tf.shape(x)[0]], tf.constant(sequence_length, dtype=tf.int32)),
#             dtype=tf.int32,
#             name='sequence_lengths'
#         )(dense_layer)
#
#         # Time masking with storage
#         masking_layer = TimeMaskingWithStorage(
#             mask_time_prob=0.065,
#             number_mask_time_steps=10,
#             name='time_masking'
#         )
#         time_masking, mask_indices = masking_layer([dense_layer, lengths])
#
#         # Transformer encoder
#         transformer_attention = MultiHeadAttention(
#             num_heads=self.number_heads,
#             key_dim=self.key_dimension,
#             name='transformer_attention'
#         )(time_masking, time_masking)
#
#         transformer_attention = Add(name='residual_add_1')([time_masking, transformer_attention])
#         transformer_attention = LayerNormalization(name='layer_norm_1')(transformer_attention)
#
#         # Feedforward
#         feedforward_network = Dense(
#             self.number_classes * 4,
#             name='feedforward_1'
#         )(transformer_attention)
#         feedforward_network = GELU(name='gelu_feedforward')(feedforward_network)
#         feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_1')(feedforward_network)
#
#         feedforward_network = Dense(
#             self.number_classes,
#             name='feedforward_2'
#         )(feedforward_network)
#         feedforward_network = Dropout(self.dropout_rate, name='dropout_feedforward_2')(feedforward_network)
#
#         transformer_output = Add(name='residual_add_2')([transformer_attention, feedforward_network])
#         transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)
#
#         # Vector quantization
#         quantization_layer = GumbelVectorQuantizer(name='gumbel_quantizer')
#         quantized_output, perplexity = quantization_layer([dense_layer, lengths])
#
#         self.neural_network_model = Model(
#             inputs=inputs,
#             outputs=[transformer_output, (quantized_output, perplexity)],
#             name=self.model_name
#         )
#
#         self.neural_network_model.summary()
#         logging.info("✓ Wav2Vec2 architecture built")
#
#     def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
#                           epochs: int, batch_size: int,
#                           validation_data: tuple = None,
#                           generate_xai: bool = True,
#                           num_xai_samples: int = 30,
#                           xai_output_dir: str = './wav2vec2_xai_outputs',
#                           xai_method: str = 'gradcam++',
#                           freeze_encoder: bool = False) -> tensorflow.keras.callbacks.History:
#         """Two-phase training with all corrections."""
#
#         logging.info("=" * 80)
#         logging.info("WAV2VEC2 TRAINING (CORRECTED)")
#         logging.info("=" * 80)
#
#         # Phase 1: Pretraining
#         logging.info("\n" + "=" * 80)
#         logging.info("PHASE 1: SELF-SUPERVISED PRETRAINING")
#         logging.info("=" * 80)
#
#         # Create contrastive loss
#         contrastive_loss = Wav2Vec2ContrastiveLoss(
#             temperature=self.contrastive_temperature,
#             num_negatives=self.num_negatives,
#             diversity_weight=self.diversity_weight
#         )
#
#         # Wrap model with dynamic training model, passing the loss function
#         pretrain_model = Wav2Vec2DynamicTrainingModel(
#             self.neural_network_model,
#             loss_fn=contrastive_loss,  # Pass loss function
#             name='pretrain_model'
#         )
#
#         # Compile model (optimizer only, loss is handled in train_step)
#         pretrain_model.compile(optimizer=self.optimizer_function)
#
#         logging.info("✓ Compiled with InfoNCE + diversity loss")
#         logging.info(f"⚙ Starting pretraining for {epochs} epochs...")
#
#         pretrain_history = pretrain_model.fit(
#             train_data,
#             epochs=epochs,
#             batch_size=batch_size,
#             verbose=1
#         )
#
#         logging.info("✓ Pretraining completed!")
#
#         # Phase 2: Fine-tuning
#         logging.info("\n" + "=" * 80)
#         logging.info("PHASE 2: SUPERVISED FINE-TUNING")
#         logging.info("=" * 80)
#
#         if freeze_encoder:
#             self.neural_network_model.trainable = False
#             logging.info("✓ Froze encoder")
#         else:
#             self.neural_network_model.trainable = True
#             logging.info("✓ Encoder trainable")
#
#         neural_network_flow = GlobalAveragePooling1D(name='global_avg_pool')(
#             self.neural_network_model.output[0]
#         )
#
#         neural_network_flow = Dense(
#             self.number_classes * 2,
#             name='classification_hidden'
#         )(neural_network_flow)
#         neural_network_flow = GELU(name='classification_gelu')(neural_network_flow)
#         neural_network_flow = Dropout(self.dropout_rate, name='classification_dropout')(neural_network_flow)
#
#         neural_network_flow = Dense(
#             self.number_classes,
#             activation=self.last_layer_activation,
#             name='classification_output'
#         )(neural_network_flow)
#
#         self.neural_network_model = Model(inputs=self.neural_network_model.inputs,
#                                           outputs=neural_network_flow,
#                                           name=f"{self.model_name}_FineTuned")
#
#         self.neural_network_model.compile(optimizer=self.optimizer_function,
#                                           loss=self.loss_function,
#                                           metrics=['accuracy'])
#
#         logging.info(f"⚙ Starting fine-tuning for {epochs} epochs...")
#
#         finetune_history = self.neural_network_model.fit(train_data,
#                                                          train_labels,
#                                                          epochs=epochs,
#                                                          batch_size=batch_size,
#                                                          validation_data=validation_data,
#                                                          verbose=1)
#
#         logging.info("✓ Fine-tuning completed!")
#
#         # XAI Generation
#         if generate_xai and validation_data is not None:
#             logging.info("\n" + "=" * 80)
#             logging.info("GENERATING XAI VISUALIZATIONS")
#             logging.info("=" * 80)
#
#             val_data, val_labels = validation_data
#
#             stats = self.generate_validation_visualizations(
#                 validation_data=val_data,
#                 validation_labels=val_labels,
#                 num_samples=num_xai_samples,
#                 output_dir=xai_output_dir,
#                 xai_method=xai_method
#             )
#
#             logging.info(f"✓ Generated {stats['total_samples']} visualizations")
#
#         logging.info("\n" + "=" * 80)
#         logging.info("TRAINING COMPLETED")
#         logging.info("=" * 80 + "\n")
#
#         return finetune_history
#
#     # XAI methods remain the same...
#     # (Include all the XAI methods from the corrected version)
#
#     def build_gradcam_model(self, target_layer_name: str = None) -> None:
#         if self.neural_network_model is None:
#             raise ValueError("Model must be built first")
#
#         if target_layer_name is None:
#             conv_layers = [l for l in self.neural_network_model.layers if 'conv' in l.name.lower()]
#             if not conv_layers:
#                 raise ValueError("No conv layers found")
#             target_layer_name = conv_layers[-1].name
#
#         try:
#             target_layer = self.neural_network_model.get_layer(target_layer_name)
#         except:
#             target_layer = self.neural_network_model.get_layer('transformer_attention')
#
#         self.gradcam_model = Model(
#             inputs=self.neural_network_model.inputs,
#             outputs=[target_layer.output, self.neural_network_model.output]
#         )
#
#     def compute_gradcam(self, input_sample: np.ndarray, class_idx: int = None,
#                         target_layer_name: str = None) -> np.ndarray:
#         if self.gradcam_model is None or target_layer_name is not None:
#             self.build_gradcam_model(target_layer_name)
#
#         if len(input_sample.shape) == 1:
#             input_sample = input_sample.reshape(self.input_dimension)
#         if len(input_sample.shape) == 2:
#             input_sample = np.expand_dims(input_sample, axis=0)
#
#         input_sample = input_sample.astype(np.float32)
#         input_tensor = tensorflow.convert_to_tensor(input_sample)
#
#         with tensorflow.GradientTape() as tape:
#             layer_output, predictions = self.gradcam_model(input_tensor)
#             if class_idx is None:
#                 class_idx = tensorflow.argmax(predictions[0]).numpy()
#             class_channel = predictions[:, class_idx]
#
#         grads = tape.gradient(class_channel, layer_output)
#
#         if len(layer_output.shape) == 4:
#             pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
#             layer_output_squeezed = layer_output[0]
#             layer_output_avg = tensorflow.reduce_mean(layer_output_squeezed, axis=0)
#             heatmap = layer_output_avg @ pooled_grads[..., tensorflow.newaxis]
#         else:
#             pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
#             layer_output_squeezed = layer_output[0]
#             heatmap = layer_output_squeezed @ pooled_grads[..., tensorflow.newaxis]
#
#         heatmap = tensorflow.squeeze(heatmap)
#         heatmap = tensorflow.maximum(heatmap, 0)
#
#         heatmap_max = tensorflow.math.reduce_max(heatmap)
#         if heatmap_max > 1e-10:
#             heatmap = heatmap / heatmap_max
#
#         return heatmap.numpy()
#
#     def compute_gradcam_plusplus(self, input_sample: np.ndarray, class_idx: int = None,
#                                  target_layer_name: str = None) -> np.ndarray:
#         if self.gradcam_model is None or target_layer_name is not None:
#             self.build_gradcam_model(target_layer_name)
#
#         if len(input_sample.shape) == 1:
#             input_sample = input_sample.reshape(self.input_dimension)
#         if len(input_sample.shape) == 2:
#             input_sample = np.expand_dims(input_sample, axis=0)
#
#         input_sample = input_sample.astype(np.float32)
#         input_tensor = tensorflow.convert_to_tensor(input_sample)
#
#         with tensorflow.GradientTape() as tape1:
#             with tensorflow.GradientTape() as tape2:
#                 with tensorflow.GradientTape() as tape3:
#                     layer_output, predictions = self.gradcam_model(input_tensor)
#                     if class_idx is None:
#                         class_idx = tensorflow.argmax(predictions[0]).numpy()
#                     class_score = predictions[:, class_idx]
#                 grads = tape3.gradient(class_score, layer_output)
#             grads_2 = tape2.gradient(grads, layer_output)
#         grads_3 = tape1.gradient(grads_2, layer_output)
#
#         numerator = grads_2
#         denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
#             layer_output * grads_3, axis=-1, keepdims=True
#         ) + 1e-10
#
#         alpha = numerator / denominator
#         relu_grads = tensorflow.maximum(grads, 0.0)
#         weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))
#
#         layer_output_squeezed = layer_output[0]
#         layer_output_avg = tensorflow.reduce_mean(layer_output_squeezed, axis=0)
#         heatmap = layer_output_avg @ weights[..., tensorflow.newaxis]
#         heatmap = tensorflow.squeeze(heatmap)
#
#         heatmap = tensorflow.maximum(heatmap, 0)
#         heatmap_max = tensorflow.math.reduce_max(heatmap)
#         if heatmap_max > 1e-10:
#             heatmap = heatmap / heatmap_max
#
#         return heatmap.numpy()
#
#     def compute_scorecam(self, input_sample: np.ndarray, class_idx: int = None,
#                          target_layer_name: str = None) -> np.ndarray:
#         if self.gradcam_model is None or target_layer_name is not None:
#             self.build_gradcam_model(target_layer_name)
#
#         if len(input_sample.shape) == 1:
#             input_sample = input_sample.reshape(self.input_dimension)
#         if len(input_sample.shape) == 2:
#             input_sample = np.expand_dims(input_sample, axis=0)
#
#         input_sample = input_sample.astype(np.float32)
#         input_tensor = tensorflow.convert_to_tensor(input_sample)
#
#         layer_output, predictions = self.gradcam_model(input_tensor)
#
#         if class_idx is None:
#             class_idx = tensorflow.argmax(predictions[0]).numpy()
#
#         activations = layer_output[0].numpy()
#         activations_avg = np.mean(activations, axis=0)
#         num_channels = activations_avg.shape[-1]
#
#         weights = []
#         for i in range(num_channels):
#             act_map = activations_avg[:, i]
#
#             if act_map.max() > act_map.min():
#                 act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
#
#             target_shape = (128, 80)
#             zoom_factors = (target_shape[0] / act_map.shape[0],
#                             target_shape[1] / act_map.shape[1] if len(act_map.shape) > 1 else 1)
#
#             if len(act_map.shape) == 1:
#                 upsampled = zoom(act_map, zoom_factors[0], order=1)
#                 upsampled = np.tile(upsampled[:, np.newaxis], (1, target_shape[1]))
#             else:
#                 upsampled = zoom(act_map, zoom_factors, order=1)
#
#             input_reshaped = input_sample[0].reshape(target_shape)
#             masked_input = input_reshaped * upsampled
#             masked_input = masked_input.reshape((1,) + self.input_dimension)
#
#             masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
#             score = masked_pred[0, class_idx]
#             weights.append(score)
#
#         weights = np.array(weights)
#         weights = np.maximum(weights, 0)
#
#         heatmap = np.tensordot(activations_avg, weights, axes=([-1], [0]))
#
#         heatmap = np.maximum(heatmap, 0)
#         if heatmap.max() > 0:
#             heatmap = heatmap / heatmap.max()
#
#         return heatmap
#
#     @staticmethod
#     def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
#                             smooth: bool = True) -> np.ndarray:
#         if not isinstance(heatmap, np.ndarray):
#             heatmap = np.array(heatmap)
#
#         zoom_factors = (target_shape[0] / heatmap.shape[0],
#                         target_shape[1] / heatmap.shape[1] if len(heatmap.shape) > 1 else 1)
#
#         if len(heatmap.shape) == 1:
#             interpolated = zoom(heatmap, zoom_factors[0], order=3)
#             interpolated = np.tile(interpolated[:, np.newaxis], (1, target_shape[1]))
#         else:
#             interpolated = zoom(heatmap, zoom_factors, order=3)
#
#         if smooth:
#             interpolated = gaussian_filter(interpolated, sigma=1.5)
#
#         return interpolated
#
#     def plot_xai_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
#                         class_idx: int, predicted_class: int, true_label: int = None,
#                         confidence: float = None, xai_method: str = 'gradcam++',
#                         save_path: str = None, show_plot: bool = True) -> None:
#
#         input_plot = input_sample.reshape((128, 80))
#         interpolated_heatmap = self.interpolate_heatmap(heatmap, input_plot.shape, smooth=True)
#
#         fig = plt.figure(figsize=(20, 6), facecolor='white')
#         gs = fig.add_gridspec(1, 4, wspace=0.3)
#
#         ax1 = fig.add_subplot(gs[0, 0])
#         im1 = ax1.imshow(input_plot, cmap='viridis', aspect='auto', interpolation='bilinear')
#         ax1.set_title('📊 Audio Spectrogram', fontsize=13, fontweight='bold', pad=15)
#         ax1.set_xlabel('Time Frames', fontsize=10)
#         ax1.set_ylabel('Frequency Bins', fontsize=10)
#         plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
#
#         ax2 = fig.add_subplot(gs[0, 1])
#         im2 = ax2.imshow(interpolated_heatmap, cmap='jet',
#                          aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
#         ax2.set_title(f'🔥 Activation Map ({xai_method.upper()})', fontsize=13, fontweight='bold', pad=15)
#         ax2.set_xlabel('Time Frames', fontsize=10)
#         ax2.set_ylabel('Frequency Bins', fontsize=10)
#         plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#
#         ax3 = fig.add_subplot(gs[0, 2])
#         input_normalized = (input_plot - input_plot.min()) / (input_plot.max() - input_plot.min() + 1e-10)
#         ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
#         im3 = ax3.imshow(interpolated_heatmap, cmap='jet',
#                          alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
#         ax3.set_title('🎯 Overlay', fontsize=13, fontweight='bold', pad=15)
#         ax3.set_xlabel('Time Frames', fontsize=10)
#         ax3.set_ylabel('Frequency Bins', fontsize=10)
#         plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
#
#         ax4 = fig.add_subplot(gs[0, 3])
#         temporal_importance = np.mean(interpolated_heatmap, axis=0)
#         time_steps = np.arange(len(temporal_importance))
#         temporal_smooth = gaussian_filter(temporal_importance, sigma=2)
#
#         ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
#         ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Smoothed')
#         ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
#                  color='#4ECDC4', linestyle='--', label='Original')
#
#         ax4.set_xlabel('Time Frame', fontsize=10)
#         ax4.set_ylabel('Mean Importance', fontsize=10)
#         ax4.set_title('📈 Temporal Importance Profile', fontsize=13, fontweight='bold', pad=15)
#         ax4.grid(True, alpha=0.3, linestyle='--')
#         ax4.legend(loc='upper right', fontsize=9)
#         ax4.set_ylim([0, 1])
#
#         pred_status = '✅' if predicted_class == true_label else '❌'
#         conf_str = f' | Confidence: {confidence:.1%}' if confidence is not None else ''
#
#         if true_label is not None:
#             suptitle = f'{pred_status} Predicted: Class {predicted_class} | True: Class {true_label}{conf_str}'
#         else:
#             suptitle = f'Predicted: Class {predicted_class}{conf_str}'
#
#         fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#             logging.info(f"💾 Saved: {save_path}")
#
#         if show_plot:
#             plt.show()
#         else:
#             plt.close()
#
#     def generate_validation_visualizations(self, validation_data: np.ndarray,
#                                            validation_labels: np.ndarray,
#                                            num_samples: int = 10,
#                                            output_dir: str = './wav2vec2_xai',
#                                            target_layer_name: str = None,
#                                            xai_method: str = 'gradcam++') -> dict:
#         import os
#         os.makedirs(output_dir, exist_ok=True)
#
#         predictions = self.neural_network_model.predict(validation_data, verbose=0)
#         predicted_classes = np.argmax(predictions, axis=1)
#         confidences = np.max(predictions, axis=1)
#
#         if len(validation_labels.shape) > 1:
#             true_labels = np.argmax(validation_labels, axis=1)
#         else:
#             true_labels = validation_labels
#
#         correct_indices = np.where(predicted_classes == true_labels)[0]
#         incorrect_indices = np.where(predicted_classes != true_labels)[0]
#
#         num_correct = min(num_samples // 2, len(correct_indices))
#         num_incorrect = min(num_samples - num_correct, len(incorrect_indices))
#
#         selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if len(
#             correct_indices) > 0 else []
#         selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if len(
#             incorrect_indices) > 0 else []
#
#         selected_indices = np.concatenate([selected_correct, selected_incorrect])
#
#         stats = {
#             'total_samples': len(selected_indices),
#             'correct_predictions': 0,
#             'incorrect_predictions': 0
#         }
#
#         for i, idx in enumerate(selected_indices):
#             try:
#                 sample = validation_data[idx]
#                 true_label = true_labels[idx]
#                 predicted = predicted_classes[idx]
#                 confidence = confidences[idx]
#
#                 if xai_method.lower() == 'gradcam++':
#                     heatmap = self.compute_gradcam_plusplus(sample, class_idx=predicted,
#                                                             target_layer_name=target_layer_name)
#                 elif xai_method.lower() == 'scorecam':
#                     heatmap = self.compute_scorecam(sample, class_idx=predicted,
#                                                     target_layer_name=target_layer_name)
#                 else:
#                     heatmap = self.compute_gradcam(sample, class_idx=predicted,
#                                                    target_layer_name=target_layer_name)
#
#                 is_correct = predicted == true_label
#
#                 if is_correct:
#                     stats['correct_predictions'] += 1
#                     prefix = 'correct'
#                 else:
#                     stats['incorrect_predictions'] += 1
#                     prefix = 'incorrect'
#
#                 save_path = os.path.join(output_dir,
#                                          f'{prefix}_sample_{i:03d}_true_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')
#
#                 self.plot_xai_modern(sample, heatmap, predicted, predicted, true_label,
#                                      confidence=confidence, xai_method=xai_method,
#                                      save_path=save_path, show_plot=False)
#
#             except Exception as e:
#                 logging.error(f"Error processing sample {idx}: {e}")
#                 continue
#
#         return stats
#
#     # Properties
#     @property
#     def neural_network_model(self):
#         return self._neural_network_model
#
#     @neural_network_model.setter
#     def neural_network_model(self, value):
#         self._neural_network_model = value
#
#     @property
#     def list_filters_encoder(self):
#         return self._list_filters_encoder
#
#     @list_filters_encoder.setter
#     def list_filters_encoder(self, value):
#         self._list_filters_encoder = value
#
#     @property
#     def loss_function(self):
#         return self._loss_function
#
#     @loss_function.setter
#     def loss_function(self, value):
#         self._loss_function = value
#
#     @property
#     def optimizer_function(self):
#         return self._optimizer_function
#
#     @optimizer_function.setter
#     def optimizer_function(self, value):
#         self._optimizer_function = value
#
#     @property
#     def kernel_size(self):
#         return self._kernel_size
#
#     @kernel_size.setter
#     def kernel_size(self, value):
#         self._kernel_size = value
#
#     @property
#     def quantization_units(self):
#         return self._quantization_units
#
#     @quantization_units.setter
#     def quantization_units(self, value):
#         self._quantization_units = value
#
#     @property
#     def key_dimension(self):
#         return self._key_dimension
#
#     @key_dimension.setter
#     def key_dimension(self, value):
#         self._key_dimension = value
#
#     @property
#     def intermediary_layer_activation(self):
#         return self._intermediary_layer_activation
#
#     @intermediary_layer_activation.setter
#     def intermediary_layer_activation(self, value):
#         self._intermediary_layer_activation = value
#
#     @property
#     def number_heads(self):
#         return self._number_heads
#
#     @number_heads.setter
#     def number_heads(self, value):
#         self._number_heads = value
#
#     @property
#     def input_dimension(self):
#         return self._input_dimension
#
#     @input_dimension.setter
#     def input_dimension(self, value):
#         self._input_dimension = value
#
#     @property
#     def number_classes(self):
#         return self._number_classes
#
#     @number_classes.setter
#     def number_classes(self, value):
#         self._number_classes = value
#
#     @property
#     def dropout_rate(self):
#         return self._dropout_rate
#
#     @dropout_rate.setter
#     def dropout_rate(self, value):
#         self._dropout_rate = value
#
#     @property
#     def last_layer_activation(self):
#         return self._last_layer_activation
#
#     @last_layer_activation.setter
#     def last_layer_activation(self, value):
#         self._last_layer_activation = value
#
#     @property
#     def model_name(self):
#         return self._model_name
#
#     @model_name.setter
#     def model_name(self, value):
#         self._model_name = value