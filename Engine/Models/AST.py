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
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    from pathlib import Path
    import tensorflow

    from tensorflow.keras import models
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention

    from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
    from Engine.Layers.DistillationCLSTokenLayer import DistillationCLSTokenLayer
    from Engine.Models.Process.AST_Process import ProcessAST

except ImportError as error:
    print(error)
    import sys

    sys.exit(-1)



class AudioSpectrogramTransformer(ProcessAST):

    def __init__(self, arguments):
        """
        Initialize the Audio Spectrogram Transformer.

        Args:
            arguments (object): Configuration object containing model parameters with attributes:
                - ast_head_size: Size of attention heads
                - ast_number_heads: Number of attention heads
                - ast_number_blocks: Number of transformer blocks
                - number_classes: Number of output classes
                - ast_patch_size: Tuple of (height, width) for patches
                - ast_dropout: Dropout rate
                - ast_optimizer_function: Optimizer name
                - ast_loss_function: Loss function name
                - ast_normalization_epsilon: Epsilon for layer normalization
                - ast_intermediary_activation: Activation for intermediate layers
                - ast_projection_dimension: Dimension for token projection
                - ast_number_filters_spectrogram: Number of spectrogram filters
        """
        super().__init__(arguments)
        self.neural_network_model = None
        self.attention_model = None
        self.attention_layers = []
        self.head_size = arguments.ast_head_size
        self.number_heads = arguments.ast_head_size
        self.number_blocks = arguments.ast_number_blocks
        self.number_classes = arguments.number_classes
        self.patch_size = arguments.ast_patch_size
        self.dropout = arguments.ast_dropout
        self.optimizer_function = arguments.ast_optimizer_function
        self.loss_function = arguments.ast_loss_function
        self.normalization_epsilon = arguments.ast_normalization_epsilon
        self.last_activation_layer = arguments.ast_intermediary_activation
        self.projection_dimension = arguments.ast_projection_dimension
        self.intermediary_activation = arguments.ast_intermediary_activation
        self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram
        self.model_name = "AST_Distillation"
        self.use_distillation = True

    def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Build the transformer encoder with multiple blocks.

        Each transformer block consists of:
        1. Layer normalization
        2. Multi-head self-attention with residual connection
        3. Layer normalization
        4. Feed-forward network with residual connection

        Args:
            inputs (tensorflow.Tensor): Input tensor of shape (batch_size, sequence_length, projection_dim)

        Returns:
            tensorflow.Tensor: Output tensor of same shape as inputs after transformer processing
        """
        neural_network_flow = inputs

        for block_idx in range(self.number_blocks):
            neural_network_flow_normalized = LayerNormalization(epsilon=self.normalization_epsilon,
                                                                name=f'norm1_block_{block_idx}')(neural_network_flow)

            attention_layer = MultiHeadAttention(key_dim=self.projection_dimension,
                                                 num_heads=self.number_heads,
                                                 dropout=self.dropout,
                                                 name=f'attention_block_{block_idx}')

            attention_output, attention_scores = attention_layer(neural_network_flow_normalized,
                                                                 neural_network_flow_normalized,
                                                                 return_attention_scores=True)

            self.attention_layers.append((attention_layer, attention_scores))
            attention_output = Dropout(self.dropout, name=f'dropout1_block_{block_idx}')(attention_output)
            neural_network_flow = Add(name=f'add1_block_{block_idx}')([attention_output, neural_network_flow])

            neural_network_flow_normalized = LayerNormalization(epsilon=self.normalization_epsilon,
                                        name=f'norm2_block_{block_idx}')(neural_network_flow)

            ffn_dim = self.projection_dimension * 4
            classification_head_output = Dense(ffn_dim,
                                               activation=self.intermediary_activation,
                                               name=f'ffn1_block_{block_idx}')(neural_network_flow_normalized)

            classification_head_output = Dropout(self.dropout,
                                                 name=f'dropout2_block_{block_idx}')(classification_head_output)
            classification_head_output = Dense(self.projection_dimension,
                                               name=f'ffn2_block_{block_idx}')(classification_head_output)
            classification_head_output = Dropout(self.dropout,
                                                 name=f'dropout3_block_{block_idx}')(classification_head_output)
            neural_network_flow = Add(name=f'add2_block_{block_idx}')([classification_head_output, neural_network_flow])

        neural_network_flow = LayerNormalization(epsilon=self.normalization_epsilon, name='final_norm')(neural_network_flow)

        return neural_network_flow

    def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
        """
        Build the complete Audio Spectrogram Transformer model.

        Model Architecture:
        1. Input patches processing and flattening
        2. Linear projection to embedding dimension
        3. Addition of CLS and/or distillation tokens
        4. Positional embeddings
        5. Transformer encoder blocks
        6. Classification heads with optional distillation

        Args:
            number_patches (int): Number of patches extracted from the spectrogram

        Returns:
            tensorflow.keras.models.Model: Compiled transformer model

        Raises:
            ValueError: If required components are not properly initialized
        """
        self.attention_layers = []

        inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
        input_flatten = TimeDistributed(Flatten())(inputs)
        linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)

        if self.use_distillation:
            neural_model_flow = DistillationCLSTokenLayer()(linear_projection)
            num_special_tokens = 2
        else:
            from Engine.Layers.CLSTokenLayer import CLSTokenLayer
            neural_model_flow = CLSTokenLayer()(linear_projection)
            num_special_tokens = 1

        positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches + num_special_tokens - 1,
                                                                self.projection_dimension)(neural_model_flow)

        neural_model_flow = neural_model_flow + positional_embeddings_layer

        neural_model_flow = self.transformer_encoder(neural_model_flow)

        if self.use_distillation:
            cls_output = neural_model_flow[:, 0, :]
            dist_output = neural_model_flow[:, 1, :]

            cls_output = Dropout(self.dropout, name='cls_dropout')(cls_output)
            dist_output = Dropout(self.dropout, name='dist_dropout')(dist_output)

            cls_logits = Dense(self.projection_dimension,
                               activation=self.intermediary_activation,
                               name='cls_mlp_hidden')(cls_output)

            cls_logits = Dropout(self.dropout)(cls_logits)
            cls_logits = Dense(self.number_classes,
                               activation='softmax',
                               name='cls_head')(cls_logits)

            distillation_logits = Dense(self.projection_dimension,
                                        activation=self.intermediary_activation,
                                        name='dist_mlp_hidden')(dist_output)

            distillation_logits = Dropout(self.dropout)(distillation_logits)

            distillation_logits = Dense(self.number_classes,
                                        activation='softmax',
                                        name='dist_head')(distillation_logits)

            outputs = tensorflow.keras.layers.Average(name='average_predictions')([cls_logits, distillation_logits])

        else:
            cls_output = neural_model_flow[:, 0, :]
            cls_output = Dropout(self.dropout)(cls_output)

            feedforward_hidden = Dense(self.projection_dimension,
                                       activation=self.intermediary_activation,
                                       name='mlp_head_hidden')(cls_output)

            feedforward_hidden = Dropout(self.dropout)(feedforward_hidden)

            outputs = Dense(self.number_classes,
                            activation='softmax',
                            name='classification_head')(feedforward_hidden)

        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
        self.neural_network_model.summary()

        return self.neural_network_model

    def build_attention_model(self):
        """
        Build a model specifically for extracting attention weights.

        Returns:
            tensorflow.keras.models.Model: Model that outputs attention scores from all layers

        Raises:
            ValueError: If main model hasn't been built yet
        """
        if self.neural_network_model is None:
            raise ValueError("The main model must be built before creating attention model.")

        attention_score_tensors = [scores for _, scores in self.attention_layers]

        if not attention_score_tensors:
            print("Warning: No attention scores found in the model.")
            return None

        self.attention_model = models.Model(inputs=self.neural_network_model.input,
                                            outputs=attention_score_tensors,
                                            name='attention_extractor')

        return self.attention_model

    def extract_attention_weights(self, data_sample):
        """
        Extract attention weights from all transformer blocks for a given input sample.

        Args:
            data_sample (np.ndarray): Input data sample of shape (patches, height, width)
                                    or (batch_size, patches, height, width)

        Returns:
            list: List of attention weight matrices from all transformer blocks

        Example:
            ```python
            # For a single sample
            attention_weights = model.extract_attention_weights(sample_data)

            # For batch processing
            batch_attention = [model.extract_attention_weights(sample) for sample in batch]
            ```
        """
        if self.attention_model is None:
            self.build_attention_model()

        if len(data_sample.shape) == 3:
            data_sample = numpy.expand_dims(data_sample, axis=0)

        attention_outputs = self.attention_model.predict(data_sample, verbose=0)

        if not isinstance(attention_outputs, list):
            attention_outputs = [attention_outputs]

        return attention_outputs

    def compute_attention_rollout(self, attention_weights):
        """
        Compute attention rollout to understand global attention patterns.

        Attention rollout combines attention matrices across layers to show
        how information flows through the entire transformer architecture.

        Args:
            attention_weights (list): List of attention weight matrices from all blocks

        Returns:
            np.ndarray: Rollout matrix showing cumulative attention flow

        Reference:
            Abnar et al. "Quantifying Attention Flow in Transformers" (2020)
        """
        processed_attentions = []

        for attention in attention_weights:
            if len(attention.shape) == 4:

                attention_avg = numpy.mean(attention[0], axis=0)
            elif len(attention.shape) == 3:
                if attention.shape[0] == self.number_heads:
                    attention_avg = numpy.mean(attention, axis=0)
                else:
                    attention_avg = attention[0]
            elif len(attention.shape) == 2:
                attention_avg = attention
            else:
                continue
            processed_attentions.append(attention_avg)

        if not processed_attentions:
            return None

        rollout = numpy.eye(processed_attentions[0].shape[0])

        for attention_matrix in processed_attentions:

            attention_normalized = attention_matrix / (attention_matrix.sum(axis=-1, keepdims=True) + 1e-10)
            attention_with_residual = 0.5 * attention_normalized + 0.5 * numpy.eye(attention_normalized.shape[0])
            rollout = numpy.matmul(attention_with_residual, rollout)

        return rollout

    def compute_attention_flow(self, attention_weights, target_token=0):
        """
        Compute attention flow from a specific token to all other tokens.

        This shows how much each patch is influenced by the target token (usually CLS or DIST token)
        throughout the entire transformer architecture.

        Args:
            attention_weights (list): List of attention weight matrices
            target_token (int): Index of the token to compute flow from (0 for CLS, 1 for DIST)

        Returns:
            np.ndarray: Attention flow vector showing influence of target token on all tokens
        """
        rollout = self.compute_attention_rollout(attention_weights)

        if rollout is None:
            return None

        flow = rollout[target_token, :]

        return flow

    def visualize_attention_flow(self, data_samples, labels=None, num_samples=4,
                                 output_dir='attention_visualizations'):
        """
        Generate comprehensive attention visualization for multiple data samples.

        Creates unified visualization plots including:
        - Input spectrogram with patch boundaries
        - Attention rollout matrix
        - CLS and DIST token attention flows overlaid on spectrogram
        - Individual block attention maps
        - Statistical analysis of attention patterns

        Args:
            data_samples (np.ndarray): Batch of input samples
            labels (np.ndarray, optional): Ground truth labels for evaluation
            num_samples (int): Number of samples to visualize
            output_dir (str): Directory to save visualization images

        Example:
            ```python
            # Visualize attention for validation set
            model.visualize_attention_flow(
                data_samples=X_val,
                labels=y_val,
                num_samples=8,
                output_dir='attention_analysis'
            )
            ```
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_samples = min(num_samples, len(data_samples))
        sample_indices = numpy.random.choice(len(data_samples), num_samples, replace=False)

        for idx, sample_idx in enumerate(sample_indices):

            sample = data_samples[sample_idx:sample_idx + 1]
            attention_weights = self.extract_attention_weights(sample)
            prediction = self.neural_network_model.predict(sample, verbose=0)
            predicted_class = numpy.argmax(prediction[0])
            confidence = numpy.max(prediction[0])

            true_class = numpy.argmax(labels[sample_idx]) if labels is not None else None

            self._plot_unified_attention_analysis(attention_weights,
                                                  sample_idx,
                                                  data_sample=sample,
                                                  predicted_class=predicted_class,
                                                  true_class=true_class,
                                                  confidence=confidence,
                                                  output_path=output_path)

    @staticmethod
    def _reconstruct_spectrogram(sample_data):
        """
        Reconstruct spectrogram from patch representation.

        Args:
            sample_data (np.ndarray): Patch data of shape (num_patches, height, width)

        Returns:
            tuple: (reconstructed_spectrogram, (patches_x, patches_y)) or (None, None) if reconstruction fails
        """
        if len(sample_data.shape) != 3:
            return None, None

        num_patches = sample_data.shape[0]
        patch_height = sample_data.shape[1]
        patch_width = sample_data.shape[2]

        number_patches_x = 5
        number_patches_y = 64

        expected_patches = number_patches_x * number_patches_y

        if num_patches != expected_patches:

            number_patches_y = int(numpy.sqrt(num_patches))
            while num_patches % number_patches_y != 0:
                number_patches_y += 1
            number_patches_x = num_patches // number_patches_y

        reconstructed_rows = []
        patch_idx = 0
        for freq_idx in range(number_patches_y):
            row_patches = []
            for time_idx in range(number_patches_x):
                if patch_idx < num_patches:
                    row_patches.append(sample_data[patch_idx])
                    patch_idx += 1
            if row_patches:
                reconstructed_rows.append(numpy.hstack(row_patches))

        if reconstructed_rows:
            reconstructed_spectrogram = numpy.vstack(reconstructed_rows)
            return reconstructed_spectrogram, (number_patches_x, number_patches_y)

        return None, None

    def _plot_unified_attention_analysis(self, attention_weights, sample_idx, data_sample=None,
                                         predicted_class=None, true_class=None, confidence=None,
                                         output_path=None):
        """
        Create comprehensive attention analysis visualization.

        Internal method that generates a multi-panel figure showing various aspects
        of attention mechanisms in the transformer.
        """
        num_blocks = len(attention_weights)

        # Criar figura com layout otimizado
        fig = plt.figure(figsize=(28, 18))
        gs = fig.add_gridspec(5, num_blocks + 2, height_ratios=[2, 2, 1.5, 1.5, 1.5],
                              hspace=0.4, wspace=0.3)

        # Título
        title_parts = [f'Comprehensive Attention Analysis – Sample {sample_idx}']
        if predicted_class is not None:
            status = "✅" if (true_class is not None and predicted_class == true_class) else "❌"
            title_parts.append(f'\n{status} Prediction: Class {predicted_class}')
            if confidence is not None:
                title_parts.append(f'(Confidence: {confidence:.1%})')
            if true_class is not None:
                title_parts.append(f'| Real: Class {true_class}')

        fig.suptitle(' '.join(title_parts), fontsize=20, fontweight='bold', y=0.98)
        plt.style.use('seaborn-v0_8-whitegrid')

        processed_attentions = []
        for attention in attention_weights:
            if len(attention.shape) == 4:
                attention_avg = numpy.mean(attention[0], axis=0)
            elif len(attention.shape) == 3:
                if attention.shape[0] == self.number_heads:
                    attention_avg = numpy.mean(attention, axis=0)
                else:
                    attention_avg = attention[0]
            elif len(attention.shape) == 2:
                attention_avg = attention
            else:
                continue
            processed_attentions.append(attention_avg)

        if not processed_attentions:
            plt.close()
            return

        ax_spec = fig.add_subplot(gs[0, 0])
        reconstructed_spectrogram = None

        if data_sample is not None:
            sample_data = data_sample[0]
            reconstructed_spectrogram, patch_config = self._reconstruct_spectrogram(sample_data)

            if reconstructed_spectrogram is not None:
                im = ax_spec.imshow(reconstructed_spectrogram, cmap='magma', aspect='auto',
                                    origin='lower', interpolation='bilinear')
                ax_spec.set_title(
                    f'Input Spectrogram\n{reconstructed_spectrogram.shape[0]}×{reconstructed_spectrogram.shape[1]}',
                    fontweight='bold', fontsize=11)
                ax_spec.set_xlabel('Time', fontweight='bold', fontsize=9)
                ax_spec.set_ylabel('Frequency', fontweight='bold', fontsize=9)

                # Linhas de separação de patches
                if patch_config:
                    number_patches_x, _ = patch_config
                    patch_height = sample_data.shape[1]
                    for i in range(1, number_patches_x):
                        ax_spec.axvline(x=i * patch_height - 0.5, color='cyan',
                                        linestyle='-', linewidth=1.5, alpha=0.7)

                cbar = plt.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
                cbar.set_label('Magnitude', rotation=270, labelpad=15, fontsize=9)

        ax_rollout = fig.add_subplot(gs[0, 1])
        rollout = self.compute_attention_rollout(attention_weights)

        if rollout is not None:

            rollout_safe = rollout + 1e-10

            im_rollout = ax_rollout.imshow(
                rollout_safe,
                cmap='viridis',
                aspect='auto',
                interpolation='bilinear',
                norm=LogNorm(vmin=rollout_safe.min(), vmax=rollout_safe.max())
            )
            ax_rollout.set_title('Attention Rollout (Log Scale)\n(Attention Accumulation)',
                                 fontweight='bold', fontsize=11)
            ax_rollout.set_xlabel('Token Position', fontsize=9)
            ax_rollout.set_ylabel('Token Position', fontsize=9)

            cbar = plt.colorbar(im_rollout, ax=ax_rollout, fraction=0.046, pad=0.04)
            cbar.set_label('Rollout Weight (Log)', rotation=270, labelpad=15, fontsize=9)

        ax_flow_cls = fig.add_subplot(gs[1, 0])
        flow_cls = self.compute_attention_flow(attention_weights, target_token=0)

        if flow_cls is not None and reconstructed_spectrogram is not None:

            num_special_tokens = 2 if self.use_distillation else 1
            flow_patches = flow_cls[num_special_tokens:]

            if patch_config:
                number_patches_x, number_patches_y = patch_config
                patch_height = sample_data.shape[1]
                patch_width = sample_data.shape[2]

                flow_reconstructed_rows = []
                patch_idx = 0
                for freq_idx in range(number_patches_y):
                    row_patches = []
                    for time_idx in range(number_patches_x):
                        if patch_idx < len(flow_patches):
                            flow_patch = numpy.full((patch_height, patch_width), flow_patches[patch_idx])
                            row_patches.append(flow_patch)
                            patch_idx += 1

                    if row_patches:
                        flow_reconstructed_rows.append(numpy.hstack(row_patches))

                if flow_reconstructed_rows:
                    flow_reconstructed = numpy.vstack(flow_reconstructed_rows)

                    im_flow = ax_flow_cls.imshow(flow_reconstructed, cmap='hot', aspect='auto',
                                                 origin='lower', interpolation='bilinear', alpha=0.8)
                    ax_flow_cls.set_title(' CLS Token Attention Flow\n(Patch Importance)',
                                          fontweight='bold', fontsize=11)
                    ax_flow_cls.set_xlabel('Time', fontweight='bold', fontsize=9)
                    ax_flow_cls.set_ylabel('Frequency', fontweight='bold', fontsize=9)

                    for i in range(1, number_patches_x):
                        ax_flow_cls.axvline(x=i * patch_height - 0.5, color='white',
                                            linestyle='-', linewidth=1.5, alpha=0.7)

                    cbar = plt.colorbar(im_flow, ax=ax_flow_cls, fraction=0.046, pad=0.04)
                    cbar.set_label('Flow Weight', rotation=270, labelpad=15, fontsize=9)

        if self.use_distillation:
            ax_flow_dist = fig.add_subplot(gs[1, 1])
            flow_dist = self.compute_attention_flow(attention_weights, target_token=1)

            if flow_dist is not None and reconstructed_spectrogram is not None:
                flow_patches = flow_dist[2:]

                if patch_config:
                    number_patches_x, number_patches_y = patch_config
                    patch_height = sample_data.shape[1]
                    patch_width = sample_data.shape[2]

                    flow_reconstructed_rows = []
                    patch_idx = 0
                    for freq_idx in range(number_patches_y):
                        row_patches = []
                        for time_idx in range(number_patches_x):
                            if patch_idx < len(flow_patches):
                                # Criar patch com o valor de flow
                                flow_patch = numpy.full((patch_height, patch_width), flow_patches[patch_idx])
                                row_patches.append(flow_patch)
                                patch_idx += 1
                        if row_patches:
                            flow_reconstructed_rows.append(numpy.hstack(row_patches))

                    if flow_reconstructed_rows:
                        flow_reconstructed = numpy.vstack(flow_reconstructed_rows)

                        im_flow = ax_flow_dist.imshow(flow_reconstructed, cmap='plasma', aspect='auto',
                                                      origin='lower', interpolation='bilinear', alpha=0.8)
                        ax_flow_dist.set_title('DIST Token Attention Flow\n(Patch Importance)',
                                               fontweight='bold', fontsize=11)
                        ax_flow_dist.set_xlabel('Time', fontweight='bold', fontsize=9)
                        ax_flow_dist.set_ylabel('Frequency', fontweight='bold', fontsize=9)


                        for i in range(1, number_patches_x):
                            ax_flow_dist.axvline(x=i * patch_height - 0.5, color='white',
                                                 linestyle='-', linewidth=1.5, alpha=0.7)

                        cbar = plt.colorbar(im_flow, ax=ax_flow_dist, fraction=0.046, pad=0.04)
                        cbar.set_label('Flow Weight', rotation=270, labelpad=15, fontsize=9)

        for block_idx, attention_avg in enumerate(processed_attentions):
            row = 0 if block_idx < len(processed_attentions) // 2 else 1
            col = (block_idx % (len(processed_attentions) // 2 + 1)) + 2

            if col < num_blocks + 2:
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(attention_avg, cmap='magma', aspect='auto', interpolation='bilinear',
                               vmin=0, vmax=attention_avg.max())

                ax.set_title(f'Block {block_idx + 1}\n(μ={attention_avg.mean():.3f})',
                             fontweight='bold', fontsize=10, pad=8)
                ax.set_xlabel('Key', fontsize=8)

                if col == 2:
                    ax.set_ylabel('Query', fontsize=8)

                ax.tick_params(labelsize=7)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)

        ax_avg = fig.add_subplot(gs[2, :num_blocks])
        avg_attentions = [numpy.mean(att) for att in processed_attentions]
        blocks = [f'B{i + 1}' for i in range(len(avg_attentions))]
        colors = plt.cm.viridis(numpy.linspace(0.3, 0.9, len(blocks)))
        bars = ax_avg.bar(blocks, avg_attentions, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.2)

        for bar, val in zip(bars, avg_attentions):
            height = bar.get_height()
            ax_avg.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax_avg.set_xlabel('Transformer Block', fontweight='bold', fontsize=10)
        ax_avg.set_ylabel('Average Attention', fontweight='bold', fontsize=10)
        ax_avg.set_title('Average Attention Distribution per Block', fontweight='bold', fontsize=12, pad=10)
        ax_avg.grid(True, alpha=0.3, axis='y')

        ax_hist = fig.add_subplot(gs[2, num_blocks:])
        all_weights = numpy.concatenate([att.flatten() for att in processed_attentions])
        ax_hist.hist(all_weights, bins=50, color='#16A085', alpha=0.7, edgecolor='black', linewidth=0.8)
        ax_hist.axvline(all_weights.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {all_weights.mean():.3f}')
        ax_hist.set_title('Weight Distribution', fontweight='bold', fontsize=12)
        ax_hist.legend(fontsize=9)

        ax_entropy = fig.add_subplot(gs[3, :num_blocks])
        entropies = []

        for attention_avg in processed_attentions:
            attention_safe = attention_avg + 1e-10
            attention_norm = attention_safe / numpy.sum(attention_safe, axis=1, keepdims=True)
            entropy_per_query = -numpy.sum(attention_norm * numpy.log(attention_norm + 1e-10), axis=1)
            entropies.append(numpy.mean(entropy_per_query))

        ax_entropy.plot(blocks, entropies, marker='o', linewidth=3, markersize=10,
                        color='#E74C3C', markeredgecolor='black', markeredgewidth=1.5)
        ax_entropy.fill_between(range(len(blocks)), entropies, alpha=0.3, color='#E74C3C')
        ax_entropy.set_title('Attention Diversity (Entropy) per Block', fontweight='bold', fontsize=12)
        ax_entropy.grid(True, alpha=0.3)

        ax_range = fig.add_subplot(gs[3, num_blocks:])
        attention_ranges = [att.max() - att.min() for att in processed_attentions]
        bars_range = ax_range.bar(blocks, attention_ranges, color=colors, alpha=0.8,
                                  edgecolor='black', linewidth=1.2)
        ax_range.set_title('Attention Range', fontweight='bold', fontsize=12)
        ax_range.grid(True, alpha=0.3, axis='y')

        ax_stats = fig.add_subplot(gs[4, :])
        ax_stats.axis('off')

        stats_text = "Global Statistics\n" + "=" * 80 + "\n\n"
        all_attentions = numpy.concatenate([att.flatten() for att in processed_attentions])
        stats_text += f"Overall Average: {all_attentions.mean():.4f} | "
        stats_text += f"Standard Deviation: {all_attentions.std():.4f} | "
        stats_text += f"Median: {numpy.median(all_attentions):.4f}\n"

        if flow_cls is not None:
            stats_text += f"\n CLS Flow - Top 5 patches: {numpy.argsort(flow_cls[2:])[-5:][::-1]}\n"
        if self.use_distillation and flow_dist is not None:
            stats_text += f" DIST Flow - Top 5 patches: {numpy.argsort(flow_dist[2:])[-5:][::-1]}\n"

        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                      fontsize=10, verticalalignment='top', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=1))

        plt.style.use('default')

        if output_path:
            filename = f'unified_attention_flow_sample_{sample_idx}'
            if predicted_class is not None:
                filename += f'_pred{predicted_class}'
            plt.savefig(output_path / f'{filename}.png', dpi=200, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        plt.close()

    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True):
        """
        Compile and train the Audio Spectrogram Transformer model.

        Args:
            train_data (np.ndarray): Training data samples
            train_labels (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_data (tuple, optional): (val_data, val_labels) for validation
            visualize_attention (bool): Whether to generate attention visualizations after training

        Returns:
            tensorflow.keras.callbacks.History: Training history object

        Example:
            ```python
            history = model.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                visualize_attention=True
            )
            ```
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data,
                                                         train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)

        # if visualize_attention and validation_data is not None:
        #     val_data, val_labels = validation_data
        #
        #     self.visualize_attention_flow(val_data,
        #                                   labels=val_labels,
        #                                   num_samples=128,
        #                                   output_dir='Maps_AST')

        return training_history

    # Properties
    @property
    def head_size(self):
        return self._head_size

    @head_size.setter
    def head_size(self, head_size: int):
        self._head_size = head_size

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, num_heads: int):
        self._number_heads = num_heads

    @property
    def number_blocks(self):
        return self._number_blocks

    @number_blocks.setter
    def number_blocks(self, number_blocks: int):
        self._number_blocks = number_blocks

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, number_classes: int):
        self._number_classes = number_classes

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: tuple):
        self._patch_size = patch_size

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, dropout: float):
        self._dropout = dropout

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, optimizer_function: str):
        self._optimizer_function = optimizer_function

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: str):
        self._loss_function = loss_function

    @property
    def normalization_epsilon(self):
        return self._normalization_epsilon

    @normalization_epsilon.setter
    def normalization_epsilon(self, normalization_epsilon: float):
        self._normalization_epsilon = normalization_epsilon

    @property
    def last_activation_layer(self):
        return self._last_activation_layer

    @last_activation_layer.setter
    def last_activation_layer(self, last_activation_layer: str):
        self._last_activation_layer = last_activation_layer

    @property
    def projection_dimension(self):
        return self._projection_dimension

    @projection_dimension.setter
    def projection_dimension(self, projection_dimension: int):
        self._projection_dimension = projection_dimension

    @property
    def intermediary_activation(self):
        return self._intermediary_activation

    @intermediary_activation.setter
    def intermediary_activation(self, intermediary_activation: str):
        self._intermediary_activation = intermediary_activation

    @property
    def number_filters_spectrogram(self):
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, number_filters_spectrogram: int):
        self._number_filters_spectrogram = number_filters_spectrogram