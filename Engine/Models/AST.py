# # # !/usr/bin/python3
# # # -*- coding: utf-8 -*-
# #
# # __author__ = 'unknown'
# # __email__ = 'unknown@unknown.com.br'
# # __version__ = '{1}.{0}.{2}'
# # __initial_data__ = '2025/04/1'
# # __last_update__ = '2025/10/18'
# # __credits__ = ['unknown']
# #
# # # MIT License
# # #
# # # Copyright (c) 2025 unknown
# # #
# # # Permission is hereby granted, free of charge, to any person obtaining a copy
# # # of this software and associated documentation files (the "Software"), to deal
# # # in the Software without restriction, including without limitation the rights
# # # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # # copies of the Software, and to permit persons to whom the Software is
# # # furnished to do so, subject to the following conditions:
# # #
# # # The above copyright notice and this permission notice shall be included in all
# # # copies or substantial portions of the Software.
# # #
# # # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # # SOFTWARE.
# #
# #
# # try:
# #
# #     import sys
# #     import numpy as np
# #     import matplotlib.pyplot as plt
# #     import seaborn as sns
# #     from pathlib import Path
# #
# #     import tensorflow
# #
# #     from tensorflow.keras import models
# #
# #     from tensorflow.keras.layers import Add
# #     from tensorflow.keras.layers import Input
# #     from tensorflow.keras.layers import Layer
# #     from tensorflow.keras.layers import Dense
# #
# #     from tensorflow.keras.layers import Conv1D
# #     from tensorflow.keras.layers import Dropout
# #     from tensorflow.keras.layers import Flatten
# #
# #     from tensorflow.keras.layers import Embedding
# #     from tensorflow.keras.layers import Concatenate
# #
# #     from tensorflow.keras.layers import TimeDistributed
# #
# #     from Engine.Layers.CLSTokenLayer import CLSTokenLayer
# #
# #     from tensorflow.keras.layers import LayerNormalization
# #     from tensorflow.keras.layers import MultiHeadAttention
# #
# #     from Engine.Models.Process.AST_Process import ProcessAST
# #     from tensorflow.keras.layers import GlobalAveragePooling1D
# #
# #     from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
# #
# # except ImportError as error:
# #     print(error)
# #     sys.exit(-1)
# #
# #
# # class AudioSpectrogramTransformer(ProcessAST):
# #     """
# #     @AudioSpectrogramTransformer
# #
# #         The Audio Spectrogram Transformer (AST) is a deep learning model designed for
# #         audio classification tasks. It leverages a transformer architecture, similar
# #         to the one used in natural language processing, and applies it to spectrograms
# #         of audio data. The model is composed of a transformer encoder, followed by
# #         a feedforward network and a classification layer.
# #
# #     Reference:
# #         Gong, Y., Xu, M., Li, J., Liu, Z., & Xu, B. (2021). AST: Audio Spectrogram Transformer.
# #         *arXiv preprint arXiv:2104.01778*. https://arxiv.org/abs/2104.01778
# #
# #     Attributes:
# #         @neural_network_model (tensorflow.keras.Model): The Keras model representing the AST network.
# #         @attention_model (tensorflow.keras.Model): Model for extracting attention weights.
# #         @attention_layers (list): List of MultiHeadAttention layers for visualization.
# #         @head_size (int): The size of the attention heads in multi-head attention layers.
# #         @number_heads (int): The number of attention heads in the multi-head attention mechanism.
# #         @number_blocks (int): The number of transformer blocks in the model.
# #         @number_classes (int): The number of output classes for classification.
# #         @patch_size (tuple): The size of each spectrogram patch.
# #         @dropout (float): The dropout rate for regularization.
# #         @optimizer_function (str): The optimizer function used during training (e.g., 'adam').
# #         @loss_function (str): The loss function used during training (e.g., 'categorical_crossentropy').
# #         @normalization_epsilon (float): The epsilon value used for layer normalization.
# #         @last_activation_layer (str): The activation function for the last layer (e.g., 'softmax').
# #         @projection_dimension (int): The dimension for the linear projection layer.
# #         @intermediary_activation (str): The activation function used in intermediary layers (e.g., 'relu').
# #         @number_filters_spectrogram (int): The number of filters used in the spectrogram extraction.
# #         @model_name (str): The name of the model (default is "AST").
# #
# #     Example:
# #         >>> # Instantiate the AudioSpectrogramTransformer model
# #         ...     model = AudioSpectrogramTransformer(
# #         ...     projection_dimension=512,  # Projection dimension for transformer
# #         ...     head_size=64,  # Head size for multi-head attention
# #         ...     num_heads=8,  # Number of attention heads
# #         ...     number_blocks=12,  # Number of transformer blocks
# #         ...     number_classes=10,  # Number of output classes
# #         ...     patch_size=(32, 32),  # Spectrogram patch size
# #         ...     dropout=0.1,  # Dropout rate for regularization
# #         ...     intermediary_activation='relu',  # Activation function for intermediate layers
# #         ...     loss_function='categorical_crossentropy',  # Loss function for training
# #         ...     last_activation_layer='softmax',  # Activation function for the last layer
# #         ...     optimizer_function='adam',  # Optimizer function
# #         ...     normalization_epsilon=1e-6,  # Epsilon for layer normalization
# #         ...     number_filters_spectrogram=64  # Number of filters in the spectrogram
# #         ...     )
# #         ...     # Build the model
# #         ...     model.build_model(number_patches=10)
# #         ...
# #         ...     # Compile and train the model
# #         ...     training_history = model.compile_and_train(
# #         ...     train_data=X_train,  # Training data
# #         ...     train_labels=y_train,  # Training labels
# #         ...     epochs=10,  # Number of epochs for training
# #         ...     batch_size=32,  # Batch size
# #         ...     validation_data=(X_val, y_val)  # Optional validation data
# #         ...     )
# #         >>>
# #
# #     """
# #
# #     def __init__(self, arguments):
# #
# #         """
# #         Initialize the AudioSpectrogramTransformer model with the specified hyperparameters.
# #
# #         Args:
# #             @projection_dimension (int): The projection dimension for each input patch.
# #             @head_size (int): The size of each attention head in the multi-head attention mechanism.
# #             @num_heads (int): The number of attention heads in the multi-head attention layer.
# #             @number_blocks (int): The number of transformer blocks (layers) in the encoder.
# #             @number_classes (int): The number of output classes for the classification task.
# #             @patch_size (tuple): The size of the input spectrogram patches, defined by the
# #              (time, frequency) dimensions.
# #             @dropout (float): The dropout rate to be applied for regularization during training.
# #             @intermediary_activation (str): The activation function to be used in the intermediate
# #              layers, typically 'relu' or 'gelu'.
# #             @loss_function (str): The loss function to use for training, such as 'categorical_crossentropy'
# #              for multi-class classification.
# #             @last_activation_layer (str): The activation function for the final output layer,
# #              commonly 'softmax' for classification.
# #             @optimizer_function (str): The optimizer function to be used, such as 'adam' or 'sgd'.
# #             @normalization_epsilon (float): A small constant added for numerical stability in
# #              layer normalization.
# #             @number_filters_spectrogram (int): The number of filters to apply for feature extraction
# #              from the spectrogram before the transformer.
# #
# #         """
# #         super().__init__(arguments)
# #         self.neural_network_model = None
# #         self.attention_model = None
# #         self.attention_layers = []
# #         self.head_size = arguments.ast_head_size
# #         self.number_heads = arguments.ast_head_size
# #         self.number_blocks = arguments.ast_number_blocks
# #         self.number_classes = arguments.number_classes
# #         self.patch_size = arguments.ast_patch_size
# #         self.dropout = arguments.ast_dropout
# #         self.optimizer_function = arguments.ast_optimizer_function
# #         self.loss_function = arguments.ast_loss_function
# #         self.normalization_epsilon = arguments.ast_normalization_epsilon
# #         self.last_activation_layer = arguments.ast_intermediary_activation
# #         self.projection_dimension = arguments.ast_projection_dimension
# #         self.intermediary_activation = arguments.ast_intermediary_activation
# #         self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram
# #         self.model_name = "AST"
# #
# #     def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
# #         """
# #         Transformer encoder com correções para melhorar a qualidade da atenção.
# #         """
# #         x = inputs
# #
# #         for block_idx in range(self.number_blocks):
# #             # ===== Multi-Head Attention Block =====
# #             # Layer Normalization ANTES da atenção
# #             x_norm = LayerNormalization(epsilon=self.normalization_epsilon, name=f'norm1_block_{block_idx}')(x)
# #
# #             # Multi-head attention com key_dim apropriado
# #             attention_layer = MultiHeadAttention(
# #                 key_dim=self.projection_dimension,
# #                 num_heads=self.number_heads,
# #                 dropout=self.dropout,
# #                 name=f'attention_block_{block_idx}'
# #             )
# #
# #             attention_output, attention_scores = attention_layer(x_norm, x_norm, return_attention_scores=True)
# #
# #             self.attention_layers.append((attention_layer, attention_scores))
# #
# #             attention_output = Dropout(self.dropout, name=f'dropout1_block_{block_idx}')(attention_output)
# #
# #             x = Add(name=f'add1_block_{block_idx}')([attention_output, x])
# #
# #             x_norm = LayerNormalization(epsilon=self.normalization_epsilon, name=f'norm2_block_{block_idx}')(x)
# #
# #             ffn_dim = self.projection_dimension * 4
# #
# #             ffn_output = Dense(ffn_dim, activation=self.intermediary_activation, name=f'ffn1_block_{block_idx}')(x_norm)
# #
# #             ffn_output = Dropout(self.dropout, name=f'dropout2_block_{block_idx}')(ffn_output)
# #
# #             ffn_output = Dense(self.projection_dimension, name=f'ffn2_block_{block_idx}')(ffn_output)
# #
# #             ffn_output = Dropout(self.dropout, name=f'dropout3_block_{block_idx}')(ffn_output)
# #
# #             x = Add(name=f'add2_block_{block_idx}')([ffn_output, x])
# #
# #         # Layer Normalization final
# #         x = LayerNormalization(epsilon=self.normalization_epsilon, name='final_norm')(x)
# #
# #         return x
# #
# #     def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
# #         """
# #         Builds the AST model consisting of a transformer encoder and a classification head.
# #
# #         Args:
# #             number_patches (int): The number of patches in the input spectrogram.
# #         """
# #
# #         # Reset attention layers list
# #         self.attention_layers = []
# #
# #         # Define the input layer with shape (number_patches, projection_dimension)
# #         inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
# #         input_flatten = TimeDistributed(Flatten())(inputs)
# #         linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)
# #
# #         # Add CLS token at the beginning (CLSTokenLayer já retorna [CLS, patches] concatenados)
# #         neural_model_flow = CLSTokenLayer()(linear_projection)
# #
# #         # Add positional embeddings to ALL positions (CLS + patches)
# #         # PositionalEmbeddingsLayer adiciona +1 internamente para o CLS token
# #         positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches,
# #                                                                 self.projection_dimension)(neural_model_flow)
# #         neural_model_flow = neural_model_flow + positional_embeddings_layer
# #
# #         # Pass the input through the transformer encoder
# #         neural_model_flow = self.transformer_encoder(neural_model_flow)
# #
# #         # Apply dropout for regularization
# #         neural_model_flow = Flatten()(neural_model_flow)
# #         neural_model_flow = Dropout(self.dropout)(neural_model_flow)
# #         # Define the output layer with the specified number of classes and activation function
# #         outputs = Dense(self.number_classes, activation='softmax')(neural_model_flow)
# #
# #         # Create the Keras model
# #         self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
# #         self.neural_network_model.summary()
# #
# #         return self.neural_network_model
# #
# #     def build_attention_model(self):
# #         """
# #         Builds a model to extract attention weights from all transformer blocks.
# #         This model outputs the attention weights for visualization.
# #         """
# #         if self.neural_network_model is None:
# #             raise ValueError("The main model must be built before creating attention model.")
# #
# #         # Get attention score tensors that were saved during model building
# #         attention_score_tensors = [scores for _, scores in self.attention_layers]
# #
# #         if not attention_score_tensors:
# #             print("Warning: No attention scores found in the model.")
# #             return None
# #
# #         # Create a model that outputs attention scores
# #         self.attention_model = models.Model(inputs=self.neural_network_model.input,
# #                                             outputs=attention_score_tensors,
# #                                             name='attention_extractor')
# #
# #         return self.attention_model
# #
# #     def extract_attention_weights(self, data_sample):
# #         """
# #         Extract attention weights for a given input sample.
# #
# #         Args:
# #             data_sample: Input data sample with shape matching model input.
# #
# #         Returns:
# #             list: List of attention weight arrays for each transformer block.
# #         """
# #         if self.attention_model is None:
# #             self.build_attention_model()
# #
# #         # Ensure the sample has batch dimension
# #         if len(data_sample.shape) == 3:
# #             data_sample = np.expand_dims(data_sample, axis=0)
# #
# #         # Get attention outputs
# #         attention_outputs = self.attention_model.predict(data_sample, verbose=0)
# #
# #         # Ensure it's a list even if single output
# #         if not isinstance(attention_outputs, list):
# #             attention_outputs = [attention_outputs]
# #
# #         return attention_outputs
# #
# #     def visualize_attention_flow(self, data_samples, labels=None, num_samples=4,
# #                                  output_dir='attention_visualizations'):
# #         """
# #         Visualize attention flow across all transformer blocks for multiple samples.
# #
# #         Args:
# #             data_samples: Input data samples (validation or test set).
# #             labels: Optional labels for the samples.
# #             num_samples: Number of samples to visualize.
# #             output_dir: Directory to save visualization plots.
# #         """
# #         # Create output directory
# #         output_path = Path(output_dir)
# #         output_path.mkdir(parents=True, exist_ok=True)
# #
# #         # Select random samples
# #         num_samples = min(num_samples, len(data_samples))
# #         sample_indices = np.random.choice(len(data_samples), num_samples, replace=False)
# #
# #         for idx, sample_idx in enumerate(sample_indices):
# #             sample = data_samples[sample_idx:sample_idx + 1]
# #
# #             # Extract attention weights
# #             attention_weights = self.extract_attention_weights(sample)
# #
# #             # Get model prediction
# #             prediction = self.neural_network_model.predict(sample, verbose=0)
# #             predicted_class = np.argmax(prediction[0])
# #             confidence = np.max(prediction[0])
# #
# #             true_class = np.argmax(labels[sample_idx]) if labels is not None else None
# #
# #             # Create unified comprehensive visualization
# #             self._plot_unified_attention_analysis(attention_weights,
# #                                                   sample_idx,
# #                                                   data_sample=sample,
# #                                                   predicted_class=predicted_class,
# #                                                   true_class=true_class,
# #                                                   confidence=confidence,
# #                                                   output_path=output_path)
# #
# #     def _plot_unified_attention_analysis(self, attention_weights, sample_idx, data_sample=None,
# #                                          predicted_class=None, true_class=None, confidence=None,
# #                                          output_path=None):
# #         """
# #         Create a unified comprehensive visualization with all essential attention analysis.
# #         """
# #         num_blocks = len(attention_weights)
# #
# #         # Create main figure with optimized layout
# #         fig = plt.figure(figsize=(24, 16))
# #         gs = fig.add_gridspec(4, num_blocks + 2, height_ratios=[2, 1.2, 1.2, 1.2],
# #                               hspace=0.4, wspace=0.3)
# #
# #         # Create title
# #         title_parts = [f'Análise Completa de Atenção - Amostra {sample_idx}']
# #         if predicted_class is not None:
# #             status = "✅" if (true_class is not None and predicted_class == true_class) else "❌"
# #             title_parts.append(f'\n{status} Predição: Classe {predicted_class}')
# #             if confidence is not None:
# #                 title_parts.append(f'(Confiança: {confidence:.1%})')
# #             if true_class is not None:
# #                 title_parts.append(f'| Real: Classe {true_class}')
# #
# #         fig.suptitle(' '.join(title_parts), fontsize=18, fontweight='bold', y=0.98)
# #         plt.style.use('seaborn-v0_8-whitegrid')
# #
# #         # Process attention weights
# #         processed_attentions = []
# #         for attention in attention_weights:
# #             if len(attention.shape) == 4:
# #                 attention_avg = np.mean(attention[0], axis=0)
# #             elif len(attention.shape) == 3:
# #                 if attention.shape[0] == self.number_heads:
# #                     attention_avg = np.mean(attention, axis=0)
# #                 else:
# #                     attention_avg = attention[0]
# #             elif len(attention.shape) == 2:
# #                 attention_avg = attention
# #             else:
# #                 continue
# #             processed_attentions.append(attention_avg)
# #
# #         if not processed_attentions:
# #             plt.close()
# #             return
# #
# #         # Input Spectrogram (leftmost)
# #         ax_spec = fig.add_subplot(gs[0, 0])
# #         if data_sample is not None:
# #             sample_data = data_sample[0]
# #             if len(sample_data.shape) == 3:
# #                 num_patches = sample_data.shape[0]
# #                 patch_height = sample_data.shape[1]
# #                 patch_width = sample_data.shape[2]
# #
# #                 number_patches_x = 5
# #                 number_patches_y = 64
# #
# #                 expected_patches = number_patches_x * number_patches_y
# #                 if num_patches != expected_patches:
# #                     number_patches_y = int(np.sqrt(num_patches))
# #                     while num_patches % number_patches_y != 0:
# #                         number_patches_y += 1
# #                     number_patches_x = num_patches // number_patches_y
# #
# #                 reconstructed_rows = []
# #                 patch_idx = 0
# #                 for freq_idx in range(number_patches_y):
# #                     row_patches = []
# #                     for time_idx in range(number_patches_x):
# #                         if patch_idx < num_patches:
# #                             row_patches.append(sample_data[patch_idx])
# #                             patch_idx += 1
# #                     if row_patches:
# #                         reconstructed_rows.append(np.hstack(row_patches))
# #
# #                 if reconstructed_rows:
# #                     reconstructed_spectrogram = np.vstack(reconstructed_rows)
# #                     im = ax_spec.imshow(reconstructed_spectrogram, cmap='magma', aspect='auto',
# #                                         origin='lower', interpolation='bilinear')
# #                     ax_spec.set_title(
# #                         f'🎵 Input Spectrogram\n{reconstructed_spectrogram.shape[0]}×{reconstructed_spectrogram.shape[1]}',
# #                         fontweight='bold', fontsize=11)
# #                     ax_spec.set_xlabel('Time', fontweight='bold', fontsize=9)
# #                     ax_spec.set_ylabel('Frequency', fontweight='bold', fontsize=9)
# #
# #                     for i in range(1, number_patches_x):
# #                         ax_spec.axvline(x=i * patch_height - 0.5, color='cyan',
# #                                         linestyle='-', linewidth=1.5, alpha=0.7)
# #
# #                     cbar = plt.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
# #                     cbar.set_label('Magnitude', rotation=270, labelpad=15, fontsize=9)
# #             elif len(sample_data.shape) == 2:
# #                 im = ax_spec.imshow(sample_data.T, cmap='magma', aspect='auto',
# #                                     origin='lower', interpolation='bilinear')
# #                 ax_spec.set_title('🎵 Input Spectrogram', fontweight='bold', fontsize=11)
# #                 cbar = plt.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
# #         else:
# #             ax_spec.text(0.5, 0.5, 'Spectrogram\nnot available',
# #                          ha='center', va='center', transform=ax_spec.transAxes, fontsize=10)
# #             ax_spec.set_title('🎵 Input Spectrogram', fontweight='bold', fontsize=11)
# #
# #         # Attention blocks (remaining columns in row 1)
# #         for block_idx, attention_avg in enumerate(processed_attentions):
# #             ax = fig.add_subplot(gs[0, block_idx + 1])
# #             im = ax.imshow(attention_avg, cmap='magma', aspect='auto', interpolation='bilinear',
# #                            vmin=0, vmax=attention_avg.max())
# #
# #             ax.set_title(f'Block {block_idx + 1}\n(μ={attention_avg.mean():.3f})',
# #                          fontweight='bold', fontsize=10, pad=8)
# #             ax.set_xlabel('Key', fontsize=8)
# #             if block_idx == 0:
# #                 ax.set_ylabel('Query', fontsize=8)
# #             ax.tick_params(labelsize=7)
# #
# #             cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# #             cbar.ax.tick_params(labelsize=7)
# #             cbar.set_label('Peso', rotation=270, labelpad=12, fontsize=8)
# #
# #             stats_text = f'Max: {attention_avg.max():.3f}\nMin: {attention_avg.min():.3f}'
# #             ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
# #                     fontsize=7, verticalalignment='top',
# #                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# #
# #         # ============= ROW 2: ATTENTION METRICS =============
# #
# #         # Average attention per block
# #         ax_avg = fig.add_subplot(gs[1, :num_blocks])
# #         avg_attentions = [np.mean(att) for att in processed_attentions]
# #         blocks = [f'B{i + 1}' for i in range(len(avg_attentions))]
# #         colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(blocks)))
# #         bars = ax_avg.bar(blocks, avg_attentions, color=colors, alpha=0.8,
# #                           edgecolor='black', linewidth=1.2)
# #
# #         for bar, val in zip(bars, avg_attentions):
# #             height = bar.get_height()
# #             ax_avg.text(bar.get_x() + bar.get_width() / 2., height,
# #                         f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
# #
# #         ax_avg.set_xlabel('Transformer Block', fontweight='bold', fontsize=10)
# #         ax_avg.set_ylabel('Atenção Média', fontweight='bold', fontsize=10)
# #         ax_avg.set_title('📊 Distribuição da Atenção Média por Bloco', fontweight='bold', fontsize=12, pad=10)
# #         ax_avg.grid(True, alpha=0.3, axis='y')
# #         ax_avg.set_ylim(0, max(avg_attentions) * 1.15)
# #
# #         # Attention Distribution Histogram
# #         ax_hist = fig.add_subplot(gs[1, num_blocks:])
# #         all_weights = np.concatenate([att.flatten() for att in processed_attentions])
# #         ax_hist.hist(all_weights, bins=50, color='#16A085', alpha=0.7,
# #                      edgecolor='black', linewidth=0.8)
# #         ax_hist.axvline(all_weights.mean(), color='red', linestyle='--', linewidth=2,
# #                         label=f'Média: {all_weights.mean():.3f}')
# #         ax_hist.axvline(np.median(all_weights), color='orange', linestyle='--', linewidth=2,
# #                         label=f'Mediana: {np.median(all_weights):.3f}')
# #
# #         ax_hist.set_title('📊 Distribuição dos Pesos', fontweight='bold', fontsize=12)
# #         ax_hist.set_xlabel('Peso de Atenção', fontweight='bold')
# #         ax_hist.set_ylabel('Frequência', fontweight='bold')
# #         ax_hist.legend(fontsize=9, framealpha=0.9)
# #         ax_hist.grid(True, alpha=0.3, axis='y')
# #
# #         # ============= ROW 3: ENTROPY + RANGE =============
# #
# #         # Attention entropy
# #         ax_entropy = fig.add_subplot(gs[2, :num_blocks])
# #         entropies = []
# #         for attention_avg in processed_attentions:
# #             attention_safe = attention_avg + 1e-10
# #             attention_norm = attention_safe / np.sum(attention_safe, axis=1, keepdims=True)
# #             entropy_per_query = -np.sum(attention_norm * np.log(attention_norm + 1e-10), axis=1)
# #             entropies.append(np.mean(entropy_per_query))
# #
# #         ax_entropy.plot(blocks, entropies, marker='o', linewidth=3, markersize=10,
# #                         color='#E74C3C', markeredgecolor='black', markeredgewidth=1.5, label='Entropia Média')
# #         ax_entropy.fill_between(range(len(blocks)), entropies, alpha=0.3, color='#E74C3C')
# #
# #         for i, (block, entropy) in enumerate(zip(blocks, entropies)):
# #             ax_entropy.text(i, entropy, f'{entropy:.2f}', ha='center', va='bottom',
# #                             fontsize=8, fontweight='bold')
# #
# #         ax_entropy.set_xlabel('Transformer Block', fontweight='bold', fontsize=10)
# #         ax_entropy.set_ylabel('Entropia Média', fontweight='bold', fontsize=10)
# #         ax_entropy.set_title('Diversidade da Atenção (Entropia) por Bloco',
# #                              fontweight='bold', fontsize=12, pad=10)
# #         ax_entropy.grid(True, alpha=0.3)
# #         ax_entropy.legend(loc='best', fontsize=9, framealpha=0.9)
# #
# #         # Attention Range Evolution
# #         ax_range = fig.add_subplot(gs[2, num_blocks:])
# #         attention_ranges = [att.max() - att.min() for att in processed_attentions]
# #         colors_range = plt.cm.viridis(np.linspace(0.3, 0.9, len(blocks)))
# #         bars_range = ax_range.bar(blocks, attention_ranges, color=colors_range, alpha=0.8,
# #                                   edgecolor='black', linewidth=1.2)
# #
# #         for bar, val in zip(bars_range, attention_ranges):
# #             height = bar.get_height()
# #             ax_range.text(bar.get_x() + bar.get_width() / 2., height,
# #                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
# #
# #         ax_range.set_title('Range de Atenção', fontweight='bold', fontsize=12)
# #         ax_range.set_xlabel('Transformer Block', fontweight='bold')
# #         ax_range.set_ylabel('Max - Min', fontweight='bold')
# #         ax_range.grid(True, alpha=0.3, axis='y')
# #
# #         # ============= ROW 4: ATTENTION CONCENTRATION + STATISTICS =============
# #
# #         # Attention concentration (max/min per block)
# #         ax_conc = fig.add_subplot(gs[3, :num_blocks])
# #         max_attentions = [np.max(att) for att in processed_attentions]
# #         min_attentions = [np.min(att) for att in processed_attentions]
# #
# #         ax_conc.plot(blocks, max_attentions, marker='s', linewidth=2.5, markersize=8,
# #                      color='#27AE60', markeredgecolor='black', markeredgewidth=1,
# #                      label='Máximo', alpha=0.8)
# #         ax_conc.plot(blocks, min_attentions, marker='s', linewidth=2.5, markersize=8,
# #                      color='#3498DB', markeredgecolor='black', markeredgewidth=1,
# #                      label='Mínimo', alpha=0.8)
# #         ax_conc.fill_between(range(len(blocks)), min_attentions, max_attentions,
# #                              alpha=0.2, color='gray')
# #
# #         ax_conc.set_xlabel('Transformer Block', fontweight='bold', fontsize=10)
# #         ax_conc.set_ylabel('Peso de Atenção', fontweight='bold', fontsize=10)
# #         ax_conc.set_title('📍 Range de Atenção por Bloco', fontweight='bold', fontsize=12, pad=10)
# #         ax_conc.grid(True, alpha=0.3)
# #         ax_conc.legend(loc='best', fontsize=9, framealpha=0.9)
# #
# #         # Statistics panel
# #         ax_stats = fig.add_subplot(gs[3, num_blocks:])
# #         ax_stats.axis('off')
# #
# #         stats_text = "ESTATÍSTICAS GLOBAIS\n" + "=" * 30 + "\n\n"
# #         all_attentions = np.concatenate([att.flatten() for att in processed_attentions])
# #         stats_text += f"Média Geral: {all_attentions.mean():.4f}\n"
# #         stats_text += f"Desvio Padrão: {all_attentions.std():.4f}\n"
# #         stats_text += f"Mediana: {np.median(all_attentions):.4f}\n"
# #         stats_text += f"Mínimo Global: {all_attentions.min():.4f}\n"
# #         stats_text += f"Máximo Global: {all_attentions.max():.4f}\n\n"
# #         stats_text += "=" * 30 + "\n\n"
# #         stats_text += "📊 POR BLOCO:\n\n"
# #
# #         for idx, att in enumerate(processed_attentions):
# #             stats_text += f"Block {idx + 1}:\n"
# #             stats_text += f"  • Média: {att.mean():.4f}\n"
# #             stats_text += f"  • Std: {att.std():.4f}\n"
# #             stats_text += f"  • Entropia: {entropies[idx]:.4f}\n"
# #             stats_text += f"  • Range: [{att.min():.4f}, {att.max():.4f}]\n\n"
# #
# #         ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
# #                       fontsize=9, verticalalignment='top', family='monospace',
# #                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=1))
# #
# #         plt.style.use('default')
# #
# #         if output_path:
# #             filename = f'unified_attention_analysis_sample_{sample_idx}'
# #             if predicted_class is not None:
# #                 filename += f'_pred{predicted_class}'
# #             plt.savefig(output_path / f'{filename}.png', dpi=200, bbox_inches='tight',
# #                         facecolor='white', edgecolor='none')
# #         plt.close()
# #
# #     def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
# #                           batch_size: int, validation_data: tuple = None,
# #                           visualize_attention: bool = True) -> tensorflow.keras.callbacks.History:
# #         """
# #         Compiles and trains the AST model using the specified training data and configuration.
# #         After training, generates attention flow visualizations if validation data is provided.
# #
# #         Args:
# #             train_data (tensorflow.Tensor): The input training data.
# #             train_labels (tensorflow.Tensor): The corresponding labels for the training data.
# #             epochs (int): Number of training epochs.
# #             batch_size (int): Size of the batches for each training step.
# #             validation_data (tuple, optional): A tuple containing validation data and labels.
# #             visualize_attention (bool): Whether to generate attention visualizations after training.
# #
# #         Returns:
# #             tensorflow.keras.callbacks.History: The history object containing training metrics and performance.
# #         """
# #
# #         # Compile the model with the specified optimizer, loss function, and metrics
# #         self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
# #                                           metrics=['accuracy'])
# #
# #         # Train the model
# #         training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
# #                                                          batch_size=batch_size,
# #                                                          validation_data=validation_data
# #                                                          )
# #
# #         # Generate attention visualizations after training
# #         if visualize_attention and validation_data is not None:
# #             val_data, val_labels = validation_data
# #             self.visualize_attention_flow(val_data, labels=val_labels, num_samples=128, output_dir='MAPS_AST')
# #
# #         return training_history
# #
# #     @property
# #     def head_size(self):
# #         return self._head_size
# #
# #     @head_size.setter
# #     def head_size(self, head_size: int):
# #         self._head_size = head_size
# #
# #     @property
# #     def number_heads(self):
# #         return self._number_heads
# #
# #     @number_heads.setter
# #     def number_heads(self, num_heads: int):
# #         self._number_heads = num_heads
# #
# #     @property
# #     def number_blocks(self):
# #         return self._number_blocks
# #
# #     @number_blocks.setter
# #     def number_blocks(self, number_blocks: int):
# #         self._number_blocks = number_blocks
# #
# #     @property
# #     def number_classes(self):
# #         return self._number_classes
# #
# #     @number_classes.setter
# #     def number_classes(self, number_classes: int):
# #         self._number_classes = number_classes
# #
# #     @property
# #     def patch_size(self):
# #         return self._patch_size
# #
# #     @patch_size.setter
# #     def patch_size(self, patch_size: tuple):
# #         self._patch_size = patch_size
# #
# #     @property
# #     def dropout(self):
# #         return self._dropout
# #
# #     @dropout.setter
# #     def dropout(self, dropout: float):
# #         self._dropout = dropout
# #
# #     @property
# #     def optimizer_function(self):
# #         return self._optimizer_function
# #
# #     @optimizer_function.setter
# #     def optimizer_function(self, optimizer_function: str):
# #         self._optimizer_function = optimizer_function
# #
# #     @property
# #     def loss_function(self):
# #         return self._loss_function
# #
# #     @loss_function.setter
# #     def loss_function(self, loss_function: str):
# #         self._loss_function = loss_function
# #
# #     @property
# #     def normalization_epsilon(self):
# #         return self._normalization_epsilon
# #
# #     @normalization_epsilon.setter
# #     def normalization_epsilon(self, normalization_epsilon: float):
# #         self._normalization_epsilon = normalization_epsilon
# #
# #     @property
# #     def last_activation_layer(self):
# #         return self._last_activation_layer
# #
# #     @last_activation_layer.setter
# #     def last_activation_layer(self, last_activation_layer: str):
# #         self._last_activation_layer = last_activation_layer
# #
# #     @property
# #     def projection_dimension(self):
# #         return self._projection_dimension
# #
# #     @projection_dimension.setter
# #     def projection_dimension(self, projection_dimension: int):
# #         self._projection_dimension = projection_dimension
# #
# #     @property
# #     def intermediary_activation(self):
# #         return self._intermediary_activation
# #
# #     @intermediary_activation.setter
# #     def intermediary_activation(self, intermediary_activation: str):
# #         self._intermediary_activation = intermediary_activation
# #
# #     @property
# #     def number_filters_spectrogram(self):
# #         return self._number_filters_spectrogram
# #
# #     @number_filters_spectrogram.setter
# #     def number_filters_spectrogram(self, number_filters_spectrogram: int):
# #         self._number_filters_spectrogram = number_filters_spectrogram
#
#
# # !/usr/bin/python3
# # -*- coding: utf-8 -*-
#
# """
# AudioSpectrogramTransformer com Distillation Token
# Implementação mais fiel ao paper original AST (Gong et al., 2021)
# que usa tanto CLS quanto Distillation tokens do DeiT.
# """
#
# __author__ = 'Kayuã Oleques Paim'
# __version__ = '{1}.{0}.{3}'
# __last_update__ = '2025/10/18'
#
# try:
#     import tensorflow
#     from tensorflow.keras import models
#     from tensorflow.keras.layers import (
#         Add, Input, Layer, Dense, Dropout, Flatten,
#         TimeDistributed, LayerNormalization, MultiHeadAttention
#     )
#
#     from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
#     from Engine.Models.Process.AST_Process import ProcessAST
#
# except ImportError as error:
#     print(error)
#     import sys
#
#     sys.exit(-1)
#
#
# class DistillationCLSTokenLayer(Layer):
#     """
#     Adiciona CLS token E Distillation token ao início da sequência.
#
#     O AST original usa DeiT (Data-efficient Image Transformer) que introduziu
#     o distillation token para melhorar o treinamento com knowledge distillation.
#
#     Durante treinamento: ambos os tokens são usados (com duas heads de classificação)
#     Durante inferência: média das predições dos dois tokens
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.cls_token = None
#         self.dist_token = None
#
#     def build(self, input_shape):
#         embedding_dim = input_shape[-1]
#
#         # CLS token (para classificação)
#         self.cls_token = self.add_weight(
#             shape=(1, 1, embedding_dim),
#             initializer='random_normal',
#             trainable=True,
#             name='cls_token'
#         )
#
#         # Distillation token (para knowledge distillation)
#         self.dist_token = self.add_weight(
#             shape=(1, 1, embedding_dim),
#             initializer='random_normal',
#             trainable=True,
#             name='distillation_token'
#         )
#
#         super().build(input_shape)
#
#     def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
#         batch_size = tensorflow.shape(inputs)[0]
#
#         # Tile tokens para o batch size
#         cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])
#         dist_tokens = tensorflow.tile(self.dist_token, [batch_size, 1, 1])
#
#         # Concatenar: [CLS, DIST, patch1, patch2, ...]
#         return tensorflow.concat([cls_tokens, dist_tokens, inputs], axis=1)
#
#     def get_config(self):
#         return super().get_config()
#
#
# class AudioSpectrogramTransformer(ProcessAST):
#     """
#     Audio Spectrogram Transformer com Distillation Token.
#
#     Diferenças do AST básico:
#     - Adiciona distillation token além do CLS token
#     - Usa duas classification heads (uma para cada token)
#     - Durante inferência, faz média das predições
#
#     Esta implementação é mais fiel ao AST original que usa DeiT como base.
#     """
#
#     def __init__(self, arguments):
#         super().__init__(arguments)
#         self.neural_network_model = None
#         self.attention_model = None
#         self.attention_layers = []
#         self.head_size = arguments.ast_head_size
#         self.number_heads = arguments.ast_head_size
#         self.number_blocks = arguments.ast_number_blocks
#         self.number_classes = arguments.number_classes
#         self.patch_size = arguments.ast_patch_size
#         self.dropout = arguments.ast_dropout
#         self.optimizer_function = arguments.ast_optimizer_function
#         self.loss_function = arguments.ast_loss_function
#         self.normalization_epsilon = arguments.ast_normalization_epsilon
#         self.last_activation_layer = arguments.ast_intermediary_activation
#         self.projection_dimension = arguments.ast_projection_dimension
#         self.intermediary_activation = arguments.ast_intermediary_activation
#         self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram
#         self.model_name = "AST_Distillation"
#         self.use_distillation = True  # Flag para ativar/desativar distillation
#
#     def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
#         """
#         Transformer encoder com Pre-LN (como no ViT/AST original).
#         """
#         x = inputs
#
#         for block_idx in range(self.number_blocks):
#             # Multi-Head Attention Block com Pre-LN
#             x_norm = LayerNormalization(
#                 epsilon=self.normalization_epsilon,
#                 name=f'norm1_block_{block_idx}'
#             )(x)
#
#             attention_layer = MultiHeadAttention(
#                 key_dim=self.projection_dimension,
#                 num_heads=self.number_heads,
#                 dropout=self.dropout,
#                 name=f'attention_block_{block_idx}'
#             )
#
#             attention_output, attention_scores = attention_layer(
#                 x_norm, x_norm,
#                 return_attention_scores=True
#             )
#
#             self.attention_layers.append((attention_layer, attention_scores))
#             attention_output = Dropout(self.dropout, name=f'dropout1_block_{block_idx}')(attention_output)
#             x = Add(name=f'add1_block_{block_idx}')([attention_output, x])
#
#             # Feed-Forward Network com Pre-LN
#             x_norm = LayerNormalization(
#                 epsilon=self.normalization_epsilon,
#                 name=f'norm2_block_{block_idx}'
#             )(x)
#
#             ffn_dim = self.projection_dimension * 4
#             ffn_output = Dense(
#                 ffn_dim,
#                 activation=self.intermediary_activation,
#                 name=f'ffn1_block_{block_idx}'
#             )(x_norm)
#             ffn_output = Dropout(self.dropout, name=f'dropout2_block_{block_idx}')(ffn_output)
#             ffn_output = Dense(self.projection_dimension, name=f'ffn2_block_{block_idx}')(ffn_output)
#             ffn_output = Dropout(self.dropout, name=f'dropout3_block_{block_idx}')(ffn_output)
#             x = Add(name=f'add2_block_{block_idx}')([ffn_output, x])
#
#         # Layer Normalization final
#         x = LayerNormalization(epsilon=self.normalization_epsilon, name='final_norm')(x)
#         return x
#
#     def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
#         """
#         Constrói o modelo AST com distillation token.
#
#         Arquitetura:
#         1. Patches de entrada são linearmente projetados
#         2. CLS e Distillation tokens são adicionados
#         3. Positional embeddings são somados
#         4. Transformer encoder processa a sequência
#         5. Duas classification heads (uma para CLS, outra para Dist)
#         6. Output final é a média das duas predições
#
#         Args:
#             number_patches (int): Número de patches do espectrograma
#         """
#         self.attention_layers = []
#
#         # Input: patches do espectrograma
#         inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
#         input_flatten = TimeDistributed(Flatten())(inputs)
#         linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)
#
#         # Adicionar CLS e Distillation tokens
#         if self.use_distillation:
#             # [CLS, DIST, patch1, patch2, ..., patchN]
#             neural_model_flow = DistillationCLSTokenLayer()(linear_projection)
#             num_special_tokens = 2
#         else:
#             # Fallback: apenas CLS token
#             from Engine.Layers.CLSTokenLayer import CLSTokenLayer
#             neural_model_flow = CLSTokenLayer()(linear_projection)
#             num_special_tokens = 1
#
#         # Positional embeddings para TODOS os tokens (CLS + DIST + patches)
#         # O PositionalEmbeddingsLayer deve criar embeddings para number_patches + num_special_tokens posições
#         positional_embeddings_layer = PositionalEmbeddingsLayer(
#             number_patches + num_special_tokens - 1,  # Ajuste baseado na sua implementação
#             self.projection_dimension
#         )(neural_model_flow)
#         neural_model_flow = neural_model_flow + positional_embeddings_layer
#
#         # Transformer encoder
#         neural_model_flow = self.transformer_encoder(neural_model_flow)
#
#         if self.use_distillation:
#             # Extrair CLS token (posição 0) e Distillation token (posição 1)
#             cls_output = neural_model_flow[:, 0, :]
#             dist_output = neural_model_flow[:, 1, :]
#
#             # Duas classification heads
#             cls_output = Dropout(self.dropout, name='cls_dropout')(cls_output)
#             dist_output = Dropout(self.dropout, name='dist_dropout')(dist_output)
#
#             # MLP heads
#             cls_logits = Dense(
#                 self.projection_dimension,
#                 activation=self.intermediary_activation,
#                 name='cls_mlp_hidden'
#             )(cls_output)
#             cls_logits = Dropout(self.dropout)(cls_logits)
#             cls_logits = Dense(
#                 self.number_classes,
#                 activation='softmax',
#                 name='cls_head'
#             )(cls_logits)
#
#             dist_logits = Dense(
#                 self.projection_dimension,
#                 activation=self.intermediary_activation,
#                 name='dist_mlp_hidden'
#             )(dist_output)
#             dist_logits = Dropout(self.dropout)(dist_logits)
#             dist_logits = Dense(
#                 self.number_classes,
#                 activation='softmax',
#                 name='dist_head'
#             )(dist_logits)
#
#             # Média das duas predições (como no AST/DeiT original)
#             outputs = tensorflow.keras.layers.Average(name='average_predictions')([cls_logits, dist_logits])
#
#         else:
#             # Apenas CLS token
#             cls_output = neural_model_flow[:, 0, :]
#             cls_output = Dropout(self.dropout)(cls_output)
#
#             mlp_hidden = Dense(
#                 self.projection_dimension,
#                 activation=self.intermediary_activation,
#                 name='mlp_head_hidden'
#             )(cls_output)
#             mlp_hidden = Dropout(self.dropout)(mlp_hidden)
#
#             outputs = Dense(
#                 self.number_classes,
#                 activation='softmax',
#                 name='classification_head'
#             )(mlp_hidden)
#
#         # Criar modelo
#         self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
#         self.neural_network_model.summary()
#
#         return self.neural_network_model
#
#     def build_attention_model(self):
#         """Constrói modelo para extrair attention weights."""
#         if self.neural_network_model is None:
#             raise ValueError("The main model must be built before creating attention model.")
#
#         attention_score_tensors = [scores for _, scores in self.attention_layers]
#
#         if not attention_score_tensors:
#             print("Warning: No attention scores found in the model.")
#             return None
#
#         self.attention_model = models.Model(
#             inputs=self.neural_network_model.input,
#             outputs=attention_score_tensors,
#             name='attention_extractor'
#         )
#
#         return self.attention_model
#
#     def compile_and_train(self, train_data, train_labels, epochs: int,
#                           batch_size: int, validation_data=None,
#                           visualize_attention: bool = True):
#         """
#         Compila e treina o modelo.
#
#         Nota: Para usar distillation training adequadamente, você precisaria
#         de um teacher model e implementar a loss de distillation. Esta
#         implementação usa apenas a arquitetura com dois tokens.
#         """
#         self.neural_network_model.compile(
#             optimizer=self.optimizer_function,
#             loss=self.loss_function,
#             metrics=['accuracy']
#         )
#
#         training_history = self.neural_network_model.fit(
#             train_data, train_labels,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_data=validation_data
#         )
#
#         if visualize_attention and validation_data is not None:
#             val_data, val_labels = validation_data
#             # Métodos de visualização herdados da classe base
#             if hasattr(self, 'visualize_attention_flow'):
#                 self.visualize_attention_flow(
#                     val_data,
#                     labels=val_labels,
#                     num_samples=128,
#                     output_dir='MAPS_AST_DISTILLATION'
#                 )
#
#         return training_history
#
#     # Properties (herdados da classe base)
#     @property
#     def head_size(self):
#         return self._head_size
#
#     @head_size.setter
#     def head_size(self, head_size: int):
#         self._head_size = head_size
#
#     @property
#     def number_heads(self):
#         return self._number_heads
#
#     @number_heads.setter
#     def number_heads(self, num_heads: int):
#         self._number_heads = num_heads
#
#     @property
#     def number_blocks(self):
#         return self._number_blocks
#
#     @number_blocks.setter
#     def number_blocks(self, number_blocks: int):
#         self._number_blocks = number_blocks
#
#     @property
#     def number_classes(self):
#         return self._number_classes
#
#     @number_classes.setter
#     def number_classes(self, number_classes: int):
#         self._number_classes = number_classes
#
#     @property
#     def patch_size(self):
#         return self._patch_size
#
#     @patch_size.setter
#     def patch_size(self, patch_size: tuple):
#         self._patch_size = patch_size
#
#     @property
#     def dropout(self):
#         return self._dropout
#
#     @dropout.setter
#     def dropout(self, dropout: float):
#         self._dropout = dropout
#
#     @property
#     def optimizer_function(self):
#         return self._optimizer_function
#
#     @optimizer_function.setter
#     def optimizer_function(self, optimizer_function: str):
#         self._optimizer_function = optimizer_function
#
#     @property
#     def loss_function(self):
#         return self._loss_function
#
#     @loss_function.setter
#     def loss_function(self, loss_function: str):
#         self._loss_function = loss_function
#
#     @property
#     def normalization_epsilon(self):
#         return self._normalization_epsilon
#
#     @normalization_epsilon.setter
#     def normalization_epsilon(self, normalization_epsilon: float):
#         self._normalization_epsilon = normalization_epsilon
#
#     @property
#     def last_activation_layer(self):
#         return self._last_activation_layer
#
#     @last_activation_layer.setter
#     def last_activation_layer(self, last_activation_layer: str):
#         self._last_activation_layer = last_activation_layer
#
#     @property
#     def projection_dimension(self):
#         return self._projection_dimension
#
#     @projection_dimension.setter
#     def projection_dimension(self, projection_dimension: int):
#         self._projection_dimension = projection_dimension
#
#     @property
#     def intermediary_activation(self):
#         return self._intermediary_activation
#
#     @intermediary_activation.setter
#     def intermediary_activation(self, intermediary_activation: str):
#         self._intermediary_activation = intermediary_activation
#
#     @property
#     def number_filters_spectrogram(self):
#         return self._number_filters_spectrogram
#
#     @number_filters_spectrogram.setter
#     def number_filters_spectrogram(self, number_filters_spectrogram: int):
#         self._number_filters_spectrogram = number_filters_spectrogram


# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
AudioSpectrogramTransformer com Distillation Token
Implementação com visualizações de Attention Flow e Attention Rollout
"""

__author__ = 'Kayuã Oleques Paim'
__version__ = '{1}.{0}.{4}'
__last_update__ = '2025/10/18'

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import tensorflow
    from tensorflow.keras import models
    from tensorflow.keras.layers import (
        Add, Input, Layer, Dense, Dropout, Flatten,
        TimeDistributed, LayerNormalization, MultiHeadAttention
    )

    from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
    from Engine.Models.Process.AST_Process import ProcessAST

except ImportError as error:
    print(error)
    import sys

    sys.exit(-1)


class DistillationCLSTokenLayer(Layer):
    """
    Adiciona CLS token E Distillation token ao início da sequência.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = None
        self.dist_token = None

    def build(self, input_shape):
        embedding_dim = input_shape[-1]

        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

        self.dist_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='distillation_token'
        )

        super().build(input_shape)

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        batch_size = tensorflow.shape(inputs)[0]
        cls_tokens = tensorflow.tile(self.cls_token, [batch_size, 1, 1])
        dist_tokens = tensorflow.tile(self.dist_token, [batch_size, 1, 1])
        return tensorflow.concat([cls_tokens, dist_tokens, inputs], axis=1)

    def get_config(self):
        return super().get_config()


class AudioSpectrogramTransformer(ProcessAST):
    """
    Audio Spectrogram Transformer com Distillation Token.
    Inclui visualizações de Attention Flow e Attention Rollout.
    """

    def __init__(self, arguments):
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
        """Transformer encoder com Pre-LN."""
        x = inputs

        for block_idx in range(self.number_blocks):
            x_norm = LayerNormalization(
                epsilon=self.normalization_epsilon,
                name=f'norm1_block_{block_idx}'
            )(x)

            attention_layer = MultiHeadAttention(
                key_dim=self.projection_dimension,
                num_heads=self.number_heads,
                dropout=self.dropout,
                name=f'attention_block_{block_idx}'
            )

            attention_output, attention_scores = attention_layer(
                x_norm, x_norm,
                return_attention_scores=True
            )

            self.attention_layers.append((attention_layer, attention_scores))
            attention_output = Dropout(self.dropout, name=f'dropout1_block_{block_idx}')(attention_output)
            x = Add(name=f'add1_block_{block_idx}')([attention_output, x])

            x_norm = LayerNormalization(
                epsilon=self.normalization_epsilon,
                name=f'norm2_block_{block_idx}'
            )(x)

            ffn_dim = self.projection_dimension * 4
            ffn_output = Dense(
                ffn_dim,
                activation=self.intermediary_activation,
                name=f'ffn1_block_{block_idx}'
            )(x_norm)
            ffn_output = Dropout(self.dropout, name=f'dropout2_block_{block_idx}')(ffn_output)
            ffn_output = Dense(self.projection_dimension, name=f'ffn2_block_{block_idx}')(ffn_output)
            ffn_output = Dropout(self.dropout, name=f'dropout3_block_{block_idx}')(ffn_output)
            x = Add(name=f'add2_block_{block_idx}')([ffn_output, x])

        x = LayerNormalization(epsilon=self.normalization_epsilon, name='final_norm')(x)
        return x

    def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
        """Constrói o modelo AST com distillation token."""
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

        positional_embeddings_layer = PositionalEmbeddingsLayer(
            number_patches + num_special_tokens - 1,
            self.projection_dimension
        )(neural_model_flow)
        neural_model_flow = neural_model_flow + positional_embeddings_layer

        neural_model_flow = self.transformer_encoder(neural_model_flow)

        if self.use_distillation:
            cls_output = neural_model_flow[:, 0, :]
            dist_output = neural_model_flow[:, 1, :]

            cls_output = Dropout(self.dropout, name='cls_dropout')(cls_output)
            dist_output = Dropout(self.dropout, name='dist_dropout')(dist_output)

            cls_logits = Dense(
                self.projection_dimension,
                activation=self.intermediary_activation,
                name='cls_mlp_hidden'
            )(cls_output)
            cls_logits = Dropout(self.dropout)(cls_logits)
            cls_logits = Dense(
                self.number_classes,
                activation='softmax',
                name='cls_head'
            )(cls_logits)

            dist_logits = Dense(
                self.projection_dimension,
                activation=self.intermediary_activation,
                name='dist_mlp_hidden'
            )(dist_output)
            dist_logits = Dropout(self.dropout)(dist_logits)
            dist_logits = Dense(
                self.number_classes,
                activation='softmax',
                name='dist_head'
            )(dist_logits)

            outputs = tensorflow.keras.layers.Average(name='average_predictions')([cls_logits, dist_logits])

        else:
            cls_output = neural_model_flow[:, 0, :]
            cls_output = Dropout(self.dropout)(cls_output)

            mlp_hidden = Dense(
                self.projection_dimension,
                activation=self.intermediary_activation,
                name='mlp_head_hidden'
            )(cls_output)
            mlp_hidden = Dropout(self.dropout)(mlp_hidden)

            outputs = Dense(
                self.number_classes,
                activation='softmax',
                name='classification_head'
            )(mlp_hidden)

        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
        self.neural_network_model.summary()

        return self.neural_network_model

    def build_attention_model(self):
        """Constrói modelo para extrair attention weights."""
        if self.neural_network_model is None:
            raise ValueError("The main model must be built before creating attention model.")

        attention_score_tensors = [scores for _, scores in self.attention_layers]

        if not attention_score_tensors:
            print("Warning: No attention scores found in the model.")
            return None

        self.attention_model = models.Model(
            inputs=self.neural_network_model.input,
            outputs=attention_score_tensors,
            name='attention_extractor'
        )

        return self.attention_model

    def extract_attention_weights(self, data_sample):
        """Extract attention weights for a given input sample."""
        if self.attention_model is None:
            self.build_attention_model()

        if len(data_sample.shape) == 3:
            data_sample = np.expand_dims(data_sample, axis=0)

        attention_outputs = self.attention_model.predict(data_sample, verbose=0)

        if not isinstance(attention_outputs, list):
            attention_outputs = [attention_outputs]

        return attention_outputs

    def compute_attention_rollout(self, attention_weights):
        """
        Computa o Attention Rollout através de todas as camadas.

        Rollout mostra como a atenção se acumula através das camadas,
        multiplicando as matrizes de atenção sucessivamente.

        Args:
            attention_weights: Lista de matrizes de atenção de cada camada

        Returns:
            np.ndarray: Matriz de rollout final
        """
        # Processar attention weights
        processed_attentions = []
        for attention in attention_weights:
            if len(attention.shape) == 4:
                # [batch, heads, query, key] -> média sobre heads
                attention_avg = np.mean(attention[0], axis=0)
            elif len(attention.shape) == 3:
                if attention.shape[0] == self.number_heads:
                    attention_avg = np.mean(attention, axis=0)
                else:
                    attention_avg = attention[0]
            elif len(attention.shape) == 2:
                attention_avg = attention
            else:
                continue
            processed_attentions.append(attention_avg)

        if not processed_attentions:
            return None

        # Adicionar identidade residual (skip connection)
        # A = 0.5 * Attention + 0.5 * Identity
        rollout = np.eye(processed_attentions[0].shape[0])

        for attention_matrix in processed_attentions:
            # Normalizar para que cada linha some 1
            attention_normalized = attention_matrix / (attention_matrix.sum(axis=-1, keepdims=True) + 1e-10)

            # Adicionar identidade para representar skip connections
            attention_with_residual = 0.5 * attention_normalized + 0.5 * np.eye(attention_normalized.shape[0])

            # Multiplicar acumulativamente
            rollout = np.matmul(attention_with_residual, rollout)

        return rollout

    def compute_attention_flow(self, attention_weights, target_token=0):
        """
        Computa o Attention Flow para um token específico.

        Flow mostra quanto cada patch contribui para a predição final,
        seguindo o caminho de atenção do token de classe (CLS).

        Args:
            attention_weights: Lista de matrizes de atenção
            target_token: Índice do token alvo (0 = CLS, 1 = DIST)

        Returns:
            np.ndarray: Vetor de flow para cada posição
        """
        rollout = self.compute_attention_rollout(attention_weights)

        if rollout is None:
            return None

        # Extrair atenção do token alvo para todos os outros tokens
        flow = rollout[target_token, :]

        return flow

    def visualize_attention_flow(self, data_samples, labels=None, num_samples=4,
                                 output_dir='attention_visualizations'):
        """
        Visualiza attention flow com rollout para múltiplas amostras.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_samples = min(num_samples, len(data_samples))
        sample_indices = np.random.choice(len(data_samples), num_samples, replace=False)

        for idx, sample_idx in enumerate(sample_indices):
            sample = data_samples[sample_idx:sample_idx + 1]

            # Extrair attention weights
            attention_weights = self.extract_attention_weights(sample)

            # Predição do modelo
            prediction = self.neural_network_model.predict(sample, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            true_class = np.argmax(labels[sample_idx]) if labels is not None else None

            # Criar visualização completa
            self._plot_unified_attention_analysis(
                attention_weights,
                sample_idx,
                data_sample=sample,
                predicted_class=predicted_class,
                true_class=true_class,
                confidence=confidence,
                output_path=output_path
            )

    def _reconstruct_spectrogram(self, sample_data):
        """
        Reconstrói o espectrograma a partir dos patches.
        """
        if len(sample_data.shape) != 3:
            return None, None

        num_patches = sample_data.shape[0]
        patch_height = sample_data.shape[1]
        patch_width = sample_data.shape[2]

        # Tentar configuração padrão
        number_patches_x = 5
        number_patches_y = 64

        expected_patches = number_patches_x * number_patches_y
        if num_patches != expected_patches:
            # Calcular dimensões automaticamente
            number_patches_y = int(np.sqrt(num_patches))
            while num_patches % number_patches_y != 0:
                number_patches_y += 1
            number_patches_x = num_patches // number_patches_y

        # Reconstruir o espectrograma
        reconstructed_rows = []
        patch_idx = 0
        for freq_idx in range(number_patches_y):
            row_patches = []
            for time_idx in range(number_patches_x):
                if patch_idx < num_patches:
                    row_patches.append(sample_data[patch_idx])
                    patch_idx += 1
            if row_patches:
                reconstructed_rows.append(np.hstack(row_patches))

        if reconstructed_rows:
            reconstructed_spectrogram = np.vstack(reconstructed_rows)
            return reconstructed_spectrogram, (number_patches_x, number_patches_y)

        return None, None

    def _plot_unified_attention_analysis(self, attention_weights, sample_idx, data_sample=None,
                                         predicted_class=None, true_class=None, confidence=None,
                                         output_path=None):
        """
        Visualização unificada com Spectrogram, Attention Flow e Attention Rollout.
        """
        num_blocks = len(attention_weights)

        # Criar figura com layout otimizado
        fig = plt.figure(figsize=(28, 18))
        gs = fig.add_gridspec(5, num_blocks + 2, height_ratios=[2, 2, 1.5, 1.5, 1.5],
                              hspace=0.4, wspace=0.3)

        # Título
        title_parts = [f'Análise Completa de Atenção - Amostra {sample_idx}']
        if predicted_class is not None:
            status = "✅" if (true_class is not None and predicted_class == true_class) else "❌"
            title_parts.append(f'\n{status} Predição: Classe {predicted_class}')
            if confidence is not None:
                title_parts.append(f'(Confiança: {confidence:.1%})')
            if true_class is not None:
                title_parts.append(f'| Real: Classe {true_class}')

        fig.suptitle(' '.join(title_parts), fontsize=20, fontweight='bold', y=0.98)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Processar attention weights
        processed_attentions = []
        for attention in attention_weights:
            if len(attention.shape) == 4:
                attention_avg = np.mean(attention[0], axis=0)
            elif len(attention.shape) == 3:
                if attention.shape[0] == self.number_heads:
                    attention_avg = np.mean(attention, axis=0)
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

        # ============= ROW 1: INPUT SPECTROGRAM =============
        ax_spec = fig.add_subplot(gs[0, 0])
        reconstructed_spectrogram = None

        if data_sample is not None:
            sample_data = data_sample[0]
            reconstructed_spectrogram, patch_config = self._reconstruct_spectrogram(sample_data)

            if reconstructed_spectrogram is not None:
                im = ax_spec.imshow(reconstructed_spectrogram, cmap='magma', aspect='auto',
                                    origin='lower', interpolation='bilinear')
                ax_spec.set_title(
                    f'🎵 Input Spectrogram\n{reconstructed_spectrogram.shape[0]}×{reconstructed_spectrogram.shape[1]}',
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

        # ============= ROW 1: ATTENTION ROLLOUT =============
        ax_rollout = fig.add_subplot(gs[0, 1])
        rollout = self.compute_attention_rollout(attention_weights)

        if rollout is not None:
            im_rollout = ax_rollout.imshow(rollout, cmap='viridis', aspect='auto',
                                           interpolation='bilinear', vmin=0, vmax=rollout.max())
            ax_rollout.set_title('🔄 Attention Rollout\n(Acumulação de Atenção)',
                                 fontweight='bold', fontsize=11)
            ax_rollout.set_xlabel('Token Position', fontsize=9)
            ax_rollout.set_ylabel('Token Position', fontsize=9)

            cbar = plt.colorbar(im_rollout, ax=ax_rollout, fraction=0.046, pad=0.04)
            cbar.set_label('Rollout Weight', rotation=270, labelpad=15, fontsize=9)

        # ============= ROW 2: CLS ATTENTION FLOW =============
        ax_flow_cls = fig.add_subplot(gs[1, 0])
        flow_cls = self.compute_attention_flow(attention_weights, target_token=0)

        if flow_cls is not None and reconstructed_spectrogram is not None:
            # Remover CLS e DIST tokens do flow (se usar distillation)
            num_special_tokens = 2 if self.use_distillation else 1
            flow_patches = flow_cls[num_special_tokens:]

            # Reshape flow para a forma do espectrograma
            if patch_config:
                number_patches_x, number_patches_y = patch_config
                flow_reshaped = flow_patches[:number_patches_x * number_patches_y].reshape(
                    number_patches_y, number_patches_x
                )

                # Upsample para o tamanho do espectrograma
                patch_height = sample_data.shape[1]
                patch_width = sample_data.shape[2]
                flow_upsampled = np.repeat(np.repeat(flow_reshaped, patch_height, axis=0),
                                           patch_width, axis=1)

                im_flow = ax_flow_cls.imshow(flow_upsampled, cmap='hot', aspect='auto',
                                             origin='lower', interpolation='bilinear', alpha=0.8)
                ax_flow_cls.set_title('🎯 CLS Token Attention Flow\n(Importância dos Patches)',
                                      fontweight='bold', fontsize=11)
                ax_flow_cls.set_xlabel('Time', fontweight='bold', fontsize=9)
                ax_flow_cls.set_ylabel('Frequency', fontweight='bold', fontsize=9)

                cbar = plt.colorbar(im_flow, ax=ax_flow_cls, fraction=0.046, pad=0.04)
                cbar.set_label('Flow Weight', rotation=270, labelpad=15, fontsize=9)

        # ============= ROW 2: DIST ATTENTION FLOW (se usar distillation) =============
        if self.use_distillation:
            ax_flow_dist = fig.add_subplot(gs[1, 1])
            flow_dist = self.compute_attention_flow(attention_weights, target_token=1)

            if flow_dist is not None and reconstructed_spectrogram is not None:
                flow_patches = flow_dist[2:]  # Remover CLS e DIST

                if patch_config:
                    number_patches_x, number_patches_y = patch_config
                    flow_reshaped = flow_patches[:number_patches_x * number_patches_y].reshape(
                        number_patches_y, number_patches_x
                    )

                    patch_height = sample_data.shape[1]
                    patch_width = sample_data.shape[2]
                    flow_upsampled = np.repeat(np.repeat(flow_reshaped, patch_height, axis=0),
                                               patch_width, axis=1)

                    im_flow = ax_flow_dist.imshow(flow_upsampled, cmap='plasma', aspect='auto',
                                                  origin='lower', interpolation='bilinear', alpha=0.8)
                    ax_flow_dist.set_title('🎯 DIST Token Attention Flow\n(Importância dos Patches)',
                                           fontweight='bold', fontsize=11)
                    ax_flow_dist.set_xlabel('Time', fontweight='bold', fontsize=9)
                    ax_flow_dist.set_ylabel('Frequency', fontweight='bold', fontsize=9)

                    cbar = plt.colorbar(im_flow, ax=ax_flow_dist, fraction=0.046, pad=0.04)
                    cbar.set_label('Flow Weight', rotation=270, labelpad=15, fontsize=9)

        # ============= ROW 1-2: ATTENTION BLOCKS =============
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

        # ============= ROW 3: MÉTRICAS =============
        ax_avg = fig.add_subplot(gs[2, :num_blocks])
        avg_attentions = [np.mean(att) for att in processed_attentions]
        blocks = [f'B{i + 1}' for i in range(len(avg_attentions))]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(blocks)))
        bars = ax_avg.bar(blocks, avg_attentions, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.2)

        for bar, val in zip(bars, avg_attentions):
            height = bar.get_height()
            ax_avg.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax_avg.set_xlabel('Transformer Block', fontweight='bold', fontsize=10)
        ax_avg.set_ylabel('Atenção Média', fontweight='bold', fontsize=10)
        ax_avg.set_title('📊 Distribuição da Atenção Média por Bloco', fontweight='bold', fontsize=12, pad=10)
        ax_avg.grid(True, alpha=0.3, axis='y')

        # Histograma
        ax_hist = fig.add_subplot(gs[2, num_blocks:])
        all_weights = np.concatenate([att.flatten() for att in processed_attentions])
        ax_hist.hist(all_weights, bins=50, color='#16A085', alpha=0.7,
                     edgecolor='black', linewidth=0.8)
        ax_hist.axvline(all_weights.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Média: {all_weights.mean():.3f}')
        ax_hist.set_title('📊 Distribuição dos Pesos', fontweight='bold', fontsize=12)
        ax_hist.legend(fontsize=9)

        # ============= ROW 4: ENTROPIA =============
        ax_entropy = fig.add_subplot(gs[3, :num_blocks])
        entropies = []
        for attention_avg in processed_attentions:
            attention_safe = attention_avg + 1e-10
            attention_norm = attention_safe / np.sum(attention_safe, axis=1, keepdims=True)
            entropy_per_query = -np.sum(attention_norm * np.log(attention_norm + 1e-10), axis=1)
            entropies.append(np.mean(entropy_per_query))

        ax_entropy.plot(blocks, entropies, marker='o', linewidth=3, markersize=10,
                        color='#E74C3C', markeredgecolor='black', markeredgewidth=1.5)
        ax_entropy.fill_between(range(len(blocks)), entropies, alpha=0.3, color='#E74C3C')
        ax_entropy.set_title('Diversidade da Atenção (Entropia) por Bloco',
                             fontweight='bold', fontsize=12)
        ax_entropy.grid(True, alpha=0.3)

        # Range
        ax_range = fig.add_subplot(gs[3, num_blocks:])
        attention_ranges = [att.max() - att.min() for att in processed_attentions]
        bars_range = ax_range.bar(blocks, attention_ranges, color=colors, alpha=0.8,
                                  edgecolor='black', linewidth=1.2)
        ax_range.set_title('Range de Atenção', fontweight='bold', fontsize=12)
        ax_range.grid(True, alpha=0.3, axis='y')

        # ============= ROW 5: ESTATÍSTICAS =============
        ax_stats = fig.add_subplot(gs[4, :])
        ax_stats.axis('off')

        stats_text = "ESTATÍSTICAS GLOBAIS\n" + "=" * 80 + "\n\n"
        all_attentions = np.concatenate([att.flatten() for att in processed_attentions])
        stats_text += f"Média Geral: {all_attentions.mean():.4f} | "
        stats_text += f"Desvio Padrão: {all_attentions.std():.4f} | "
        stats_text += f"Mediana: {np.median(all_attentions):.4f}\n"

        if flow_cls is not None:
            stats_text += f"\n🎯 CLS Flow - Top 5 patches: {np.argsort(flow_cls[2:])[-5:][::-1]}\n"
        if self.use_distillation and flow_dist is not None:
            stats_text += f"🎯 DIST Flow - Top 5 patches: {np.argsort(flow_dist[2:])[-5:][::-1]}\n"

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
        """Compila e treina o modelo com visualizações após o treinamento."""
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

        # Gerar visualizações após o treinamento
        if visualize_attention and validation_data is not None:
            val_data, val_labels = validation_data
            print("\n🎨 Gerando visualizações de Attention Flow e Rollout...")
            self.visualize_attention_flow(
                val_data,
                labels=val_labels,
                num_samples=128,
                output_dir='MAPS_AST_DISTILLATION'
            )
            print("✅ Visualizações salvas em MAPS_AST_DISTILLATION/")

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