#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{1}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/15'
__credits__ = ['unknown']

# CORREÇÕES APLICADAS:
# 1. ✅ Positional embeddings agora cobrem CLS token + patches
# 2. ✅ XAI usando gradientes reais (não simulação)
# 3. ✅ Patch importance com normalização corrigida
# 4. ✅ Attention visualization mais informativa
# 5. ✅ Código otimizado para convergência

try:
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import models
    from tensorflow.keras.layers import Add, Input, Layer, Dense
    from tensorflow.keras.layers import Conv1D, Dropout, Flatten
    from tensorflow.keras.layers import Embedding, Concatenate
    from tensorflow.keras.layers import TimeDistributed

    from Engine.Layers.CLSTokenLayer import CLSTokenLayer
    from Engine.Models.Process.AST_Process import ProcessAST
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
    from tensorflow.keras.layers import MultiHeadAttention

except ImportError as error:
    print(error)
    sys.exit(-1)


class AudioSpectrogramTransformer(ProcessAST):
    """
    Audio Spectrogram Transformer WITHOUT Layer Normalization
    WITH CORRECTED XAI and Model Architecture

    BUGS CORRIGIDOS:
    ================
    1. ✅ Positional embeddings agora tem dimensão correta (num_patches + 1 para incluir CLS)
    2. ✅ XAI usando gradientes REAIS ao invés de simulação
    3. ✅ Patch importance com normalização por percentil
    4. ✅ Visualizações mais informativas e contrastadas
    """

    def __init__(self, arguments):
        super().__init__(arguments)
        self.neural_network_model = None
        self.attention_model = None
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
        self.model_name = "AST_NoNorm"

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def transformer_encoder(self, inputs: tensorflow.Tensor, block_idx: int = 0) -> tensorflow.Tensor:
        """
        Transformer encoder WITHOUT layer normalization.
        """
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            key_dim=self.head_size,
            num_heads=self.number_heads,
            dropout=self.dropout,
            name=f'multi_head_attention_block_{block_idx}'
        )(inputs, inputs)

        attention_output = Dropout(self.dropout, name=f'dropout_1_block_{block_idx}')(attention_output)
        neural_model_flow = Add(name=f'add_1_block_{block_idx}')([attention_output, inputs])

        # Feedforward network
        ffn_output = Dense(
            neural_model_flow.shape[2],
            activation=self.intermediary_activation,
            name=f'ffn_dense_block_{block_idx}'
        )(neural_model_flow)

        ffn_output = Dropout(self.dropout, name=f'dropout_2_block_{block_idx}')(ffn_output)
        output = Add(name=f'add_2_block_{block_idx}')([ffn_output, neural_model_flow])

        return output

    def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
        """
        Build AST model WITHOUT layer normalization - WITH CORRECTED ARCHITECTURE
        """
        # Input layer
        inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]), name='input_layer')
        input_flatten = TimeDistributed(Flatten(), name='time_distributed_flatten')(inputs)
        linear_projection = TimeDistributed(Dense(self.projection_dimension),
                                            name='linear_projection')(input_flatten)

        # CLS token
        cls_tokens_layer = CLSTokenLayer(self.projection_dimension, name='cls_token')(linear_projection)

        # Concatenate CLS token with patches
        neural_model_flow = Concatenate(axis=1, name='concat_cls')([cls_tokens_layer, linear_projection])

        # 🔥 CORREÇÃO CRÍTICA: Positional embeddings para (num_patches + 1) incluindo CLS token
        positional_embeddings_layer = PositionalEmbeddingsLayer(
            number_patches + 1,  # ✅ CORRIGIDO: +1 para incluir CLS token
            self.projection_dimension,
            name='positional_embeddings'
        )(neural_model_flow)  # ✅ CORRIGIDO: usar neural_model_flow que já inclui CLS

        neural_model_flow = Add(name='add_positional_embeddings')([neural_model_flow, positional_embeddings_layer])

        # Transformer blocks
        for block_idx in range(self.number_blocks):
            neural_model_flow = self.transformer_encoder(neural_model_flow, block_idx)

        # Global average pooling (sem layer norm)
        neural_model_flow = GlobalAveragePooling1D(name='global_avg_pooling')(neural_model_flow)

        # Output
        neural_model_flow = Dropout(self.dropout, name='final_dropout')(neural_model_flow)
        outputs = Dense(self.number_classes, activation=self.last_activation_layer,
                        name='output_layer')(neural_model_flow)

        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
        self.neural_network_model.summary()
        return self.neural_network_model

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int, validation_data: tuple = None,
                          generate_attention_maps: bool = True, num_samples: int = 30,
                          output_dir: str = './attention_visualizations') -> tensorflow.keras.callbacks.History:
        """
        Compile and train the model.
        """
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

        if validation_data is not None:
            print(f"Acurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")

        if generate_attention_maps and validation_data is not None:
            print("\n" + "=" * 80)
            print("GERANDO VISUALIZAÇÕES XAI - VERSÃO CORRIGIDA")
            print("=" * 80)

            val_data, val_labels = validation_data
            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=num_samples,
                output_dir=output_dir
            )

            print(f"\n✓ Visualizações salvas em: {output_dir}")
            print("=" * 80 + "\n")

        return training_history

    def compute_gradient_based_attention(self, input_sample: np.ndarray,
                                         class_idx: int = None) -> np.ndarray:
        """
        🔥 NOVO: Compute attention using REAL gradients (not simulation)

        This computes how much each patch influences the prediction using gradient flow.
        """
        if len(input_sample.shape) == 3:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.neural_network_model(input_tensor, training=False)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_score = predictions[:, class_idx]

        # Compute gradients
        gradients = tape.gradient(class_score, input_tensor)
        gradients_np = gradients.numpy()[0]  # Remove batch

        # Compute attention per patch using gradient magnitude
        patch_attention = np.mean(np.abs(gradients_np), axis=(1, 2))

        # Normalize using percentile for better contrast
        p95 = np.percentile(patch_attention, 95)
        patch_attention = np.clip(patch_attention / (p95 + 1e-8), 0, 1)

        return patch_attention

    def compute_patch_importance(self, input_sample: np.ndarray,
                                 class_idx: int = None) -> np.ndarray:
        """
        🔥 CORRIGIDO: Compute patch importance with better normalization
        """
        if len(input_sample.shape) == 3:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.neural_network_model(input_tensor, training=False)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_score = predictions[:, class_idx]

        gradients = tape.gradient(class_score, input_tensor)
        gradients_np = gradients.numpy()[0]

        # Compute importance using gradient * input (Grad-CAM style)
        patch_importance = np.mean(np.abs(gradients_np * input_sample[0]), axis=(1, 2))

        # Normalize with percentile for better visual contrast
        p90 = np.percentile(patch_importance, 90)
        if p90 > 0:
            patch_importance = np.clip(patch_importance / p90, 0, 1)

        return patch_importance

    @staticmethod
    def interpolate_attention_to_spectrogram(attention_weights: np.ndarray,
                                             target_shape: tuple,
                                             patch_grid: tuple,
                                             smooth: bool = True) -> np.ndarray:
        """
        Interpolate attention weights to match spectrogram dimensions.
        """
        attention_2d = attention_weights.reshape(patch_grid)

        zoom_factors = (target_shape[0] / attention_2d.shape[0],
                        target_shape[1] / attention_2d.shape[1])

        attention_map = zoom(attention_2d, zoom_factors, order=3)

        if smooth:
            attention_map = gaussian_filter(attention_map, sigma=2.0)

        return attention_map

    def plot_attention_visualization_modern(self, input_sample: np.ndarray,
                                            attention_weights: np.ndarray,
                                            patch_importance: np.ndarray,
                                            predicted_class: int,
                                            true_label: int = None,
                                            confidence: float = None,
                                            save_path: str = None,
                                            show_plot: bool = True) -> None:
        """
        🔥 CORRIGIDO: Visualização com melhor contraste e informação
        """
        num_patches = input_sample.shape[0]
        patch_h, patch_w = self.patch_size

        def find_grid_dimensions(n):
            sqrt_n = int(np.sqrt(n))
            for rows in range(sqrt_n, 0, -1):
                if n % rows == 0:
                    cols = n // rows
                    return rows, cols
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            return rows, cols

        grid_rows, grid_cols = find_grid_dimensions(num_patches)

        # Reconstruct spectrogram
        spectrogram = np.zeros((grid_rows * patch_h, grid_cols * patch_w))

        for i in range(num_patches):
            row = i // grid_cols
            col = i % grid_cols
            row_start = row * patch_h
            row_end = (row + 1) * patch_h
            col_start = col * patch_w
            col_end = (col + 1) * patch_w

            if row_end <= spectrogram.shape[0] and col_end <= spectrogram.shape[1]:
                spectrogram[row_start:row_end, col_start:col_end] = input_sample[i]

        # Interpolate attention maps
        attention_map = self.interpolate_attention_to_spectrogram(
            attention_weights,
            spectrogram.shape,
            (grid_rows, grid_cols),
            smooth=True
        )

        importance_map = self.interpolate_attention_to_spectrogram(
            patch_importance,
            spectrogram.shape,
            (grid_rows, grid_cols),
            smooth=True
        )

        # Create figure
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        cmap_input = 'viridis'
        cmap_attention = 'hot'

        # Row 1: Gradient-based Attention
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(spectrogram, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('📊 Espectrograma Original', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Tempo', fontsize=10)
        ax1.set_ylabel('Frequência', fontsize=10)
        ax1.grid(False)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(attention_map, cmap=cmap_attention, aspect='auto',
                         interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title('🎯 Gradient-based Attention', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Tempo', fontsize=10)
        ax2.set_ylabel('Frequência', fontsize=10)
        ax2.grid(False)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(spectrogram, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(attention_map, cmap=cmap_attention, alpha=0.6,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title('🔥 Attention Overlay', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Tempo', fontsize=10)
        ax3.set_ylabel('Frequência', fontsize=10)
        ax3.grid(False)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Row 2: Patch Importance
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(spectrogram, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax4.set_title('📊 Espectrograma Original', fontsize=13, fontweight='bold', pad=15)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(importance_map, cmap='RdYlGn', aspect='auto',
                         interpolation='bilinear', vmin=0, vmax=1)
        ax5.set_title('💡 Patch Importance (Grad×Input)', fontsize=13, fontweight='bold', pad=15)
        ax5.set_xlabel('Tempo', fontsize=10)
        ax5.set_ylabel('Frequência', fontsize=10)
        ax5.grid(False)
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(spectrogram, cmap='gray', aspect='auto', interpolation='bilinear')
        im6 = ax6.imshow(importance_map, cmap='RdYlGn', alpha=0.6,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax6.set_title('🎨 Importance Overlay', fontsize=13, fontweight='bold', pad=15)
        ax6.set_xlabel('Tempo', fontsize=10)
        ax6.set_ylabel('Frequência', fontsize=10)
        ax6.grid(False)
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        # Row 3: Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        patch_indices = np.arange(len(patch_importance))
        colors_importance = plt.cm.RdYlGn(patch_importance)

        ax7.bar(patch_indices, patch_importance, color=colors_importance,
                edgecolor='black', linewidth=0.5)
        ax7.set_xlabel('Índice do Patch', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Importância', fontsize=11, fontweight='bold')
        ax7.set_title('📊 Importância por Patch (Grad×Input)',
                      fontsize=13, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3, linestyle='--')
        ax7.set_ylim([0, 1.1])

        top_k = 5
        top_indices = np.argsort(patch_importance)[-top_k:]
        for idx in top_indices:
            ax7.axvline(idx, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

        ax8 = fig.add_subplot(gs[2, 2])
        colors_attention = plt.cm.hot(attention_weights)

        ax8.bar(patch_indices, attention_weights, color=colors_attention,
                edgecolor='black', linewidth=0.5)
        ax8.set_xlabel('Índice do Patch', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Atenção', fontsize=11, fontweight='bold')
        ax8.set_title('🎯 Gradient-based Attention',
                      fontsize=13, fontweight='bold')
        ax8.grid(axis='y', alpha=0.3, linestyle='--')
        ax8.set_ylim([0, 1.1])

        # Title
        pred_status = '✅' if predicted_class == true_label else '❌'
        conf_str = f' | Confiança: {confidence:.1%}' if confidence is not None else ''

        if true_label is not None:
            suptitle = f'{pred_status} Predito: Classe {predicted_class} | Verdadeiro: Classe {true_label}{conf_str}'
        else:
            suptitle = f'Predito: Classe {predicted_class}{conf_str}'

        fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Figura salva: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_validation_visualizations(self, validation_data: np.ndarray,
                                           validation_labels: np.ndarray,
                                           num_samples: int = 10,
                                           output_dir: str = './attention_visualizations') -> dict:
        """
        Generate XAI visualizations for validation samples.
        """
        import os

        print(f"\n🔍 Dados de validação: {validation_data.shape}")
        print(f"🎯 Labels de validação: {validation_labels.shape}")
        print(f"📐 Patch size: {self.patch_size}")
        print(f"🧠 Método: Gradient-based Attention + Grad×Input Importance\n")

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

        print(f"🎨 Gerando visualizações para {len(selected_indices)} amostras...")
        print(f"📁 Diretório de saída: {output_dir}\n")

        for i, idx in enumerate(selected_indices):
            try:
                sample = validation_data[idx]
                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

                # Compute gradient-based attention
                attention_weights = self.compute_gradient_based_attention(sample, class_idx=predicted)

                # Compute patch importance
                patch_importance = self.compute_patch_importance(sample, class_idx=predicted)

                is_correct = predicted == true_label
                if is_correct:
                    stats['correct_predictions'] += 1
                    prefix = 'correto'
                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorreto'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_amostra_{i:03d}_real_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                self.plot_attention_visualization_modern(
                    sample, attention_weights, patch_importance,
                    predicted, true_label, confidence=confidence,
                    save_path=save_path, show_plot=False
                )

                status = '✅' if is_correct else '❌'
                print(f"{status} Amostra {i + 1}/{len(selected_indices)}: "
                      f"Real={true_label}, Pred={predicted}, Conf={confidence:.1%}")

            except Exception as e:
                print(f"❌ Erro processando amostra {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'=' * 80}")
        print(f"✨ GERAÇÃO DE VISUALIZAÇÕES CONCLUÍDA!")
        print(f"{'=' * 80}")
        print(f"📊 Total de amostras: {stats['total_samples']}")
        print(f"✅ Predições corretas: {stats['correct_predictions']}")
        print(f"❌ Predições incorretas: {stats['incorrect_predictions']}")
        accuracy = stats['correct_predictions'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        print(f"🎯 Acurácia: {accuracy:.2f}%")
        print(f"💾 Arquivos salvos em: {output_dir}")
        print(f"{'=' * 80}\n")

        return stats

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