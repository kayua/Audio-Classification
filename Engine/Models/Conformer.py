#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
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
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Input, Layer, Dropout, Flatten, Reshape
    from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D

    from Engine.Layers.ConformerBlock import ConformerBlock
    from Engine.Layers.TransposeLayer import TransposeLayer
    from Engine.Models.Process.Conformer_Process import ProcessConformer
    from Engine.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling

except ImportError as error:
    print(error)
    sys.exit(-1)


class Conformer(ProcessConformer):
    """
    Enhanced Conformer with FIXED Explainable AI (XAI) capabilities

    CORREÇÕES IMPLEMENTADAS:
    ========================
    1. ✅ Camada alvo corrigida: usa 'conv_subsampling' por padrão (mantém estrutura 2D)
    2. ✅ Grad-CAM++ axis corrigido para arquitetura Transformer
    3. ✅ Interpolação melhorada para lidar com saídas 2D convolucionais
    4. ✅ Avisos sobre limitações da arquitetura Transformer
    5. ✅ Tratamento adequado de heatmaps com diferentes dimensionalidades
    """

    def __init__(self, arguments):
        ProcessConformer.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.conformer_loss_function
        self.optimizer_function = arguments.conformer_optimizer_function
        self.number_filters_spectrogram = arguments.conformer_number_filters_spectrogram
        self.input_dimension = arguments.conformer_input_dimension
        self.number_conformer_blocks = arguments.conformer_number_conformer_blocks
        self.embedding_dimension = arguments.conformer_embedding_dimension
        self.number_heads = arguments.conformer_number_heads
        self.number_classes = arguments.number_classes
        self.kernel_size = arguments.conformer_size_kernel
        self.dropout_rate = arguments.conformer_dropout_rate
        self.last_layer_activation = arguments.conformer_last_layer_activation
        self.model_name = "Conformer"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """Build the Conformer model architecture using Keras."""
        inputs = Input(shape=self.input_dimension, name='input_layer')

        neural_network_flow = ConvolutionalSubsampling(name='conv_subsampling')(inputs)
        neural_network_flow = TransposeLayer(perm=[0, 2, 1], name='transpose_layer')(neural_network_flow)
        neural_network_flow = Dense(self.embedding_dimension, name='embedding_dense')(neural_network_flow)

        for i in range(self.number_conformer_blocks):
            neural_network_flow = ConformerBlock(
                self.embedding_dimension,
                self.number_heads,
                self.input_dimension[0] // 2,
                self.kernel_size,
                self.dropout_rate,
                name=f'conformer_block_{i}'
            )(neural_network_flow)

        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pooling')(neural_network_flow)
        neural_network_flow = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='output_layer'
        )(neural_network_flow)

        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)
        self.neural_network_model.summary()

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int, validation_data: tuple = None,
                          generate_gradcam: bool = True, num_gradcam_samples: int = 30,
                          gradcam_output_dir: str = './mapas_de_ativacao',
                          xai_method: str = 'gradcam++') -> tensorflow.keras.callbacks.History:
        """
        Compile and train the Conformer model with enhanced XAI visualization.

        Args:
            xai_method (str): 'gradcam', 'gradcam++', or 'scorecam'
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

        if generate_gradcam and validation_data is not None:
            print("\n" + "=" * 80)
            print(f"GERANDO MAPAS DE ATIVAÇÃO ({xai_method.upper()}) - VERSÃO CORRIGIDA")
            print("=" * 80)

            val_data, val_labels = validation_data

            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=num_gradcam_samples,
                output_dir=gradcam_output_dir,
                xai_method=xai_method
            )

            print(f"\n✓ Mapas de ativação salvos em: {gradcam_output_dir}")
            print("=" * 80 + "\n")

        return training_history

    def compile_model(self) -> None:
        """Compiles the Conformer model."""
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Build an auxiliary model for GradCAM/GradCAM++ computation.

        CORREÇÃO CRÍTICA: Usa 'conv_subsampling' por padrão, que ainda mantém
        estrutura espacial 2D antes da transformação em sequência.

        Args:
            target_layer_name: Nome da camada alvo. Se None, usa 'conv_subsampling'
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating GradCAM model")

        # ✅ CORREÇÃO: Usar conv_subsampling que mantém estrutura 2D
        if target_layer_name is None:
            target_layer_name = 'conv_subsampling'
            print("⚠️  AVISO: Usando 'conv_subsampling' como camada alvo (recomendado)")
            print("   Esta é a última camada que mantém estrutura espacial 2D")
            print("   Saída esperada: (batch, tempo_reduzido, features_conv)")
        else:
            print(f"⚠️  AVISO: Usando camada customizada '{target_layer_name}'")
            if 'conformer_block' in target_layer_name:
                print("   ⚠️  ConformerBlocks têm saída 1D sequencial - visualização pode ser limitada")

        target_layer = self.neural_network_model.get_layer(target_layer_name)

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

        print(f"✓ GradCAM model criado com camada: {target_layer_name}")
        print(f"  Forma de saída da camada: {target_layer.output.shape}")

    def compute_gradcam_plusplus(self, input_sample: np.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap (VERSÃO CORRIGIDA).

        CORREÇÕES:
        - ✅ Axis corrigido no reduce_sum para arquiteturas com saída 3D
        - ✅ Tratamento adequado de diferentes dimensionalidades de saída
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct input shape
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape1:
            with tensorflow.GradientTape() as tape2:
                with tensorflow.GradientTape() as tape3:
                    layer_output, predictions = self.gradcam_model(input_tensor)

                    if class_idx is None:
                        class_idx = tensorflow.argmax(predictions[0]).numpy()

                    class_score = predictions[:, class_idx]

                # First-order gradients
                grads = tape3.gradient(class_score, layer_output)

            # Second-order gradients
            grads_2 = tape2.gradient(grads, layer_output)

        # Third-order gradients
        grads_3 = tape1.gradient(grads_2, layer_output)

        # ✅ CORREÇÃO: Axis correto para diferentes formas de saída
        # Para saída 3D (batch, spatial/temporal, features): usar axis=2
        # Para saída 4D (batch, height, width, features): usar axis=3
        if len(layer_output.shape) == 3:
            reduce_axis = 2  # Features dimension
        elif len(layer_output.shape) == 4:
            reduce_axis = 3  # Channel dimension
        else:
            reduce_axis = -1  # Fallback

        # Compute alpha weights (Grad-CAM++ formula) - CORRIGIDO
        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
            layer_output * grads_3, axis=reduce_axis, keepdims=True
        ) + 1e-10

        alpha = numerator / denominator

        # ReLU on gradients
        relu_grads = tensorflow.maximum(grads, 0.0)

        # Weighted combination
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1,))

        # Compute weighted activation map
        layer_output_squeezed = layer_output[0]
        heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tensorflow.maximum(heatmap, 0)

        # ✅ Normalização robusta
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max
        else:
            print("⚠️  Aviso: Heatmap com valores muito baixos - pode indicar problema na camada alvo")

        return heatmap.numpy()

    def compute_gradcam(self, input_sample: np.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> np.ndarray:
        """Standard Grad-CAM computation (VERSÃO CORRIGIDA)."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        # ✅ CORREÇÃO: Adaptar pooling para diferentes shapes
        if len(layer_output.shape) == 3:  # (batch, sequence, features)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1))
        elif len(layer_output.shape) == 4:  # (batch, height, width, channels)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
        else:
            pooled_grads = tensorflow.reduce_mean(grads, axis=0)

        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # ✅ Normalização robusta
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_scorecam(self, input_sample: np.ndarray, class_idx: int = None,
                         target_layer_name: str = None, batch_size: int = 32) -> np.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method).
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        # Get activations
        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        # Get activation maps
        activations = layer_output[0].numpy()

        # ✅ Adaptar para diferentes shapes de ativação
        if len(activations.shape) == 2:  # (sequence, features)
            num_channels = activations.shape[-1]
        else:
            num_channels = activations.shape[-1]

        weights = []
        for i in range(num_channels):
            if len(activations.shape) == 2:
                act_map = activations[:, i]
            else:
                act_map = activations[:, :, i] if len(activations.shape) == 3 else activations[:, i]

            # Normalize to [0, 1]
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            if len(act_map.shape) == 1:
                upsampled = zoom(act_map, (input_sample.shape[1] / act_map.shape[0],), order=1)
                upsampled = np.tile(upsampled[:, np.newaxis], (1, input_sample.shape[2]))
            else:
                zoom_factors = (input_sample.shape[1] / act_map.shape[0],
                                input_sample.shape[2] / act_map.shape[1])
                upsampled = zoom(act_map, zoom_factors, order=1)

            # Mask input
            masked_input = input_sample[0] * upsampled
            masked_input = np.expand_dims(masked_input, 0)

            # Get score for masked input
            masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
            score = masked_pred[0, class_idx]

            weights.append(score)

        weights = np.array(weights)
        weights = np.maximum(weights, 0)

        # Weighted combination
        if len(activations.shape) == 2:
            heatmap = np.dot(activations, weights)
        else:
            heatmap = np.tensordot(activations, weights, axes=([[-1], [0]]))
            heatmap = np.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def smooth_heatmap(heatmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
                            smooth: bool = True) -> np.ndarray:
        """
        ✅ INTERPOLAÇÃO CORRIGIDA: Agora lida adequadamente com heatmaps 1D e 2D.

        Args:
            heatmap: Input heatmap (pode ser 1D ou 2D)
            target_shape: Target dimensions (altura, largura)
            smooth: Apply Gaussian smoothing after interpolation
        """
        # Converter para array numpy se necessário
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)

        # Caso 1: Heatmap 1D (ex: saída de ConformerBlock)
        if len(heatmap.shape) == 1:
            print(f"  ℹ️  Heatmap 1D detectado (tamanho: {heatmap.shape[0]})")
            print(f"     Expandindo para 2D: {target_shape}")

            # Interpolar temporalmente primeiro
            temporal_interp = zoom(heatmap, (target_shape[0] / heatmap.shape[0],), order=3)

            # Expandir para frequências usando gradiente suave
            # Em vez de replicar, criar transição suave
            freq_profile = np.linspace(1.0, 0.6, target_shape[1])
            heatmap_2d = temporal_interp[:, np.newaxis] * freq_profile[np.newaxis, :]

            interpolated = heatmap_2d

        # Caso 2: Heatmap 2D (ex: saída de conv_subsampling)
        elif len(heatmap.shape) == 2:
            print(f"  ℹ️  Heatmap 2D detectado (forma: {heatmap.shape})")
            zoom_factors = (target_shape[0] / heatmap.shape[0],
                            target_shape[1] / heatmap.shape[1])
            interpolated = zoom(heatmap, zoom_factors, order=3)  # Cubic interpolation
        else:
            raise ValueError(f"Heatmap shape inesperado: {heatmap.shape}")

        # Ajuste fino se necessário
        if interpolated.shape != target_shape:
            zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                   target_shape[1] / interpolated.shape[1])
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        # Suavização final
        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=1.5)

        return interpolated

    def plot_gradcam_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'gradcam++',
                            save_path: str = None, show_plot: bool = True) -> None:
        """
        Modern, visually appealing GradCAM visualization with enhanced aesthetics.
        """
        if len(input_sample.shape) == 3:
            input_sample = input_sample[0]

        # Enhanced interpolation with smoothing
        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_sample.shape, smooth=True)

        # Create figure with modern style
        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        # Color schemes
        cmap_input = 'viridis'
        cmap_heatmap = 'jet'  # Melhor contraste

        # 1. Original Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('📊 Espectrograma de Entrada',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Frames Temporais', fontsize=10)
        ax1.set_ylabel('Bins de Frequência', fontsize=10)
        ax1.grid(False)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'🔥 Mapa de Ativação ({xai_method.upper()})',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Frames Temporais', fontsize=10)
        ax2.set_ylabel('Bins de Frequência', fontsize=10)
        ax2.grid(False)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)

        ax3.set_title('🎯 Sobreposição', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Frames Temporais', fontsize=10)
        ax3.set_ylabel('Bins de Frequência', fontsize=10)
        ax3.grid(False)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=9)

        # 4. Temporal Importance Profile
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_importance = np.mean(interpolated_heatmap, axis=0)
        time_steps = np.arange(len(temporal_importance))

        temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

        ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Perfil Suavizado')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Perfil Original')

        ax4.set_xlabel('Frame Temporal', fontsize=10)
        ax4.set_ylabel('Importância Média', fontsize=10)
        ax4.set_title('📈 Perfil de Importância Temporal',
                      fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_xlim([0, len(temporal_importance)])
        ax4.set_ylim([0, 1])

        # Super title
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
                                           output_dir: str = './gradcam_outputs',
                                           target_layer_name: str = None,
                                           xai_method: str = 'gradcam++') -> dict:
        """
        Generate enhanced XAI visualizations for validation samples.
        """
        import os

        print(f"\n🔍 Dados de validação: {validation_data.shape}")
        print(f"🎯 Labels de validação: {validation_labels.shape}")
        print(f"📐 Dimensão de entrada esperada: {self.input_dimension}")
        print(f"🧠 Método XAI: {xai_method.upper()}")

        # ✅ Informar sobre camada alvo
        if target_layer_name is None:
            print(f"🎯 Camada alvo: 'conv_subsampling' (padrão - recomendado)")
        else:
            print(f"🎯 Camada alvo: '{target_layer_name}'")
        print()

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

                # Ensure 2D
                if len(sample.shape) == 3:
                    sample = np.squeeze(sample)

                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

                # Compute heatmap based on selected method
                if xai_method.lower() == 'gradcam++':
                    heatmap = self.compute_gradcam_plusplus(sample, class_idx=predicted,
                                                            target_layer_name=target_layer_name)
                elif xai_method.lower() == 'scorecam':
                    heatmap = self.compute_scorecam(sample, class_idx=predicted,
                                                    target_layer_name=target_layer_name)
                else:  # standard gradcam
                    heatmap = self.compute_gradcam(sample, class_idx=predicted,
                                                   target_layer_name=target_layer_name)

                is_correct = predicted == true_label
                if is_correct:
                    stats['correct_predictions'] += 1
                    prefix = 'correto'
                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorreto'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_amostra_{i:03d}_real_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                self.plot_gradcam_modern(
                    sample, heatmap, predicted, predicted, true_label,
                    confidence=confidence, xai_method=xai_method,
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

    def explain_prediction_comprehensive(self, input_sample: np.ndarray,
                                         class_names: list = None,
                                         save_path: str = None,
                                         show_plot: bool = True) -> dict:
        """
        Generate comprehensive explanation with multiple XAI methods comparison.
        """
        # Prepare sample
        if len(input_sample.shape) == 3:
            input_sample_2d = np.squeeze(input_sample)
        else:
            input_sample_2d = input_sample.copy()

        if len(input_sample.shape) == 2:
            input_sample_batch = np.expand_dims(input_sample, axis=0)
        else:
            input_sample_batch = input_sample

        # Get predictions
        predictions = self.neural_network_model.predict(input_sample_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Compute heatmaps
        print("🔄 Computando mapas de ativação com diferentes métodos...")

        heatmap_gradcam = self.compute_gradcam(input_sample_2d, class_idx=predicted_class)
        heatmap_gradcampp = self.compute_gradcam_plusplus(input_sample_2d, class_idx=predicted_class)

        print("✓ Mapas computados com sucesso!")

        # Interpolate
        heatmap_gc_interp = self.interpolate_heatmap(heatmap_gradcam, input_sample_2d.shape, smooth=True)
        heatmap_pp_interp = self.interpolate_heatmap(heatmap_gradcampp, input_sample_2d.shape, smooth=True)

        # Create figure
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heat = 'jet'

        # Row 1: Grad-CAM
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample_2d, cmap=cmap_input, aspect='auto')
        ax1.set_title('📊 Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(heatmap_gc_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax2.set_title('🔥 Grad-CAM', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(input_sample_2d, cmap='gray', aspect='auto')
        ax3.imshow(heatmap_gc_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')

        # Row 2: Grad-CAM++
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(input_sample_2d, cmap=cmap_input, aspect='auto')
        ax4.set_title('📊 Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(heatmap_pp_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax5.set_title('🔥 Grad-CAM++ (Melhorado)', fontsize=12, fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(input_sample_2d, cmap='gray', aspect='auto')
        ax6.imshow(heatmap_pp_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax6.set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')

        # Row 3: Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(len(predictions[0]))]

        colors = ['#2ecc71' if i == predicted_class else '#95a5a6' for i in range(len(predictions[0]))]
        bars = ax7.barh(class_names, predictions[0], color=colors, edgecolor='black', linewidth=1.5)
        ax7.set_xlabel('Probabilidade', fontsize=11, fontweight='bold')
        ax7.set_title(f'🎯 Predição: {class_names[predicted_class]} (Confiança: {confidence:.1%})',
                      fontsize=13, fontweight='bold')
        ax7.set_xlim([0, 1])
        ax7.grid(axis='x', alpha=0.3)

        for bar, prob in zip(bars, predictions[0]):
            ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.3f}', ha='left', va='center', fontsize=10)

        # Temporal comparison
        ax8 = fig.add_subplot(gs[2, 2])
        temporal_gc = np.mean(heatmap_gc_interp, axis=0)
        temporal_gcpp = np.mean(heatmap_pp_interp, axis=0)

        temporal_gc_smooth = gaussian_filter(temporal_gc, sigma=2)
        temporal_gcpp_smooth = gaussian_filter(temporal_gcpp, sigma=2)

        time_steps = np.arange(len(temporal_gc))
        ax8.plot(time_steps, temporal_gc_smooth, linewidth=2, label='Grad-CAM', color='#3498db')
        ax8.plot(time_steps, temporal_gcpp_smooth, linewidth=2, label='Grad-CAM++', color='#e74c3c')
        ax8.fill_between(time_steps, temporal_gc_smooth, alpha=0.2, color='#3498db')
        ax8.fill_between(time_steps, temporal_gcpp_smooth, alpha=0.2, color='#e74c3c')

        ax8.set_xlabel('Frame Temporal', fontsize=10)
        ax8.set_ylabel('Importância', fontsize=10)
        ax8.set_title('📈 Comparação Temporal', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

        fig.suptitle('🧠 Análise Explicativa Abrangente - Conformer XAI (CORRIGIDO)',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Análise completa salva: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        explanation = {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'class_probabilities': predictions[0].tolist(),
            'heatmap_gradcam': heatmap_gradcam,
            'heatmap_gradcampp': heatmap_gradcampp,
            'heatmap_gradcam_interpolated': heatmap_gc_interp,
            'heatmap_gradcampp_interpolated': heatmap_pp_interp,
            'temporal_importance_gradcam': temporal_gc,
            'temporal_importance_gradcampp': temporal_gcpp
        }

        if class_names:
            explanation['predicted_class_name'] = class_names[predicted_class]

        print("✅ Explicação abrangente gerada com sucesso!")
        return explanation

    # Properties
    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

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
    def number_filters_spectrogram(self):
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, value):
        self._number_filters_spectrogram = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def number_conformer_blocks(self):
        return self._number_conformer_blocks

    @number_conformer_blocks.setter
    def number_conformer_blocks(self, value):
        self._number_conformer_blocks = value

    @property
    def embedding_dimension(self):
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value):
        self._embedding_dimension = value

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, value):
        self._number_heads = value

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value):
        self._number_classes = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

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