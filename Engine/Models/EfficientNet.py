#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

from Engine.GradientMap.EfficientNetGradientMaps import EfficientNetGradientMaps

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
    import numpy
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import GlobalAveragePooling2D

    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications import EfficientNetB1
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications import EfficientNetB5
    from tensorflow.keras.applications import EfficientNetB6
    from tensorflow.keras.applications import EfficientNetB7

    from Engine.Models.Process.EfficientNet_Process import ProcessEfficientNet

except ImportError as error:
    print(error)
    sys.exit(-1)


class EfficientNet(ProcessEfficientNet, EfficientNetGradientMaps):
    """
    Enhanced EfficientNet with FIXED Explainable AI (XAI) capabilities

    ARQUITETURA ADAPTADA PARA EFFICIENTNET:
    =======================================
    1. ✅ CNN 2D com compound scaling (B0-B7)
    2. ✅ Grad-CAM++ otimizado para arquitetura convolucional
    3. ✅ Interpolação para mapas de ativação 2D
    4. ✅ Visualizações modernas e detalhadas
    5. ✅ Suporte completo para Score-CAM e Grad-CAM
    """

    def __init__(self, arguments):
        ProcessEfficientNet.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.efficientnet_loss_function
        self.optimizer_function = arguments.efficientnet_optimizer_function
        self.number_filters_spectrogram = arguments.efficientnet_number_filters_spectrogram
        self.input_dimension = arguments.efficientnet_input_dimension
        self.efficientnet_version = arguments.efficientnet_version  # 'B0', 'B1', ..., 'B7'
        self.number_classes = arguments.number_classes
        self.dropout_rate = arguments.efficientnet_dropout_rate
        self.last_layer_activation = arguments.efficientnet_last_layer_activation
        self.model_name = f"EfficientNet{self.efficientnet_version}"

        # Transfer learning options
        self.use_pretrained = getattr(arguments, 'efficientnet_use_pretrained', False)
        self.freeze_base = getattr(arguments, 'efficientnet_freeze_base', False)
        self.fine_tune_at = getattr(arguments, 'efficientnet_fine_tune_at', None)

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """Build the EfficientNet model architecture using Keras Applications."""
        inputs = Input(shape=self.input_dimension, name='input_layer')

        # Select EfficientNet version
        efficientnet_models = {
            'B0': EfficientNetB0,
            'B1': EfficientNetB1,
            'B2': EfficientNetB2,
            'B3': EfficientNetB3,
            'B4': EfficientNetB4,
            'B5': EfficientNetB5,
            'B6': EfficientNetB6,
            'B7': EfficientNetB7
        }

        if self.efficientnet_version not in efficientnet_models:
            raise ValueError(f"Invalid EfficientNet version: {self.efficientnet_version}. "
                             f"Choose from: {list(efficientnet_models.keys())}")

        EfficientNetModel = efficientnet_models[self.efficientnet_version]

        # Determine if we can use ImageNet weights
        # ImageNet weights require specific minimum sizes and 3 channels
        min_sizes = {'B0': 32, 'B1': 32, 'B2': 32, 'B3': 40, 'B4': 48, 'B5': 56, 'B6': 64, 'B7': 72}
        min_size = min_sizes.get(self.efficientnet_version, 32)

        can_use_imagenet = (
                self.input_dimension[0] >= min_size and
                self.input_dimension[1] >= min_size and
                self.input_dimension[2] == 3
        )

        # Decide on weights
        use_pretrained = getattr(self, 'use_pretrained', False)
        if use_pretrained and can_use_imagenet:
            weights = 'imagenet'
            print(f"✅ Usando pesos pré-treinados do ImageNet para EfficientNet-{self.efficientnet_version}")
        else:
            weights = None
            if use_pretrained and not can_use_imagenet:
                print(f"⚠️  Dimensões de entrada {self.input_dimension} incompatíveis com ImageNet.")
                print(f"    Mínimo necessário: ({min_size}, {min_size}, 3)")
                print(f"    Treinando EfficientNet-{self.efficientnet_version} do zero.")
            else:
                print(f"🔨 Construindo EfficientNet-{self.efficientnet_version} do zero (sem pesos pré-treinados)")

        # Create base model
        try:
            base_model = EfficientNetModel(
                include_top=False,
                weights=weights,
                input_tensor=inputs,
                pooling=None
            )
        except Exception as e:
            # Fallback: criar sem pesos pré-treinados
            print(f"⚠️  Erro ao carregar com weights='{weights}': {str(e)}")
            print(f"    Tentando novamente sem pesos pré-treinados...")
            base_model = EfficientNetModel(
                include_top=False,
                weights=None,
                input_tensor=inputs,
                pooling=None
            )

        # Freeze base model if specified
        if self.freeze_base:
            base_model.trainable = False
            print(f"🔒 Base do modelo congelada (não treináve)")
        elif self.fine_tune_at is not None:
            # Fine-tune from a specific layer
            base_model.trainable = True
            for layer in base_model.layers[:self.fine_tune_at]:
                layer.trainable = False
            print(f"🔓 Fine-tuning a partir da camada {self.fine_tune_at}")
        else:
            print(f"🔓 Todas as camadas treináveis")

        # Add custom classification head
        neural_network_flow = base_model.output
        neural_network_flow = GlobalAveragePooling2D(name='global_avg_pooling')(neural_network_flow)

        if self.dropout_rate > 0:
            neural_network_flow = Dropout(self.dropout_rate, name='dropout')(neural_network_flow)

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
        Compile and train the EfficientNet model with enhanced XAI visualization.

        Args:
            xai_method (str): 'gradcam', 'gradcam++', or 'scorecam'
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(train_data, train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)

        if validation_data is not None:
            print(f"Acurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")

        if generate_gradcam and validation_data is not None:
            val_data, val_labels = validation_data

            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=128,
                output_dir='Maps_EfficientNet',
                xai_method=xai_method
            )

        return training_history

    def compile_model(self) -> None:
        """Compiles the EfficientNet model."""
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

    @staticmethod
    def smooth_heatmap(heatmap: numpy.ndarray, sigma: float = 2.0) -> numpy.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: numpy.ndarray, target_shape: tuple,
                            smooth: bool = True) -> numpy.ndarray:
        """
        INTERPOLAÇÃO para EfficientNet: Mapeia heatmaps 2D para spectrogramas.

        Args:
            heatmap: Input heatmap (2D spatial)
            target_shape: Target dimensions (altura, largura)
            smooth: Apply Gaussian smoothing after interpolation
        """
        # Converter para array numpy se necessário
        if not isinstance(heatmap, numpy.ndarray):
            heatmap = numpy.array(heatmap)

        if len(heatmap.shape) == 2:
            # Heatmap 2D - fazer zoom direto
            zoom_factors = (target_shape[0] / heatmap.shape[0], target_shape[1] / heatmap.shape[1])
            interpolated = zoom(heatmap, zoom_factors, order=3)

        elif len(heatmap.shape) == 1:
            # Fallback para 1D: expandir para 2D
            temporal_interp = zoom(heatmap, (target_shape[1] / heatmap.shape[0],), order=3)
            freq_profile = numpy.linspace(1.0, 0.6, target_shape[0])
            interpolated = freq_profile[:, numpy.newaxis] * temporal_interp[numpy.newaxis, :]

        else:
            raise ValueError(f"Heatmap shape inesperado: {heatmap.shape}")

        # Garantir dimensões corretas
        if interpolated.shape != target_shape:
            zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                   target_shape[1] / interpolated.shape[1])
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=2.0)

        return interpolated

    def plot_gradcam_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'gradcam++',
                            save_path: str = None, show_plot: bool = True) -> None:
        """
        Modern, visually appealing GradCAM visualization with enhanced aesthetics.
        """
        # Preparar amostra - EfficientNet usa 3 canais
        if len(input_sample.shape) == 4:
            input_sample = input_sample[0]

        # Usar apenas o primeiro canal para visualização
        if len(input_sample.shape) == 3:
            input_sample_2d = input_sample[:, :, 0]
        else:
            input_sample_2d = input_sample

        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_sample_2d.shape, smooth=True)

        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heatmap = 'jet'  # Melhor contraste

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample_2d, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title(' Spectrogram', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Temporal Frames', fontsize=10)
        ax1.set_ylabel('Frequency Bins', fontsize=10)
        ax1.grid(False)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)

        ax2.set_title(f' Activation Map ({xai_method.upper()})', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Temporal Frames', fontsize=10)
        ax2.set_ylabel('Frequency Bins', fontsize=10)
        ax2.grid(False)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_sample_2d - input_sample_2d.min()) / (
                    input_sample_2d.max() - input_sample_2d.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.4, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)

        ax3.set_title('Overlap', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Temporal Frames', fontsize=10)
        ax3.set_ylabel('Frequency Bins', fontsize=10)
        ax3.grid(False)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=9)

        # 4. Temporal Importance Profile
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_importance = np.mean(interpolated_heatmap, axis=0)
        time_steps = np.arange(len(temporal_importance))

        temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

        ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Smoothed Profile')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Original')

        ax4.set_xlabel('Temporal Frame', fontsize=10)
        ax4.set_ylabel('Average Importance', fontsize=10)
        ax4.set_title('Temporal Importance Profile', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_xlim([0, len(temporal_importance)])
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

                self.plot_gradcam_modern(sample, heatmap,
                                         predicted, predicted,
                                         true_label, confidence=confidence,
                                         xai_method=xai_method,
                                         save_path=save_path,
                                         show_plot=False)

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

        return stats

    def explain_prediction_comprehensive(self, input_sample: np.ndarray,
                                         class_names: list = None,
                                         save_path: str = None,
                                         show_plot: bool = True) -> dict:
        """
        Generate comprehensive explanation with multiple XAI methods comparison.
        """
        # Prepare sample
        if len(input_sample.shape) == 4:
            input_sample_2d = input_sample[0, :, :, 0]  # Usar primeiro canal
            input_sample_batch = input_sample
        elif len(input_sample.shape) == 3:
            input_sample_2d = input_sample[:, :, 0]
            input_sample_batch = np.expand_dims(input_sample, axis=0)
        else:
            input_sample_2d = input_sample.copy()
            input_sample_batch = np.expand_dims(np.expand_dims(input_sample, axis=-1), axis=0)

        # Get predictions
        predictions = self.neural_network_model.predict(input_sample_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Compute heatmaps
        heatmap_gradcam = self.compute_gradcam(input_sample_batch[0], class_idx=predicted_class)
        heatmap_gradcampp = self.compute_gradcam_plusplus(input_sample_batch[0], class_idx=predicted_class)

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
        ax1.set_title('Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(heatmap_gc_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Grad-CAM', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(input_sample_2d, cmap='gray', aspect='auto')
        ax3.imshow(heatmap_gc_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')

        # Row 2: Grad-CAM++
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(input_sample_2d, cmap=cmap_input, aspect='auto')
        ax4.set_title('Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(heatmap_pp_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax5.set_title('Grad-CAM++ (Melhorado)', fontsize=12, fontweight='bold')
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
        ax7.set_title(f'Predição: {class_names[predicted_class]} (Confiança: {confidence:.1%})',
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
        ax8.set_title('Comparação Temporal', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

        fig.suptitle(f'Análise Explicativa Abrangente - {self.model_name} XAI',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

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
    def efficientnet_version(self):
        return self._efficientnet_version

    @efficientnet_version.setter
    def efficientnet_version(self, value):
        self._efficientnet_version = value

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

    @property
    def freeze_base(self):
        return self._freeze_base

    @freeze_base.setter
    def freeze_base(self, value):
        self._freeze_base = value

    @property
    def fine_tune_at(self):
        return self._fine_tune_at

    @fine_tune_at.setter
    def fine_tune_at(self, value):
        self._fine_tune_at = value