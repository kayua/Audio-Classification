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
    from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization
    from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Activation

    from Engine.Models.Process.MobileNet_Process import MobileNetProcess

except ImportError as error:
    print(error)
    sys.exit(-1)


class MobileNetModel(MobileNetProcess):
    """
    Enhanced MobileNetModel with COMPLETE Explainable AI (XAI) capabilities

    FUNCIONALIDADES XAI IMPLEMENTADAS:
    ==================================
    1. ✅ Grad-CAM: Visualização padrão de mapas de ativação
    2. ✅ Grad-CAM++: Versão melhorada com ponderação mais precisa
    3. ✅ Score-CAM: Método sem gradientes (gradient-free)
    4. ✅ Visualizações modernas e interativas
    5. ✅ Geração automática para validação
    6. ✅ Análise comparativa de múltiplos métodos XAI
    """

    def __init__(self, arguments):
        """
        Initialize the MobileNetModel class with XAI capabilities.

        Args:
            @input_dimension (tuple): The shape of the input data (e.g., (128, 128, 3) for RGB images).
            @convolutional_padding (str): Padding strategy for convolution layers (e.g., 'same' or 'valid').
            @intermediary_activation (str): Activation function used in intermediary layers (e.g., 'relu').
            @last_layer_activation (str): Activation function for the output layer (e.g., 'softmax').
            @number_classes (int): The number of output classes for classification.
            @size_convolutional_filters (tuple): The size of the convolutional filters (e.g., (3, 3)).
            @alpha (float): Width multiplier for MobileNet (controls network width).
            @filters_per_block (list[int]): A list containing the number of filters for each block.
            @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer function used during model training (e.g., 'adam').
            @dropout_rate (float): The dropout rate for regularization.
        """
        MobileNetProcess.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.mobilenet_loss_function
        self.alpha = arguments.mobilenet_alpha
        self.filters_per_block = arguments.mobilenet_filters_per_block
        self.input_shape = arguments.mobilenet_input_dimension
        self.optimizer_function = arguments.mobilenet_optimizer_function
        self.dropout_rate = arguments.mobilenet_dropout_rate
        self.size_convolutional_filters = arguments.mobilenet_size_convolutional_filters
        self.number_classes = arguments.number_classes
        self.last_layer_activation = arguments.mobilenet_last_layer_activation
        self.convolutional_padding = arguments.mobilenet_convolutional_padding
        self.intermediary_activation = arguments.mobilenet_intermediary_activation
        self.model_name = "MobileNetModel"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def _depthwise_separable_conv(self, x, filters, block_idx, strides=(1, 1)):
        """
        Depthwise Separable Convolution block (core of MobileNet).

        Args:
            x: Input tensor
            filters: Number of output filters
            block_idx: Block index for naming
            strides: Strides for depthwise convolution

        Returns:
            Output tensor
        """
        # Depthwise Convolution
        x = DepthwiseConv2D(
            kernel_size=self.size_convolutional_filters,
            strides=strides,
            padding=self.convolutional_padding,
            name=f'depthwise_conv_block_{block_idx}'
        )(x)
        x = BatchNormalization(name=f'depthwise_bn_block_{block_idx}')(x)
        x = Activation(self.intermediary_activation, name=f'depthwise_act_block_{block_idx}')(x)

        # Pointwise Convolution (1x1 conv)
        x = Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            name=f'pointwise_conv_block_{block_idx}'
        )(x)
        x = BatchNormalization(name=f'pointwise_bn_block_{block_idx}')(x)
        x = Activation(self.intermediary_activation, name=f'pointwise_act_block_{block_idx}')(x)

        return x

    def build_model(self):
        """
        Build the MobileNet architecture using Keras Functional API with proper layer naming for XAI.

        This method defines the structure of the MobileNet network with named layers
        to enable XAI visualization on specific layers.
        """
        inputs = Input(shape=self.input_shape, name='input_layer')

        # Initial standard convolution
        x = Conv2D(
            int(32 * self.alpha),
            kernel_size=self.size_convolutional_filters,
            strides=(2, 2),
            padding=self.convolutional_padding,
            name='initial_conv'
        )(inputs)
        x = BatchNormalization(name='initial_bn')(x)
        x = Activation(self.intermediary_activation, name='initial_activation')(x)

        # Depthwise Separable Convolution blocks
        for block_idx, filters in enumerate(self.filters_per_block):
            # Apply width multiplier
            filters = int(filters * self.alpha)

            # Determine stride (reduce spatial dimensions every few blocks)
            stride = (2, 2) if block_idx in [1, 3, 5] else (1, 1)

            x = self._depthwise_separable_conv(
                x,
                filters,
                block_idx,
                strides=stride
            )

            # Add dropout after each block
            x = Dropout(
                self.dropout_rate,
                name=f'dropout_block_{block_idx}'
            )(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D(name='global_avg_pooling')(x)

        # Final dropout before classification
        x = Dropout(self.dropout_rate, name='final_dropout')(x)

        # Output layer
        outputs = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='output_layer'
        )(x)

        self.neural_network_model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        self.neural_network_model.summary()

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int, validation_data: tuple = None,
                          generate_gradcam: bool = True, num_gradcam_samples: int = 30,
                          gradcam_output_dir: str = './mapas_de_ativacao',
                          xai_method: str = 'gradcam++') -> tensorflow.keras.callbacks.History:
        """
        Compile and train the model with enhanced XAI visualization.

        Args:
            train_data: Training input data
            train_labels: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data tuple (X_val, y_val)
            generate_gradcam: Whether to generate activation maps after training
            num_gradcam_samples: Number of samples to visualize
            gradcam_output_dir: Output directory for visualizations
            xai_method: XAI method to use ('gradcam', 'gradcam++', or 'scorecam')

        Returns:
            Training history object
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

        if generate_gradcam and validation_data is not None:
            val_data, val_labels = validation_data

            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=128,
                output_dir='Maps_MobileNet',
                xai_method=xai_method
            )

        return training_history

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Build an auxiliary model for Visualization/Visualization++ computation.

        Args:
            target_layer_name: Name of target layer. If None, uses last pointwise conv layer
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating Visualization model")

        # Find last pointwise convolution layer if no target specified
        if target_layer_name is None:
            # Get last block's pointwise conv
            last_block_idx = len(self.filters_per_block) - 1
            target_layer_name = f'pointwise_conv_block_{last_block_idx}'

        target_layer = self.neural_network_model.get_layer(target_layer_name)

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

    @staticmethod
    def smooth_heatmap(heatmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
                            smooth: bool = True) -> np.ndarray:
        """
        Interpolate heatmap to target shape with optional smoothing.

        Args:
            heatmap: Input heatmap (2D array)
            target_shape: Target dimensions (height, width)
            smooth: Apply Gaussian smoothing after interpolation

        Returns:
            Interpolated heatmap
        """
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)

        # For 2D heatmap
        if len(heatmap.shape) == 2:
            zoom_factors = (target_shape[0] / heatmap.shape[0],
                            target_shape[1] / heatmap.shape[1])
            interpolated = zoom(heatmap, zoom_factors, order=3)
        else:
            raise ValueError(f"Heatmap shape inesperado: {heatmap.shape}")

        # Fine adjustment
        if interpolated.shape != target_shape:
            zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                   target_shape[1] / interpolated.shape[1])
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        # Smoothing
        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=1.5)

        return interpolated

    def plot_gradcam_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'gradcam++',
                            save_path: str = None, show_plot: bool = True) -> None:
        """
        Modern, visually appealing Visualization visualization with enhanced aesthetics.

        Args:
            input_sample: Input image
            heatmap: Activation heatmap
            class_idx: Class index used for computation
            predicted_class: Predicted class
            true_label: True label (optional)
            confidence: Prediction confidence
            xai_method: XAI method name
            save_path: Path to save figure
            show_plot: Whether to display the plot
        """
        # Handle input dimensions
        if len(input_sample.shape) == 4:
            input_sample = input_sample[0]
        if len(input_sample.shape) == 3 and input_sample.shape[-1] == 1:
            input_sample = np.squeeze(input_sample, axis=-1)

        # Enhanced interpolation with smoothing
        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_sample.shape[:2], smooth=True)

        # Create figure with modern style
        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        # Color schemes
        cmap_input = 'viridis'
        cmap_heatmap = 'jet'

        # 1. Original Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('Spectrogram', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Temporal Frames', fontsize=10)
        ax1.set_ylabel('Frequency Bins', fontsize=10)
        ax1.grid(False)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'Activation Map ({xai_method.upper()})',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Temporal Frames', fontsize=10)
        ax2.set_ylabel('Frequency Bins', fontsize=10)
        ax2.grid(False)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)

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
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Perfil Suavizado')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Perfil Original')

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

        Args:
            validation_data: Validation input data
            validation_labels: Validation labels
            num_samples: Number of samples to visualize
            output_dir: Output directory for saving visualizations
            target_layer_name: Target layer for XAI (if None, uses default)
            xai_method: XAI method ('gradcam', 'gradcam++', or 'scorecam')

        Returns:
            Dictionary with statistics about generated visualizations
        """
        import os

        if target_layer_name is None:
            last_block_idx = len(self.filters_per_block) - 1

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
                    prefix = 'correto'

                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorreto'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_amostra_{i:03d}_real_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                self.plot_gradcam_modern(sample,
                                         heatmap,
                                         predicted,
                                         predicted,
                                         true_label,
                                         confidence=confidence,
                                         xai_method=xai_method,
                                         save_path=save_path, show_plot=False)


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

        Args:
            input_sample: Input image to explain
            class_names: List of class names (optional)
            save_path: Path to save comprehensive analysis figure
            show_plot: Whether to display the plot

        Returns:
            Dictionary with explanation data and heatmaps
        """
        # Prepare sample
        if len(input_sample.shape) == 4:
            input_sample_plot = input_sample[0]
        else:
            input_sample_plot = input_sample.copy()

        if len(input_sample_plot.shape) == 3 and input_sample_plot.shape[-1] == 1:
            input_sample_plot = np.squeeze(input_sample_plot, axis=-1)

        # Get predictions
        if len(input_sample.shape) == 2:
            input_sample_batch = np.expand_dims(input_sample, axis=(0, -1))
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample_batch = np.expand_dims(input_sample, axis=0)
        else:
            input_sample_batch = input_sample

        predictions = self.neural_network_model.predict(input_sample_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Compute heatmaps
        heatmap_gradcam = self.compute_gradcam(input_sample, class_idx=predicted_class)
        heatmap_gradcampp = self.compute_gradcam_plusplus(input_sample, class_idx=predicted_class)

        # Interpolate
        heatmap_gc_interp = self.interpolate_heatmap(heatmap_gradcam, input_sample_plot.shape[:2], smooth=True)
        heatmap_pp_interp = self.interpolate_heatmap(heatmap_gradcampp, input_sample_plot.shape[:2], smooth=True)

        # Create figure
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heat = 'jet'

        # Row 1: Grad-CAM
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample_plot, cmap=cmap_input, aspect='auto')
        ax1.set_title('Original Spectrogram', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(heatmap_gc_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Grad-CAM', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(input_sample_plot, cmap='gray', aspect='auto')
        ax3.imshow(heatmap_gc_interp, cmap=cmap_heat, alpha=0.35, aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')

        # Row 2: Grad-CAM++
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(input_sample_plot, cmap=cmap_input, aspect='auto')
        ax4.set_title('Original Spectrogram', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(heatmap_pp_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax5.set_title('Grad-CAM++', fontsize=12, fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(input_sample_plot, cmap='gray', aspect='auto')
        ax6.imshow(heatmap_pp_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax6.set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')

        # Row 3: Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(predictions[0]))]

        colors = ['#2ecc71' if i == predicted_class else '#95a5a6' for i in range(len(predictions[0]))]
        bars = ax7.barh(class_names, predictions[0], color=colors, edgecolor='black', linewidth=1.5)
        ax7.set_xlabel('Probability', fontsize=11, fontweight='bold')
        ax7.set_title(f'Prediction: {class_names[predicted_class]} (Confidence: {confidence:.1%})',
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

        ax8.set_xlabel('Time Frame', fontsize=10)
        ax8.set_ylabel('Importance', fontsize=10)
        ax8.set_title('Temporal Comparison', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

        fig.suptitle('MobileNet Model XAI',
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
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def filters_per_block(self):
        return self._filters_per_block

    @filters_per_block.setter
    def filters_per_block(self, value):
        self._filters_per_block = value

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = value

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, value):
        self._optimizer_function = value

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value

    @property
    def size_convolutional_filters(self):
        return self._size_convolutional_filters

    @size_convolutional_filters.setter
    def size_convolutional_filters(self, value):
        self._size_convolutional_filters = value

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value):
        self._number_classes = value

    @property
    def last_layer_activation(self):
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value):
        self._last_layer_activation = value

    @property
    def convolutional_padding(self):
        return self._convolutional_padding

    @convolutional_padding.setter
    def convolutional_padding(self, value):
        self._convolutional_padding = value

    @property
    def intermediary_activation(self):
        return self._intermediary_activation

    @intermediary_activation.setter
    def intermediary_activation(self, value):
        self._intermediary_activation = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value