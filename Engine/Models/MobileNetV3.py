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
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import ReLU
    from tensorflow.keras.layers import Multiply
    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Reshape

    from Engine.Models.Process.MobileNetV3_Process import MobileNetV3Process
    from Engine.GradientMap.MobileNetV3GradientMaps import MobileNetV3GradientMaps

except ImportError as error:
    print(error)
    sys.exit(-1)


class MobileNetV3Model(MobileNetV3Process, MobileNetV3GradientMaps):

    def __init__(self, arguments):

        MobileNetV3Process.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.variant = arguments.mobilenetv3_variant
        self.loss_function = arguments.mobilenetv3_loss_function
        self.alpha = arguments.mobilenetv3_alpha
        self.expansion_factor = arguments.mobilenetv3_expansion_factor
        self.filters_per_block = arguments.mobilenetv3_filters_per_block
        self.bneck_blocks_large = arguments.mobilenetv3_bneck_blocks_large
        self.bneck_blocks_small = arguments.mobilenetv3_bneck_blocks_small
        self.input_shape = arguments.mobilenetv3_input_dimension
        self.optimizer_function = arguments.mobilenetv3_optimizer_function
        self.dropout_rate = arguments.mobilenetv3_dropout_rate
        self.size_convolutional_filters = arguments.mobilenetv3_size_convolutional_filters
        self.number_classes = arguments.number_classes
        self.last_layer_activation = arguments.mobilenetv3_last_layer_activation
        self.convolutional_padding = arguments.mobilenetv3_convolutional_padding
        self.intermediary_activation = arguments.mobilenetv3_intermediary_activation
        self.model_name = f"MobileNetV3_{self.variant.capitalize()}"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    @staticmethod
    def hard_sigmoid(x):
        """Hard Sigmoid activation function."""
        return tensorflow.nn.relu6(x + 3.0) / 6.0

    @staticmethod
    def hard_swish(x):
        """Hard Swish activation function (more efficient than swish)."""
        return x * tensorflow.nn.relu6(x + 3.0) / 6.0

    @staticmethod
    def _make_divisible(value, divisor=8):
        """
        Ensure that all layers have a channel number divisible by divisor.
        This is important for hardware efficiency.

        Args:
            value: Original value
            divisor: Divisor (default: 8)

        Returns:
            Adjusted value divisible by divisor
        """
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _squeeze_excitation_block(self, inputs, filters, se_ratio=0.25, block_idx=0):
        """
        Squeeze-and-Excitation block for channel attention.

        Args:
            inputs: Input tensor
            filters: Number of input filters
            se_ratio: Reduction ratio for SE block
            block_idx: Block identifier for naming

        Returns:
            Output tensor after SE operation
        """
        # Squeeze: Global average pooling
        se = GlobalAveragePooling2D(name=f'se_squeeze_block_{block_idx}')(inputs)
        se = Reshape((1, 1, filters), name=f'se_reshape_block_{block_idx}')(se)

        # Excitation: FC layers
        se_filters = max(1, int(filters * se_ratio))
        se = Conv2D(se_filters, 1, padding='same', activation='relu',
                    name=f'se_reduce_block_{block_idx}')(se)
        se = Conv2D(filters, 1, padding='same',
                    name=f'se_expand_block_{block_idx}')(se)

        # Hard sigmoid activation
        se = Activation(self.hard_sigmoid, name=f'se_hsigmoid_block_{block_idx}')(se)

        # Scale
        return Multiply(name=f'se_scale_block_{block_idx}')([inputs, se])

    def _bottleneck_block(self, neural_network_flow, kernel, exp_size, out_channels,
                          use_se, activation, stride, block_idx):
        """
        MobileNetV3 Bottleneck block with SE and flexible activation.

        Architecture:
        1. Expansion: 1x1 Conv (increase channels) -> BN -> Activation
        2. Depthwise: KxK DepthwiseConv -> BN -> Activation
        3. SE Block (optional)
        4. Projection: 1x1 Conv (reduce channels) -> BN (no activation - linear bottleneck)
        5. Skip connection (if stride=1 and input_channels == output_channels)

        Args:
            neural_network_flow: Input tensor
            kernel: Kernel size for depthwise convolution
            exp_size: Expansion size (intermediate channels)
            out_channels: Number of output channels
            use_se: Whether to use Squeeze-and-Excitation
            activation: Activation function ('RE' for ReLU, 'HS' for hard-swish)
            stride: Stride for depthwise convolution
            block_idx: Block identifier for naming

        Returns:
            Output tensor
        """
        input_channels = neural_network_flow.shape[-1]
        exp_size = self._make_divisible(int(exp_size * self.alpha))
        out_channels = self._make_divisible(int(out_channels * self.alpha))

        # Determine if we should use skip connection
        use_skip_connection = (stride == 1 and input_channels == out_channels)

        # Choose activation function
        if activation == 'RE':
            act_fn = ReLU(name=f'relu_block_{block_idx}')
        elif activation == 'HS':
            act_fn = Activation(self.hard_swish, name=f'hswish_block_{block_idx}')
        else:
            act_fn = Activation(activation, name=f'act_block_{block_idx}')

        x = neural_network_flow

        # Expansion phase (only if exp_size != input_channels)
        if exp_size != input_channels:
            x = Conv2D(exp_size,
                       kernel_size=1,
                       padding='same',
                       use_bias=False,
                       name=f'bneck_expansion_conv_block_{block_idx}')(x)
            x = BatchNormalization(name=f'bneck_expansion_bn_block_{block_idx}')(x)
            if activation == 'RE':
                x = ReLU(name=f'bneck_expansion_relu_block_{block_idx}')(x)
            elif activation == 'HS':
                x = Activation(self.hard_swish, name=f'bneck_expansion_hswish_block_{block_idx}')(x)

        # Depthwise convolution phase
        x = DepthwiseConv2D(kernel_size=kernel,
                            strides=stride,
                            padding='same',
                            use_bias=False,
                            name=f'bneck_depthwise_conv_block_{block_idx}')(x)
        x = BatchNormalization(name=f'bneck_depthwise_bn_block_{block_idx}')(x)
        if activation == 'RE':
            x = ReLU(name=f'bneck_depthwise_relu_block_{block_idx}')(x)
        elif activation == 'HS':
            x = Activation(self.hard_swish, name=f'bneck_depthwise_hswish_block_{block_idx}')(x)

        # Squeeze-and-Excitation
        if use_se:
            x = self._squeeze_excitation_block(x, exp_size, block_idx=block_idx)

        # Projection phase (linear bottleneck - no activation)
        x = Conv2D(out_channels,
                   kernel_size=1,
                   padding='same',
                   use_bias=False,
                   name=f'bneck_projection_conv_block_{block_idx}')(x)
        x = BatchNormalization(name=f'bneck_projection_bn_block_{block_idx}')(x)

        # Skip connection
        if use_skip_connection:
            x = Add(name=f'bneck_skip_connection_block_{block_idx}')([neural_network_flow, x])

        return x

    def build_model(self):
        """
        Build the MobileNetV3 architecture using Keras Functional API with proper layer naming for XAI.

        This method defines the structure of the MobileNetV3 network (Large or Small variant)
        with named layers to enable XAI visualization on specific layers.
        """
        inputs = Input(shape=self.input_shape, name='input_layer')

        # Select configuration based on variant
        if self.variant == 'large':
            bneck_config = self.bneck_blocks_large
            last_conv_filters = 960
            last_point_filters = 1280
        else:  # small
            bneck_config = self.bneck_blocks_small
            last_conv_filters = 576
            last_point_filters = 1024

        # Initial convolution layer
        first_filters = self._make_divisible(16 * self.alpha)
        neural_network_flow = Conv2D(first_filters,
                                     kernel_size=3,
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False,
                                     name='initial_conv')(inputs)
        neural_network_flow = BatchNormalization(name='initial_bn')(neural_network_flow)
        neural_network_flow = Activation(self.hard_swish, name='initial_hswish')(neural_network_flow)

        # Build bottleneck blocks
        for block_id, (kernel, exp, out, se, nl, stride) in enumerate(bneck_config):
            neural_network_flow = self._bottleneck_block(
                neural_network_flow,
                kernel=kernel,
                exp_size=exp,
                out_channels=out,
                use_se=se,
                activation=nl,
                stride=stride,
                block_idx=block_id
            )

            # Add dropout after certain blocks
            if block_id % 3 == 0 and block_id > 0:
                neural_network_flow = Dropout(self.dropout_rate,
                                              name=f'dropout_block_{block_id}')(neural_network_flow)

        # Final layers (before pooling)
        last_conv_filters = self._make_divisible(last_conv_filters * self.alpha)
        neural_network_flow = Conv2D(last_conv_filters,
                                     kernel_size=1,
                                     padding='same',
                                     use_bias=False,
                                     name='final_conv')(neural_network_flow)
        neural_network_flow = BatchNormalization(name='final_bn')(neural_network_flow)
        neural_network_flow = Activation(self.hard_swish, name='final_hswish')(neural_network_flow)

        # Global Average Pooling
        neural_network_flow = GlobalAveragePooling2D(name='global_avg_pooling')(neural_network_flow)

        # Final projection
        neural_network_flow = Reshape((1, 1, last_conv_filters), name='reshape_pre_final')(neural_network_flow)

        last_point_filters = self._make_divisible(last_point_filters * self.alpha)
        neural_network_flow = Conv2D(last_point_filters,
                                     kernel_size=1,
                                     padding='same',
                                     use_bias=False,
                                     name='pre_output_conv')(neural_network_flow)
        neural_network_flow = Activation(self.hard_swish, name='pre_output_hswish')(neural_network_flow)

        # Final dropout
        neural_network_flow = Dropout(self.dropout_rate, name='final_dropout')(neural_network_flow)

        # Flatten for output
        neural_network_flow = Flatten(name='flatten')(neural_network_flow)

        # Output layer
        outputs = Dense(self.number_classes,
                        activation=self.last_layer_activation, name='output_layer')(neural_network_flow)

        self.neural_network_model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        self.neural_network_model.summary()

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int, validation_data: tuple = None,
                          generate_gradcam: bool = True, num_gradcam_samples: int = 30,
                          gradcam_output_dir: str = './mapas_de_ativacao',
                          xai_method: str = 'scorecam') -> tensorflow.keras.callbacks.History:
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
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

        # if generate_gradcam and validation_data is not None:
        #     val_data, val_labels = validation_data
        #
        #     stats = self.generate_validation_visualizations(
        #         validation_data=val_data,
        #         validation_labels=val_labels,
        #         num_samples=8,
        #         output_dir='Maps_MobileNetV3',
        #         xai_method=xai_method
        #     )

        return training_history

    @staticmethod
    def smooth_heatmap(heatmap: numpy.ndarray, sigma: float = 2.0) -> numpy.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: numpy.ndarray, target_shape: tuple,
                            smooth: bool = True) -> numpy.ndarray:
        """
        Interpolate heatmap to target shape with optional smoothing.

        Args:
            heatmap: Input heatmap (2D array)
            target_shape: Target dimensions (height, width)
            smooth: Apply Gaussian smoothing after interpolation

        Returns:
            Interpolated heatmap
        """
        if not isinstance(heatmap, numpy.ndarray):
            heatmap = numpy.array(heatmap)

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

    def plot_gradcam_modern(self, input_sample: numpy.ndarray, heatmap: numpy.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'scorecam',
                            save_path: str = None, show_plot: bool = True) -> None:
        """
        Modern, visually appealing GradCAM visualization with enhanced aesthetics.

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
            input_sample = numpy.squeeze(input_sample, axis=-1)

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
        temporal_importance = numpy.mean(interpolated_heatmap, axis=0)
        time_steps = numpy.arange(len(temporal_importance))

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

    def generate_validation_visualizations(self, validation_data: numpy.ndarray,
                                           validation_labels: numpy.ndarray,
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

        os.makedirs(output_dir, exist_ok=True)

        predictions = self.neural_network_model.predict(validation_data, verbose=0)
        predicted_classes = numpy.argmax(predictions, axis=1)
        confidences = numpy.max(predictions, axis=1)

        if len(validation_labels.shape) > 1:
            true_labels = numpy.argmax(validation_labels, axis=1)
        else:
            true_labels = validation_labels

        correct_indices = numpy.where(predicted_classes == true_labels)[0]
        incorrect_indices = numpy.where(predicted_classes != true_labels)[0]

        num_correct = min(num_samples // 2, len(correct_indices))
        num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

        selected_correct = numpy.random.choice(correct_indices, num_correct, replace=False) if len(
            correct_indices) > 0 else []
        selected_incorrect = numpy.random.choice(incorrect_indices, num_incorrect, replace=False) if len(
            incorrect_indices) > 0 else []

        selected_indices = numpy.concatenate([selected_correct, selected_incorrect])

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

    def explain_prediction_comprehensive(self, input_sample: numpy.ndarray,
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
            input_sample_plot = numpy.squeeze(input_sample_plot, axis=-1)

        # Get predictions
        if len(input_sample.shape) == 2:
            input_sample_batch = numpy.expand_dims(input_sample, axis=(0, -1))
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample_batch = numpy.expand_dims(input_sample, axis=0)
        else:
            input_sample_batch = input_sample

        predictions = self.neural_network_model.predict(input_sample_batch, verbose=0)
        predicted_class = numpy.argmax(predictions[0])
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
        temporal_gc = numpy.mean(heatmap_gc_interp, axis=0)
        temporal_gcpp = numpy.mean(heatmap_pp_interp, axis=0)

        temporal_gc_smooth = gaussian_filter(temporal_gc, sigma=2)
        temporal_gcpp_smooth = gaussian_filter(temporal_gcpp, sigma=2)

        time_steps = numpy.arange(len(temporal_gc))
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

        fig.suptitle('MobileNetV2 Model XAI',
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

    @property
    def get_neural_network_model(self):
        """Get the compiled neural network model."""
        return self.neural_network_model

    @property
    def get_gradcam_model(self):
        """Get the GradCAM model."""
        return self.gradcam_model

    @property
    def get_variant(self):
        """Get the MobileNetV3 variant (large or small)."""
        return self.variant

    @property
    def get_loss_function(self):
        """Get the loss function."""
        return self.loss_function

    @property
    def get_alpha(self):
        """Get the width multiplier alpha."""
        return self.alpha

    @property
    def get_expansion_factor(self):
        """Get the expansion factor."""
        return self.expansion_factor

    @property
    def get_filters_per_block(self):
        """Get the number of filters per block."""
        return self.filters_per_block

    @property
    def get_bneck_blocks_large(self):
        """Get the bottleneck block configuration for large variant."""
        return self.bneck_blocks_large

    @property
    def get_bneck_blocks_small(self):
        """Get the bottleneck block configuration for small variant."""
        return self.bneck_blocks_small

    @property
    def get_input_shape(self):
        """Get the input shape."""
        return self.input_shape

    @property
    def get_optimizer_function(self):
        """Get the optimizer function."""
        return self.optimizer_function

    @property
    def get_dropout_rate(self):
        """Get the dropout rate."""
        return self.dropout_rate

    @property
    def get_size_convolutional_filters(self):
        """Get the size of convolutional filters."""
        return self.size_convolutional_filters

    @property
    def get_number_classes(self):
        """Get the number of output classes."""
        return self.number_classes

    @property
    def get_last_layer_activation(self):
        """Get the activation function for the last layer."""
        return self.last_layer_activation

    @property
    def get_convolutional_padding(self):
        """Get the convolutional padding type."""
        return self.convolutional_padding

    @property
    def get_intermediary_activation(self):
        """Get the intermediary activation function."""
        return self.intermediary_activation

    @property
    def get_model_name(self):
        """Get the model name."""
        return self.model_name

    # ============================================================================
    # SETTERS
    # ============================================================================

    @get_neural_network_model.setter
    def set_neural_network_model(self, model):
        """
        Set the neural network model.

        Args:
            model: Keras Model instance
        """
        if model is not None and not isinstance(model, Model):
            raise TypeError("Model must be a Keras Model instance")
        self.neural_network_model = model

    @get_gradcam_model.setter
    def set_gradcam_model(self, model):
        """
        Set the GradCAM model.

        Args:
            model: Keras Model instance for GradCAM
        """
        if model is not None and not isinstance(model, Model):
            raise TypeError("GradCAM model must be a Keras Model instance")
        self.gradcam_model = model

    @get_variant.setter
    def set_variant(self, variant):
        """
        Set the MobileNetV3 variant.

        Args:
            variant: 'large' or 'small'
        """
        if variant not in ['large', 'small']:
            raise ValueError("Variant must be 'large' or 'small'")
        self.variant = variant
        self.model_name = f"MobileNetV3_{self.variant.capitalize()}"

    @get_loss_function.setter
    def set_loss_function(self, loss):
        """
        Set the loss function.

        Args:
            loss: Loss function (string or callable)
        """
        self.loss_function = loss

    @get_alpha.setter
    def set_alpha(self, alpha):
        """
        Set the width multiplier alpha.

        Args:
            alpha: Float value for width multiplier (typically 0.5, 0.75, 1.0, 1.25)
        """
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number")
        self.alpha = float(alpha)

    @get_expansion_factor.setter
    def set_expansion_factor(self, factor):
        """
        Set the expansion factor.

        Args:
            factor: Expansion factor for bottleneck blocks
        """
        if not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError("Expansion factor must be a positive number")
        self.expansion_factor = factor

    @get_filters_per_block.setter
    def set_filters_per_block(self, filters):
        """
        Set the number of filters per block.

        Args:
            filters: Number of filters
        """
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError("Filters per block must be a positive integer")
        self.filters_per_block = filters

    @get_bneck_blocks_large.setter
    def set_bneck_blocks_large(self, blocks):
        """
        Set the bottleneck blocks configuration for large variant.

        Args:
            blocks: List of bottleneck block configurations
        """
        if not isinstance(blocks, list):
            raise TypeError("Bottleneck blocks must be a list")
        self.bneck_blocks_large = blocks

    @get_bneck_blocks_small.setter
    def set_bneck_blocks_small(self, blocks):
        """
        Set the bottleneck blocks configuration for small variant.

        Args:
            blocks: List of bottleneck block configurations
        """
        if not isinstance(blocks, list):
            raise TypeError("Bottleneck blocks must be a list")
        self.bneck_blocks_small = blocks

    @get_input_shape.setter
    def set_input_shape(self, shape):
        """
        Set the input shape.

        Args:
            shape: Tuple representing input dimensions (height, width, channels)
        """
        if not isinstance(shape, tuple) or len(shape) != 3:
            raise ValueError("Input shape must be a tuple of 3 dimensions (H, W, C)")
        self.input_shape = shape

    @get_optimizer_function.setter
    def set_optimizer_function(self, optimizer):
        """
        Set the optimizer function.

        Args:
            optimizer: Optimizer instance or string
        """
        self.optimizer_function = optimizer

    @get_dropout_rate.setter
    def set_dropout_rate(self, rate):
        """
        Set the dropout rate.

        Args:
            rate: Dropout rate between 0 and 1
        """
        if not isinstance(rate, (int, float)) or not (0 <= rate < 1):
            raise ValueError("Dropout rate must be between 0 and 1")
        self.dropout_rate = float(rate)

    @get_size_convolutional_filters.setter
    def set_size_convolutional_filters(self, size):
        """
        Set the size of convolutional filters.

        Args:
            size: Filter size (e.g., 3 for 3x3 filters)
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Filter size must be a positive integer")
        self.size_convolutional_filters = size

    @get_number_classes.setter
    def set_number_classes(self, num_classes):
        """
        Set the number of output classes.

        Args:
            num_classes: Number of classes for classification
        """
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer")
        self.number_classes = num_classes

    @get_last_layer_activation.setter
    def set_last_layer_activation(self, activation):
        """
        Set the activation function for the last layer.

        Args:
            activation: Activation function (string or callable)
        """
        self.last_layer_activation = activation

    @get_convolutional_padding.setter
    def set_convolutional_padding(self, padding):
        """
        Set the convolutional padding type.

        Args:
            padding: Padding type ('same' or 'valid')
        """
        if padding not in ['same', 'valid']:
            raise ValueError("Padding must be 'same' or 'valid'")
        self.convolutional_padding = padding

    @get_intermediary_activation.setter
    def set_intermediary_activation(self, activation):
        """
        Set the intermediary activation function.

        Args:
            activation: Activation function (string or callable)
        """
        self.intermediary_activation = activation

    @get_model_name.setter
    def set_model_name(self, name):
        """
        Set the model name.

        Args:
            name: Model name string
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Model name must be a non-empty string")
        self.model_name = name