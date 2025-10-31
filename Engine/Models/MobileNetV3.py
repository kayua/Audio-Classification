#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

from Engine.Models.Visualization.VisualizationMobileNetV3 import VisualizationMobileNetV3

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
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Reshape

    from Engine.Models.Process.MobileNetV3_Process import MobileNetV3Process
    from Engine.GradientMap.MobileNetV3GradientMaps import MobileNetV3GradientMaps

except ImportError as error:
    print(error)
    sys.exit(-1)


class MobileNetV3Model(MobileNetV3Process, MobileNetV3GradientMaps, VisualizationMobileNetV3):

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

    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True,
                          use_early_stopping: bool = True,
                          early_stopping_monitor: str = 'val_loss',
                          early_stopping_patience: int = 10,
                          early_stopping_restore_best: bool = True,
                          early_stopping_min_delta: float = 0.0001) -> tensorflow.keras.callbacks.History:
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

        callbacks = []
        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=early_stopping_restore_best,
                min_delta=early_stopping_min_delta,
                verbose=1,
                mode='auto'
            )
            callbacks.append(early_stopping)


        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if callbacks else None
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