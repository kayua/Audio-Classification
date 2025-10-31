#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

from Engine.Models.Visualization.VisualizationMobileNetV2 import VisualizationMobileNetV2

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
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import ReLU
    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Activation

    from Engine.Models.Process.MobileNetV2_Process import MobileNetV2Process
    from Engine.GradientMap.MobileNetV2GradientMaps import MobileNetV2GradientMaps

except ImportError as error:
    print(error)
    sys.exit(-1)


class MobileNetV2Model(MobileNetV2Process, MobileNetV2GradientMaps, VisualizationMobileNetV2):

    def __init__(self, arguments):

        MobileNetV2Process.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.mobilenet_loss_function
        self.alpha = arguments.mobilenetv2_alpha
        self.expansion_factor = arguments.mobilenetv2_expansion_factor
        self.filters_per_block = arguments.mobilenetv2_filters_per_block
        self.inverted_residual_blocks = arguments.mobilenetv2_inverted_residual_blocks
        self.input_shape = arguments.mobilenetv2_input_dimension
        self.optimizer_function = arguments.mobilenetv2_optimizer_function
        self.dropout_rate = arguments.mobilenetv2_dropout_rate
        self.size_convolutional_filters = arguments.mobilenetv2_size_convolutional_filters
        self.number_classes = arguments.number_classes
        self.last_layer_activation = arguments.mobilenetv2_last_layer_activation
        self.convolutional_padding = arguments.mobilenetv2_convolutional_padding
        self.intermediary_activation = arguments.mobilenetv2_intermediary_activation
        self.model_name = "MobileNetV2Model"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

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

    def _inverted_residual_block(self, neural_network_flow, expansion, output_channels,
                                  stride, block_idx):
        """
        Inverted Residual Block (bottleneck with expansion) - core of MobileNetV2.

        Architecture:
        1. Expansion: 1x1 Conv (increase channels) -> BN -> ReLU6
        2. Depthwise: 3x3 DepthwiseConv -> BN -> ReLU6
        3. Projection: 1x1 Conv (reduce channels) -> BN (no activation - linear bottleneck)
        4. Skip connection (if stride=1 and input_channels == output_channels)

        Args:
            neural_network_flow: Input tensor
            expansion: Expansion factor for the block
            output_channels: Number of output channels
            stride: Stride for depthwise convolution
            block_idx: Block identifier for naming

        Returns:
            Output tensor
        """
        input_channels = neural_network_flow.shape[-1]
        output_channels = self._make_divisible(int(output_channels * self.alpha))

        # Determine if we should use skip connection
        use_skip_connection = (stride == 1 and input_channels == output_channels)

        x = neural_network_flow

        # Expansion phase (only if expansion != 1)
        if expansion != 1:
            expanded_channels = self._make_divisible(int(input_channels * expansion))
            x = Conv2D(expanded_channels,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=f'expansion_conv_block_{block_idx}')(x)
            x = BatchNormalization(name=f'expansion_bn_block_{block_idx}')(x)
            x = ReLU(max_value=6.0, name=f'expansion_relu6_block_{block_idx}')(x)

        # Depthwise convolution phase
        x = DepthwiseConv2D(kernel_size=self.size_convolutional_filters,
                           strides=stride,
                           padding='same',
                           use_bias=False,
                           name=f'depthwise_conv_block_{block_idx}')(x)
        x = BatchNormalization(name=f'depthwise_bn_block_{block_idx}')(x)
        x = ReLU(max_value=6.0, name=f'depthwise_relu6_block_{block_idx}')(x)

        # Projection phase (linear bottleneck - no activation)
        x = Conv2D(output_channels,
                  kernel_size=1,
                  padding='same',
                  use_bias=False,
                  name=f'projection_conv_block_{block_idx}')(x)
        x = BatchNormalization(name=f'projection_bn_block_{block_idx}')(x)

        # Skip connection
        if use_skip_connection:
            x = Add(name=f'skip_connection_block_{block_idx}')([neural_network_flow, x])

        return x

    def build_model(self):
        """
        Build the MobileNetV2 architecture using Keras Functional API with proper layer naming for XAI.

        This method defines the structure of the MobileNetV2 network with named layers
        to enable XAI visualization on specific layers.
        """
        inputs = Input(shape=self.input_shape, name='input_layer')

        # Initial convolution layer
        first_filters = self._make_divisible(32 * self.alpha)
        neural_network_flow = Conv2D(first_filters,
                   kernel_size=self.size_convolutional_filters,
                   strides=(2, 2),
                   padding='same',
                   use_bias=False,
                   name='initial_conv')(inputs)
        neural_network_flow = BatchNormalization(name='initial_bn')(neural_network_flow)
        neural_network_flow = ReLU(max_value=6.0, name='initial_relu6')(neural_network_flow)

        # Build inverted residual blocks
        block_id = 0
        for expansion, channels, num_blocks, stride in self.inverted_residual_blocks:
            # First block in the sequence (may have stride > 1)
            neural_network_flow = self._inverted_residual_block(neural_network_flow,
                                                               expansion=expansion,
                                                               output_channels=channels,
                                                               stride=stride,
                                                               block_idx=block_id)
            block_id += 1

            # Remaining blocks (stride = 1)
            for _ in range(1, num_blocks):
                neural_network_flow = self._inverted_residual_block(neural_network_flow,
                                                                   expansion=expansion,
                                                                   output_channels=channels,
                                                                   stride=1,
                                                                   block_idx=block_id)
                block_id += 1

            # Add dropout after each group of blocks
            neural_network_flow = Dropout(self.dropout_rate,
                                         name=f'dropout_block_{block_id}')(neural_network_flow)

        # Final convolution layer (expand to higher dimensional space)
        if self.alpha > 1.0:
            last_filters = self._make_divisible(1280 * self.alpha)
        else:
            last_filters = 1280

        neural_network_flow = Conv2D(last_filters,
                  kernel_size=1,
                  padding='same',
                  use_bias=False,
                  name='final_conv')(neural_network_flow)
        neural_network_flow = BatchNormalization(name='final_bn')(neural_network_flow)
        neural_network_flow = ReLU(max_value=6.0, name='final_relu6')(neural_network_flow)

        # Global Average Pooling
        neural_network_flow = GlobalAveragePooling2D(name='global_avg_pooling')(neural_network_flow)

        # Final dropout before classification
        neural_network_flow = Dropout(self.dropout_rate, name='final_dropout')(neural_network_flow)

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
        #         output_dir='Maps_MobileNetV2',
        #         xai_method=xai_method
        #     )

        return training_history



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
    def expansion_factor(self):
        return self._expansion_factor

    @expansion_factor.setter
    def expansion_factor(self, value):
        self._expansion_factor = value

    @property
    def filters_per_block(self):
        return self._filters_per_block

    @filters_per_block.setter
    def filters_per_block(self, value):
        self._filters_per_block = value

    @property
    def inverted_residual_blocks(self):
        return self._inverted_residual_blocks

    @inverted_residual_blocks.setter
    def inverted_residual_blocks(self, value):
        self._inverted_residual_blocks = value

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