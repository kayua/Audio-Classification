#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

from Engine.Layers.LayerScale import LayerScale

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

    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Layer
    from tensorflow.keras import initializers
    from Engine.Models.Process.ConvNetX_Process import ConvNetXProcess
    from Engine.GradientMap.ConvNextGradientMaps import ConvNeXtGradientMaps
    from Engine.Models.Visualization.VisualizationConvNext import VisualizationConvNexT

except ImportError as error:
    print(error)
    sys.exit(-1)



class ConvNeXtModel(ConvNetXProcess, ConvNeXtGradientMaps, VisualizationConvNexT):

    def __init__(self, arguments):

        ConvNetXProcess.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.convnext_loss_function
        self.depths = arguments.convnext_depths
        self.dims = arguments.convnext_dims
        self.input_shape = arguments.convnext_input_dimension
        self.optimizer_function = arguments.convnext_optimizer_function
        self.dropout_rate = arguments.convnext_dropout_rate
        self.layer_scale_init = arguments.convnext_layer_scale_init
        self.kernel_size = arguments.convnext_kernel_size
        self.number_classes = arguments.number_classes
        self.last_layer_activation = arguments.convnext_last_layer_activation
        self.convolutional_padding = arguments.convnext_convolutional_padding
        self.intermediary_activation = arguments.convnext_intermediary_activation
        self.model_name = "ConvNeXtModel"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def _convnext_block(self, neural_network_flow, dim, stage_idx, block_idx, drop_path_rate=0.0):
        """
        ConvNeXt block (modernized ResNet block).

        Args:
            neural_network_flow: Input tensor
            dim: Number of channels
            stage_idx: Stage index for naming
            block_idx: Block index for naming
            drop_path_rate: Drop path rate (stochastic depth)

        Returns:
            Output tensor
        """
        shortcut = neural_network_flow

        # Depthwise convolution (7x7)
        neural_network_flow = DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding=self.convolutional_padding,
            name=f'convnext_depthwise_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # Layer Normalization
        neural_network_flow = LayerNormalization(
            epsilon=1e-6,
            name=f'convnext_ln_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # Pointwise/Inverted Bottleneck (1x1 conv with 4x expansion)
        neural_network_flow = Conv2D(
            4 * dim,
            kernel_size=1,
            name=f'convnext_pwconv1_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # GELU activation
        neural_network_flow = Activation(
            self.intermediary_activation,
            name=f'convnext_act_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # Pointwise/Projection (1x1 conv back to dim)
        neural_network_flow = Conv2D(
            dim,
            kernel_size=1,
            name=f'convnext_pwconv2_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # Layer Scale
        neural_network_flow = LayerScale(
            init_value=self.layer_scale_init,
            name=f'convnext_layerscale_stage_{stage_idx}_block_{block_idx}')(neural_network_flow)

        # Residual connection
        neural_network_flow = Add(
            name=f'convnext_block_stage_{stage_idx}_block_{block_idx}')([shortcut, neural_network_flow])

        return neural_network_flow

    @staticmethod
    def _downsample_layer(neural_network_flow, dim, stage_idx):
        """
        Downsampling layer between stages.

        Args:
            neural_network_flow: Input tensor
            dim: Number of output channels
            stage_idx: Stage index for naming

        Returns:
            Downsampled tensor
        """
        # Layer Normalization
        neural_network_flow = LayerNormalization(
            epsilon=1e-6,
            name=f'downsample_ln_stage_{stage_idx}')(neural_network_flow)

        # Strided convolution for downsampling
        neural_network_flow = Conv2D(
            dim,
            kernel_size=2,
            strides=2,
            name=f'downsample_conv_stage_{stage_idx}')(neural_network_flow)

        return neural_network_flow

    def build_model(self):
        """
        Build the ConvNeXt architecture using Keras Functional API with proper layer naming for XAI.

        This method defines the structure of the ConvNeXt network with named layers
        to enable XAI visualization on specific layers.
        """
        inputs = Input(shape=self.input_shape, name='input_layer')

        # Stem (patchify layer) - 4x4 conv with stride 4
        # For spectrograms, we use 4x4 with stride 4 initially
        neural_network_flow = Conv2D(
            self.dims[0],
            kernel_size=4,
            strides=4,
            name='stem_conv')(inputs)

        neural_network_flow = LayerNormalization(
            epsilon=1e-6,
            name='stem_ln')(neural_network_flow)

        # Build stages
        for stage_idx in range(len(self.depths)):
            # Add downsampling layer between stages (except for first stage)
            if stage_idx > 0:
                neural_network_flow = self._downsample_layer(neural_network_flow, self.dims[stage_idx], stage_idx)

            # Add ConvNeXt blocks for this stage
            for block_idx in range(self.depths[stage_idx]):
                neural_network_flow = self._convnext_block(neural_network_flow,
                                                           self.dims[stage_idx], stage_idx, block_idx)

        # Global Average Pooling
        neural_network_flow = GlobalAveragePooling2D(name='global_avg_pooling')(neural_network_flow)

        # Layer Normalization before head
        neural_network_flow = LayerNormalization(epsilon=1e-6, name='head_ln')(neural_network_flow)

        # Dropout before classification
        neural_network_flow = Dropout(self.dropout_rate, name='head_dropout')(neural_network_flow)

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
        #         output_dir='Maps_ConvNeXt',
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
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, value):
        self._depths = value

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = value

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
    def layer_scale_init(self):
        return self._layer_scale_init

    @layer_scale_init.setter
    def layer_scale_init(self, value):
        self._layer_scale_init = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

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