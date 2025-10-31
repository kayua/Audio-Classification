#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

from Engine.Models.Visualization.VisualizationConformer import VisualizationConformer

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
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Engine.Layers.ConformerBlock import ConformerBlock
    from Engine.Layers.TransposeLayer import TransposeLayer
    from Engine.Models.Process.Conformer_Process import ProcessConformer
    from Engine.GradientMap.ConformerGradientMaps import ConformerGradientMaps
    from Engine.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling

except ImportError as error:
    print(error)
    sys.exit(-1)


class Conformer(ProcessConformer, ConformerGradientMaps, VisualizationConformer):
    """
    Conformer Model Implementation with Integrated XAI Capabilities.

    This class implements a Conformer neural network architecture that combines
    convolutional layers with transformer blocks for sequence processing tasks.
    It includes comprehensive Explainable AI (XAI) methods for model interpretability.

    The model features:
    - Convolutional subsampling for input processing
    - Multiple Conformer blocks with multi-head attention
    - Global average pooling for sequence aggregation
    - Integrated Grad-CAM, Grad-CAM++, and Score-CAM visualization
    - Advanced heatmap interpolation and smoothing
    - Comprehensive visualization capabilities

    Inherits from:
        ProcessConformer: Base processing functionality
        ConformerGradientMaps: Gradient-based XAI methods
    """

    def __init__(self, arguments):
        """
        Initialize the Conformer model with configuration parameters.

        Args:
            arguments (object): Configuration object containing model parameters.
                Expected attributes:
                - conformer_loss_function: Loss function for model compilation
                - conformer_optimizer_function: Optimizer for training
                - conformer_number_filters_spectrogram: Number of filters for spectrogram
                - conformer_input_dimension: Input shape dimensions
                - conformer_number_conformer_blocks: Number of Conformer blocks
                - conformer_embedding_dimension: Dimension of embedding space
                - conformer_number_heads: Number of attention heads
                - number_classes: Number of output classes
                - conformer_size_kernel: Kernel size for convolutional layers
                - conformer_dropout_rate: Dropout rate for regularization
                - conformer_last_layer_activation: Activation function for output layer
        """
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

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """
        Build the Conformer neural network architecture.

        The architecture consists of:
        1. Input layer
        2. Convolutional subsampling
        3. Transpose layer for dimension rearrangement
        4. Embedding dense layer
        5. Multiple Conformer blocks
        6. Global average pooling
        7. Output classification layer

        Returns:
            None: The model is stored in self.neural_network_model
        """
        inputs = Input(shape=self.input_dimension, name='input_layer')

        neural_network_flow = ConvolutionalSubsampling(name='conv_subsampling')(inputs)
        neural_network_flow = TransposeLayer(perm=[0, 2, 1], name='transpose_layer')(neural_network_flow)
        neural_network_flow = Dense(self.embedding_dimension, name='embedding_dense')(neural_network_flow)

        for i in range(self.number_conformer_blocks):
            neural_network_flow = ConformerBlock(self.embedding_dimension,
                                                 self.number_heads,
                                                 self.input_dimension[0] // 2,
                                                 self.kernel_size,
                                                 self.dropout_rate,
                                                 name=f'conformer_block_{i}')(neural_network_flow)

        neural_network_flow = GlobalAveragePooling1D(name='global_avg_pooling')(neural_network_flow)
        neural_network_flow = Dense(self.number_classes,
                                    activation=self.last_layer_activation,
                                    name='output_layer'
                                    )(neural_network_flow)

        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)
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
        Compile and train the Conformer model with optional XAI visualization generation.

        Args:
            train_data (tensorflow.Tensor): Training input data
            train_labels (tensorflow.Tensor): Training target labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_data (tuple, optional): Validation data as (val_data, val_labels)
            generate_gradcam (bool): Whether to generate Grad-CAM visualizations
            xai_method (str): XAI method to use ('gradcam', 'gradcam++', 'scorecam')

        Returns:
            tensorflow.keras.callbacks.History: Training history object

        Example:
            >>> conformer = Conformer(arguments)
            >>> conformer.build_model()
            >>> history = conformer.compile_and_train(
            ...     train_data=X_train,
            ...     train_labels=y_train,
            ...     epochs=50,
            ...     batch_size=32,
            ...     validation_data=(X_val, y_val),
            ...     generate_gradcam=True,
            ...     xai_method='gradcam++'
            ... )
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

        training_history = self.neural_network_model.fit(train_data, train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data,
                                                         callbacks=callbacks if callbacks else None)

        # if validation_data is not None:
        #     print(f"Acurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")
        #
        # if generate_gradcam and validation_data is not None:
        #     val_data, val_labels = validation_data
        #
        #     self.generate_validation_visualizations(validation_data=val_data,
        #                                             validation_labels=val_labels,
        #                                             num_samples=128,
        #                                             output_dir='Maps_Conformer',
        #                                             xai_method=xai_method)

        return training_history

    def compile_model(self) -> None:
        """
        Compile the model without training.

        This method prepares the model for training by compiling it with
        the specified optimizer, loss function, and metrics.
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])

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