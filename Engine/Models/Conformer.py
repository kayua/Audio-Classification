#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import sys
    import tensorflow

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape

    from tensorflow.keras.layers import Concatenate

    from Engine.Layers.ConformerBlock import ConformerBlock
    from Engine.Layers.TransposeLayer import TransposeLayer

    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Engine.Models.Process.EvaluationProcess import EvaluationProcess

    from Engine.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling


except ImportError as error:
     print(error)
     print("1. Install requirements:")
     print("  pip3 install --upgrade pip")
     print("  pip3 install -r requirements.txt ")
     print()
     sys.exit(-1)


class Conformer(EvaluationProcess):
    """
    @Conformer

        Conformer is a deep learning model designed for sequential tasks, such as speech
        recognition or audio classification. It integrates the Convolutional Neural
        Network (CNN) and Transformer architectures, leveraging their respective
        strengths in capturing both local and global dependencies in sequential data.

        The Conformer model consists of:
        - Convolutional subsampling layer for feature extraction.
        - Transformer blocks with multi-head attention for capturing long-range dependencies.
        - A final dense layer for classification.

    Reference:
        Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., ... & Wang,
        Q. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition.
        *arXiv preprint arXiv:2005.08100*. https://arxiv.org/abs/2005.08100

    Attributes:
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the Conformer network.
        @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        @optimizer_function (str): The optimizer function (e.g., 'adam').
        @number_filters_spectrogram (int): The number of filters used in the spectrogram extraction.
        @input_dimension (tuple): The input dimension for the model (e.g., (128, 80) for Mel spectrogram).
        @number_conformer_blocks (int): The number of Conformer blocks in the model.
        @embedding_dimension (int): The dimensionality of the embedding layer.
        @number_heads (int): The number of attention heads in the multi-head attention mechanism.
        @number_classes (int): The number of output classes for classification.
        @kernel_size (tuple): The kernel size for the convolutional layers.
        @dropout_rate (float): The dropout rate for regularization.
        @last_layer_activation (str): The activation function for the output layer (e.g., 'softmax').
        @model_name (str): The name of the model (default is "Conformer").

    Example:
        >>> # Instantiate the Conformer model
        ...     model = Conformer(
        ...     number_conformer_blocks=12,  # Number of Conformer blocks
        ...     embedding_dimension=512,  # Embedding dimension for the layers
        ...     number_heads=8,  # Number of attention heads
        ...     size_kernel=(3, 3),  # Kernel size for convolution layers
        ...     number_classes=10,  # Number of output classes
        ...     last_layer_activation='softmax',  # Activation for the output layer
        ...     loss_function='categorical_crossentropy',  # Loss function
        ...     optimizer_function='adam',  # Optimizer function
        ...     number_filters_spectrogram=64,  # Number of filters for the spectrogram
        ...     dropout_rate=0.2,  # Dropout rate for regularization
        ...     input_dimension=(128, 80)  # Input shape (e.g., Mel spectrogram dimensions)
        ...     )
        ...     # Build the model
        ...     model.build_model()
        ...
        ...     # Compile and train the model
        ...     training_history = model.compile_and_train(
        ...     train_data=X_train,  # Training data
        ...     train_labels=y_train,  # Training labels
        ...     epochs=10,  # Number of training epochs
        ...     batch_size=32,  # Batch size for training
        ...     validation_data=(X_val, y_val)  # Optional validation data
        ...     )
        >>>
    """

    def __init__(self, number_conformer_blocks: int, embedding_dimension: int, number_heads: int, size_kernel: tuple,
                 number_classes: int, last_layer_activation: str, loss_function: str, optimizer_function: str,
                 number_filters_spectrogram: int, dropout_rate: float, input_dimension: tuple, size_batch: int,
                 number_splits: int, number_epochs: int, window_size_factor: int, decibel_scale_factor: int,
                 hop_length: int, overlap: int, sample_rate: int, file_extension: str):
        """
        Initialize the Conformer model with the specified hyperparameters.

        Args:
            @number_conformer_blocks (int): Number of Conformer blocks in the model.
            @embedding_dimension (int): The embedding dimension for layers.
            @number_heads (int): Number of attention heads in multi-head attention.
            @size_kernel (tuple): The kernel size for convolution layers.
            @number_classes (int): The number of output classes for classification.
            @last_layer_activation (str): Activation for the output layer (e.g., 'softmax').
            @loss_function (str): The loss function for training (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer function for training (e.g., 'adam').
            @number_filters_spectrogram (int): Number of filters used in the spectrogram extraction.
            @dropout_rate (float): Dropout rate for regularization.
            @input_dimension (tuple): The shape of the input data.
        """

        super().__init__(size_batch, number_splits, number_epochs, optimizer_function, window_size_factor,
                         decibel_scale_factor, hop_length, overlap, sample_rate, file_extension)
        self.neural_network_model = None
        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.number_filters_spectrogram = number_filters_spectrogram
        self.input_dimension = input_dimension
        self.number_conformer_blocks = number_conformer_blocks
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.number_classes = number_classes
        self.kernel_size = size_kernel
        self.dropout_rate = dropout_rate
        self.last_layer_activation = last_layer_activation
        self.model_name = "Conformer"

    def build_model(self) -> None:
        """
        Build the Conformer model architecture using Keras.

        The model consists of the following components:
            - Convolutional subsampling layer for feature extraction.
            - Conformer blocks (Transformer-like blocks with convolution).
            - A final dense layer for classification.

        The model is designed for sequential tasks, such as speech recognition or audio classification.
        """

        inputs = Input(shape=self.input_dimension)

        # Initial convolutional subsampling layer
        neural_network_flow = ConvolutionalSubsampling()(inputs)
        neural_network_flow = TransposeLayer(perm=[0, 2, 1])(neural_network_flow)
        neural_network_flow = Dense(self.embedding_dimension)(neural_network_flow)

        # Adding multiple Conformer blocks
        for _ in range(self.number_conformer_blocks):
            neural_network_flow = ConformerBlock(self.embedding_dimension,
                                                 self.number_heads,
                                                 self.input_dimension[0] // 2,
                                                 self.kernel_size,
                                                 self.dropout_rate)(neural_network_flow)

        # Global average pooling after the last Conformer block
        neural_network_flow = GlobalAveragePooling1D()(neural_network_flow)

        # Final dense layer for classification
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)

        # Create the model
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compile and train the Conformer model using the specified training data and configuration.

        Args:
            train_data (tensorflow.Tensor): The input training data.
            train_labels (tensorflow.Tensor): The corresponding labels for the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of the batches for each training step.
            validation_data (tuple, optional): A tuple containing validation data and labels.

        Returns:
            tensorflow.keras.callbacks.History: The history object containing training metrics and performance.
        """

        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history

    def compile_model(self) -> None:
        """
        Compiles the Conformer model with the specified loss function and optimizer.

        This method prepares the model for training by setting the optimizer, loss function, and metrics.
        """

        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])


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
