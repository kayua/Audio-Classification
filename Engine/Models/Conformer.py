#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import os
    import sys
    import glob
    import numpy
    import librosa
    import logging
    import tensorflow

    from tqdm import tqdm

    from sklearn.utils import resample

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Engine.Layers.ConformerBlock import ConformerBlock
    from Engine.Evaluation.MetricsCalculator import MetricsCalculator
    from Engine.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class Conformer:
    """
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
        neural_network_model (tensorflow.keras.Model): The Keras model representing the Conformer network.
        loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        optimizer_function (str): The optimizer function (e.g., 'adam').
        number_filters_spectrogram (int): The number of filters used in the spectrogram extraction.
        input_dimension (tuple): The input dimension for the model (e.g., (128, 80) for Mel spectrogram).
        number_conformer_blocks (int): The number of Conformer blocks in the model.
        embedding_dimension (int): The dimensionality of the embedding layer.
        number_heads (int): The number of attention heads in the multi-head attention mechanism.
        number_classes (int): The number of output classes for classification.
        kernel_size (tuple): The kernel size for the convolutional layers.
        dropout_rate (float): The dropout rate for regularization.
        last_layer_activation (str): The activation function for the output layer (e.g., 'softmax').
        model_name (str): The name of the model (default is "Conformer").

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

    def __init__(self,
                 number_conformer_blocks: int,
                 embedding_dimension: int,
                 number_heads: int,
                 size_kernel: tuple,
                 number_classes: int,
                 last_layer_activation: str,
                 loss_function: str,
                 optimizer_function: str,
                 number_filters_spectrogram: int,
                 dropout_rate: float,
                 input_dimension: tuple):
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

