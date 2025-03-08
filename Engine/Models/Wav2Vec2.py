#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Engine.Layers.MaskLayer import MaskCreator

try:
    import os
    import sys
    import glob
    import numpy

    import librosa
    import logging
    import argparse
    import tensorflow

    from tqdm import tqdm

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Activation

    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from Engine.Loss.ContrastiveLoss import ContrastiveLoss
    from Engine.Layers.QuantizerLayerMLP import QuantizationLayer

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from sklearn.utils import resample

    from Engine.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_LIST_FILTERS_ENCODER = [8, 16, 32]


class AudioWav2Vec2(MaskCreator):
    """
    @AudioWav2Vec2
        AudioWav2Vec2 is a class that implements a deep learning model based on the Wav2Vec2
        architecture for audio feature extraction and classification tasks. This model
        utilizes a convolutional encoder followed by transformer-based attention layers,
        with quantization layers applied for feature compression.
        The architecture is particularly suited for speech recognition or other audio-related tasks.

        The model consists of:
            - Convolutional layers for feature extraction from audio waveforms.
            - Transformer blocks with multi-head attention to capture long-range dependencies.
            - Quantization layers to compress the feature representations.
            - Dense layers for the final classification.

        This model also supports Contrastive Loss during the pre-training phase to learn useful
        feature representations from unlabelled data before fine-tuning with supervised learning.

    Reference:
        Baevski, A., Zhou, Y., & Mohamed, A. R. (2020). Wav2Vec 2.0: A Framework for Self-Supervised
        Learning of Speech Representations. *Proceedings of NeurIPS 2020*. https://arxiv.org/abs/2006.11477

    Example:
        >>> # Instantiate the model
        ...     model = AudioWav2Vec2(
        ...     number_classes=10,  # Number of output classes (e.g., 10 for classification)
        ...     last_layer_activation='softmax',  # Activation function for the output layer (e.g., 'softmax')
        ...     loss_function='categorical_crossentropy',  # Loss function for training (e.g., 'categorical_crossentropy')
        ...     optimizer_function='adam',  # Optimizer for training (e.g., 'adam')
        ...     quantization_units=4,  # Number of quantization units
        ...     key_dimension=64,  # Key dimension for the transformer attention
        ...     dropout_rate=0.2,  # Dropout rate for regularization
        ...     intermediary_layer_activation='relu',  # Activation function for intermediary layers (e.g., 'relu')
        ...     input_dimension=(128, 80),  # Input dimension for the model (e.g., Mel spectrogram)
        ...     number_heads=8,  # Number of attention heads in the transformer
        ...     kernel_size=3,  # Kernel size for the convolutional layers
        ...     list_filters_encoder=[64, 128, 256]  # List of filters for the convolutional encoder
        ...     )
        ... # Build the model
        ...     model.build_model()
        ...     # Compile and train the model
        ...     training_history = model.compile_and_train(
        ...     train_data=X_train,  # Input training data
        ...     train_labels=y_train,  # Training labels
        ...     epochs=10,  # Number of epochs for training
        ...     batch_size=32,  # Batch size for training
        ...     validation_data=(X_val, y_val)  # Optional validation data
        ...     )
        >>>

    Attributes:
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the Wav2Vec2 network.
        @list_filters_encoder (list[int]): List of filter sizes for the convolutional encoder layers.
        @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        @optimizer_function (str): The optimizer used for model training (e.g., 'adam').
        @kernel_size (int): The kernel size for the convolutional layers (e.g., 3 for 3x3 kernels).
        @quantization_units (int): The number of quantization units in the quantization layer.
        @key_dimension (int): The key dimension for the multi-head attention in the transformer.
        @intermediary_layer_activation (str): Activation function used in the intermediary layers (e.g., 'relu').
        @number_heads (int): The number of attention heads in the transformer block.
        @input_dimension (tuple): The shape of the input data (e.g., (128, 80) for Mel spectrograms).
        @number_classes (int): The number of output classes for classification.
        @dropout_rate (float): The dropout rate used for regularization.
        @last_layer_activation (str): The activation function used in the output layer (e.g., 'softmax').
        @model_name (str): The name of the model (default is "Wav2Vec2").
    """

    def __init__(self,
                 number_classes: int,
                 last_layer_activation: str,
                 loss_function: str,
                 optimizer_function: str,
                 quantization_units: int,
                 key_dimension: int,
                 dropout_rate: float,
                 intermediary_layer_activation: str,
                 input_dimension: tuple,
                 number_heads: int,
                 kernel_size: int,
                 list_filters_encoder=None):
        """
        Initialize the AudioWav2Vec2 model with specified hyperparameters.

        Args:
            @number_classes (int): The number of output classes for classification tasks.
            @last_layer_activation (str): The activation function for the output layer (e.g., 'softmax').
            @loss_function (str): The loss function used for training the model (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer used for training the model (e.g., 'adam').
            @quantization_units (int): The number of quantization units.
            @key_dimension (int): The key dimension for the transformer attention.
            @dropout_rate (float): The dropout rate for regularization.
            @intermediary_layer_activation (str): The activation function for intermediary layers (e.g., 'relu').
            @input_dimension (tuple): The input dimension for the model (e.g., (128, 80) for Mel spectrograms).
            @number_heads (int): The number of attention heads in the transformer block.
            @kernel_size (int): The kernel size for the convolutional layers.
            @list_filters_encoder (list[int], optional): A list of the number of filters for each convolutional encoder block.
        """

        if list_filters_encoder is None:
            list_filters_encoder = DEFAULT_LIST_FILTERS_ENCODER  # Default filters if not provided

        # Initialize model parameters
        self.neural_network_model = None
        self.list_filters_encoder = list_filters_encoder  # Convolutional encoder filters
        self.loss_function = loss_function  # Loss function for training
        self.optimizer_function = optimizer_function  # Optimizer function
        self.kernel_size = kernel_size  # Kernel size for the convolutional layers
        self.quantization_units = quantization_units  # Number of quantization units
        self.key_dimension = key_dimension  # Key dimension for transformer attention
        self.intermediary_layer_activation = intermediary_layer_activation  # Activation for intermediary layers
        self.number_heads = number_heads  # Number of attention heads for the transformer
        self.input_dimension = input_dimension  # Input data shape
        self.number_classes = number_classes  # Number of output classes for classification
        self.dropout_rate = dropout_rate  # Dropout rate for regularization
        self.last_layer_activation = last_layer_activation  # Activation for the output layer
        self.model_name = "Wav2Vec2"  # Model name

    def build_model(self) -> None:
        """
        Build the Wav2Vec2 model architecture using Keras.

        The model consists of the following components:
            - Convolutional encoder layers with the specified filters.
            - Transformer block with multi-head attention for capturing long-range dependencies.
            - Quantization layer for feature compression.
            - Dense layers for classification.

        The model is designed for audio processing tasks, particularly feature extraction
        from waveforms (e.g., Mel spectrograms).
        """

        # Input layer: reshape the input to match the expected format
        inputs = Input(shape=self.input_dimension)
        neural_network_flow = Reshape((128, 80, 1))(inputs)

        # Apply convolutional layers to extract features from the input audio
        for number_filters in self.list_filters_encoder:
            neural_network_flow = TimeDistributed(
                Conv1D(number_filters, self.kernel_size, strides=(2,),
                       activation=self.intermediary_layer_activation))(neural_network_flow)

        # Flatten the convolutional output to feed into the dense layers
        flatten_flow = TimeDistributed(Flatten())(neural_network_flow)

        # Apply a dense layer with specified activation
        dense_layer = TimeDistributed(Dense(self.number_classes,
                                            activation=self.intermediary_layer_activation))(flatten_flow)

        # Create causal mask for the transformer attention
        causal_mask = self.create_mask(128)

        # Transformer block with multi-head attention
        transformer_attention = MultiHeadAttention(num_heads=self.number_heads,
                                                   key_dim=4)(dense_layer, dense_layer, attention_mask=causal_mask)

        # Add residual connection and normalize using LayerNormalization
        transformer_attention = Add()([dense_layer, transformer_attention])
        transformer_attention = LayerNormalization()(transformer_attention)

        # Feed-forward network (fully connected layers)
        feedforward_network = Dense(self.number_classes, activation="relu")(transformer_attention)
        feedforward_network = Dense(self.number_classes, activation="relu")(feedforward_network)

        # Add the output of the feed-forward network and normalize
        transformer_output = Add()([transformer_attention, feedforward_network])
        transformer_output = LayerNormalization()(transformer_output)

        # Quantization layer for feature compression
        quantize_layer = TimeDistributed(QuantizationLayer(4), name="Quantization")(dense_layer)

        # Create the Keras model
        self.neural_network_model = Model(inputs=inputs, outputs=[transformer_output, quantize_layer],
                                          name=self.model_name)

        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=ContrastiveLoss(margin=0.5),
                                          metrics=['accuracy'])

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the neural network model using the specified training data and configuration.

        Args:
            train_data (tensorflow.Tensor): The input training data.
            train_labels (tensorflow.Tensor): The corresponding labels for the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of the batches for each training step.
            validation_data (tuple, optional): A tuple containing validation data and labels.

        Returns:
            tensorflow.keras.callbacks.History: The history object containing training metrics and performance.
        """
        logging.info("Starting the initial compilation and training phase.")

        # Step 1: Compile the model for initial training using ContrastiveLoss
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=ContrastiveLoss(margin=0.75))
        logging.info("Model compiled with ContrastiveLoss.")

        # Step 2: Train the model on the training data
        logging.info(f"Training model for {epochs} epochs with batch size {batch_size}.")
        self.neural_network_model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        logging.info("Initial training completed. Setting the model as non-trainable.")

        # Step 3: Set the model as non-trainable and flatten the output
        self.neural_network_model.trainable = False
        neural_network_flow = Flatten()(self.neural_network_model.output[0])

        # Step 4: Add a Dense layer with the number of classes and specified activation function
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)
        logging.info(f"Added Dense layer with {self.number_classes} classes and '{self.last_layer_activation}' activation.")

        # Step 5: Recreate the model with new output
        self.neural_network_model = Model(inputs=self.neural_network_model.inputs, outputs=neural_network_flow)

        # Step 6: Compile the new model with the specified optimizer, loss function, and accuracy metric
        self.neural_network_model.compile(optimizer=self.optimizer_function,
                                          loss=self.loss_function,
                                          metrics=['accuracy'])
        logging.info("Recompiled the model with the final configuration (optimizer, loss, metrics).")

        # Step 7: Train the model with the actual training data and labels
        logging.info(f"Final training for {epochs} epochs with batch size {batch_size}.")
        training_history = self.neural_network_model.fit(train_data, train_labels,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        logging.info("Training completed successfully.")

        return training_history

    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_filters_encoder(self):
        return self._list_filters_encoder

    @list_filters_encoder.setter
    def list_filters_encoder(self, value):
        self._list_filters_encoder = value

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
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

    @property
    def quantization_units(self):
        return self._quantization_units

    @quantization_units.setter
    def quantization_units(self, value):
        self._quantization_units = value

    @property
    def key_dimension(self):
        return self._key_dimension

    @key_dimension.setter
    def key_dimension(self, value):
        self._key_dimension = value

    @property
    def intermediary_layer_activation(self):
        return self._intermediary_layer_activation

    @intermediary_layer_activation.setter
    def intermediary_layer_activation(self, value):
        self._intermediary_layer_activation = value

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, value):
        self._number_heads = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

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