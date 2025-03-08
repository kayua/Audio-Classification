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

    import logging
    import librosa
    import argparse
    import tensorflow

    from tqdm import tqdm

    from sklearn.utils import resample
    from tensorflow.keras import models

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.layers import TimeDistributed
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from Engine.Layers.CLSTokenLayer import CLSTokenLayer

    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Engine.Evaluation.MetricsCalculator import MetricsCalculator

    from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


class AudioSpectrogramTransformer:
    """
    @AudioSpectrogramTransformer

        The Audio Spectrogram Transformer (AST) is a deep learning model designed for
        audio classification tasks. It leverages a transformer architecture, similar
        to the one used in natural language processing, and applies it to spectrograms
        of audio data. The model is composed of a transformer encoder, followed by
        a feedforward network and a classification layer.

    Reference:
        Gong, Y., Xu, M., Li, J., Liu, Z., & Xu, B. (2021). AST: Audio Spectrogram Transformer.
        *arXiv preprint arXiv:2104.01778*. https://arxiv.org/abs/2104.01778

    Attributes:
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the AST network.
        @head_size (int): The size of the attention heads in multi-head attention layers.
        @number_heads (int): The number of attention heads in the multi-head attention mechanism.
        @number_blocks (int): The number of transformer blocks in the model.
        @number_classes (int): The number of output classes for classification.
        @patch_size (tuple): The size of each spectrogram patch.
        @dropout (float): The dropout rate for regularization.
        @optimizer_function (str): The optimizer function used during training (e.g., 'adam').
        @loss_function (str): The loss function used during training (e.g., 'categorical_crossentropy').
        @normalization_epsilon (float): The epsilon value used for layer normalization.
        @last_activation_layer (str): The activation function for the last layer (e.g., 'softmax').
        @projection_dimension (int): The dimension for the linear projection layer.
        @intermediary_activation (str): The activation function used in intermediary layers (e.g., 'relu').
        @number_filters_spectrogram (int): The number of filters used in the spectrogram extraction.
        @model_name (str): The name of the model (default is "AST").

    Example:
        >>> # Instantiate the AudioSpectrogramTransformer model
        ...     model = AudioSpectrogramTransformer(
        ...     projection_dimension=512,  # Projection dimension for transformer
        ...     head_size=64,  # Head size for multi-head attention
        ...     num_heads=8,  # Number of attention heads
        ...     number_blocks=12,  # Number of transformer blocks
        ...     number_classes=10,  # Number of output classes
        ...     patch_size=(32, 32),  # Spectrogram patch size
        ...     dropout=0.1,  # Dropout rate for regularization
        ...     intermediary_activation='relu',  # Activation function for intermediate layers
        ...     loss_function='categorical_crossentropy',  # Loss function for training
        ...     last_activation_layer='softmax',  # Activation function for the last layer
        ...     optimizer_function='adam',  # Optimizer function
        ...     normalization_epsilon=1e-6,  # Epsilon for layer normalization
        ...     number_filters_spectrogram=64  # Number of filters in the spectrogram
        ...     )
        ...     # Build the model
        ...     model.build_model(number_patches=10)
        ...
        ...     # Compile and train the model
        ...     training_history = model.compile_and_train(
        ...     train_data=X_train,  # Training data
        ...     train_labels=y_train,  # Training labels
        ...     epochs=10,  # Number of epochs for training
        ...     batch_size=32,  # Batch size
        ...     validation_data=(X_val, y_val)  # Optional validation data
        ...     )
        >>>

    """

    def __init__(self,
                 projection_dimension: int,
                 head_size: int,
                 num_heads: int,
                 number_blocks: int,
                 number_classes: int,
                 patch_size: tuple,
                 dropout: float,
                 intermediary_activation: str,
                 loss_function: str,
                 last_activation_layer: str,
                 optimizer_function: str,
                 normalization_epsilon: float,
                 number_filters_spectrogram):
        """
        Initialize the AudioSpectrogramTransformer model with the specified hyperparameters.

        Args:
            @projection_dimension (int): The projection dimension for each input patch.
            @head_size (int): The size of each attention head in the multi-head attention mechanism.
            @num_heads (int): The number of attention heads in the multi-head attention layer.
            @number_blocks (int): The number of transformer blocks (layers) in the encoder.
            @number_classes (int): The number of output classes for the classification task.
            @patch_size (tuple): The size of the input spectrogram patches, defined by the
             (time, frequency) dimensions.
            @dropout (float): The dropout rate to be applied for regularization during training.
            @intermediary_activation (str): The activation function to be used in the intermediate
             layers, typically 'relu' or 'gelu'.
            @loss_function (str): The loss function to use for training, such as 'categorical_crossentropy'
             for multi-class classification.
            @last_activation_layer (str): The activation function for the final output layer,
             commonly 'softmax' for classification.
            @optimizer_function (str): The optimizer function to be used, such as 'adam' or 'sgd'.
            @normalization_epsilon (float): A small constant added for numerical stability in
             layer normalization.
            @number_filters_spectrogram (int): The number of filters to apply for feature extraction
             from the spectrogram before the transformer.

        """
        self.neural_network_model = None
        self.head_size = head_size
        self.number_heads = num_heads
        self.number_blocks = number_blocks
        self.number_classes = number_classes
        self.patch_size = patch_size
        self.dropout = dropout
        self.optimizer_function = optimizer_function
        self.loss_function = loss_function
        self.normalization_epsilon = normalization_epsilon
        self.last_activation_layer = last_activation_layer
        self.projection_dimension = projection_dimension
        self.intermediary_activation = intermediary_activation
        self.number_filters_spectrogram = number_filters_spectrogram
        self.model_name = "AST"

    def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        The transformer encoder applies a series of transformer blocks (multi-head attention and feedforward layers)
        on the input tensor.
        """

        # Iterate over the number of transformer blocks
        for _ in range(self.number_blocks):
            # Apply layer normalization to the input tensor
            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(inputs)

            # Apply multi-head self-attention
            neural_model_flow = MultiHeadAttention(key_dim=self.head_size, num_heads=self.number_heads,
                                                   dropout=self.dropout)(neural_model_flow, neural_model_flow)

            # Apply dropout for regularization
            neural_model_flow = Dropout(self.dropout)(neural_model_flow)

            # Add the input tensor to the output of the self-attention layer (residual connection)
            neural_model_flow = Add()([neural_model_flow, inputs])

            # Apply layer normalization after the residual connection
            neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)

            # Apply a feedforward layer (MLP layer) to transform the features
            neural_model_flow = Dense(neural_model_flow.shape[2],
                                      activation=self.intermediary_activation)(neural_model_flow)

            # Apply dropout for regularization
            neural_model_flow = Dropout(self.dropout)(neural_model_flow)

            # Add the input tensor to the output of the MLP layer (residual connection)
            inputs = Add()([neural_model_flow, inputs])

        return inputs

    def build_model(self, number_patches: int) -> tensorflow.keras.models.Model:
        """
        Builds the AST model consisting of a transformer encoder and a classification head.

        Args:
            number_patches (int): The number of patches in the input spectrogram.
        """

        # Define the input layer with shape (number_patches, projection_dimension)
        inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
        input_flatten = TimeDistributed(Flatten())(inputs)
        linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)

        cls_tokens_layer = CLSTokenLayer(self.projection_dimension)(linear_projection)
        # Concatenate the CLS token to the input patches
        neural_model_flow = Concatenate(axis=1)([cls_tokens_layer, linear_projection])

        # Add positional embeddings to the input patches
        positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches,
                                                                self.projection_dimension)(linear_projection)
        neural_model_flow += positional_embeddings_layer

        # Pass the input through the transformer encoder
        neural_model_flow = self.transformer_encoder(neural_model_flow)

        # Apply layer normalization
        neural_model_flow = LayerNormalization(epsilon=self.normalization_epsilon)(neural_model_flow)
        # Apply global average pooling
        neural_model_flow = GlobalAveragePooling1D()(neural_model_flow)
        # Apply dropout for regularization
        neural_model_flow = Dropout(self.dropout)(neural_model_flow)
        # Define the output layer with the specified number of classes and activation function
        outputs = Dense(self.number_classes, activation=self.last_activation_layer)(neural_model_flow)

        # Create the Keras model
        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)

        return self.neural_network_model

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the AST model using the specified training data and configuration.

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


    @property
    def head_size(self):
        return self._head_size

    @head_size.setter
    def head_size(self, head_size: int):
        self._head_size = head_size

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, num_heads: int):
        self._number_heads = num_heads

    @property
    def number_blocks(self):
        return self._number_blocks

    @number_blocks.setter
    def number_blocks(self, number_blocks: int):
        self._number_blocks = number_blocks

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, number_classes: int):
        self._number_classes = number_classes

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: tuple):
        self._patch_size = patch_size

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, dropout: float):
        self._dropout = dropout

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, optimizer_function: str):
        self._optimizer_function = optimizer_function

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: str):
        self._loss_function = loss_function

    @property
    def normalization_epsilon(self):
        return self._normalization_epsilon

    @normalization_epsilon.setter
    def normalization_epsilon(self, normalization_epsilon: float):
        self._normalization_epsilon = normalization_epsilon

    @property
    def last_activation_layer(self):
        return self._last_activation_layer

    @last_activation_layer.setter
    def last_activation_layer(self, last_activation_layer: str):
        self._last_activation_layer = last_activation_layer

    @property
    def projection_dimension(self):
        return self._projection_dimension

    @projection_dimension.setter
    def projection_dimension(self, projection_dimension: int):
        self._projection_dimension = projection_dimension

    @property
    def intermediary_activation(self):
        return self._intermediary_activation

    @intermediary_activation.setter
    def intermediary_activation(self, intermediary_activation: str):
        self._intermediary_activation = intermediary_activation

    @property
    def number_filters_spectrogram(self):
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, number_filters_spectrogram: int):
        self._number_filters_spectrogram = number_filters_spectrogram

