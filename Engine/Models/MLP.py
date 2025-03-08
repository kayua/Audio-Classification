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
    import tensorflow

    from tqdm import tqdm

    from tensorflow.keras import Model
    from sklearn.utils import resample

    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Bidirectional
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Engine.Evaluation.MetricsCalculator import MetricsCalculator

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


DEFAULT_LIST_DENSE_NEURONS = [128, 129]


class DenseModel:
    """
    DenseModel is a class representing a simple Multi-Layer Perceptron (MLP) architecture.

    This class allows for the creation, compilation, and training of an MLP model, a class
    of neural networks where each layer is fully connected to the next one. MLPs are commonly
    used for supervised learning tasks such as classification and regression.

    The model includes:
        - Flexible architecture with adjustable number of neurons in each hidden layer.
        - Dropout regularization for preventing overfitting.
        - Customizable activation functions for both hidden and output layers.
        - Choice of loss function and optimizer for model training.
        - Option to use LSTM layers for sequence data processing (default is dense layers).

     Reference for Multi-Layer Perceptron (MLP):
    "A Comprehensive Review on Multi-Layer Perceptrons" (IEEE, 2015) (https://ieeexplore.ieee.org/document/7267240)

    Attributes:
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the neural network.
        @list_number_neurons (list): A list of integers representing the number of neurons in each hidden layer.
        @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        @optimizer_function (str): The optimizer function used during model training (e.g., 'adam').
        @intermediary_layer_activation (str): Activation function used in hidden layers (e.g., 'relu').
        @input_dimension (tuple): The shape of the input data (e.g., (28, 28, 1) for images).
        @number_classes (int): The number of classes for classification (e.g., 10 for MNIST).
        @dropout_rate (float): The dropout rate for regularization (e.g., 0.2).
        @last_layer_activation (str): Activation function for the output layer (e.g., 'softmax').
        @model_name (str): The name of the model (default is "MLP").

    Example:
        >>> # Instantiate the model
        ...     model = DenseModel(
        ...     number_classes=10,  # Number of output classes
        ...     last_layer_activation='softmax',  # Activation function for output layer
        ...     loss_function='categorical_crossentropy',  # Loss function for classification task
        ...     optimizer_function='adam',  # Optimizer for training the model
        ...     dropout_rate=0.2,  # Dropout rate to avoid overfitting
        ...     intermediary_layer_activation='relu',  # Activation function for hidden layers
        ...     input_dimension=(28, 28, 1)  # Input shape of images (e.g., for MNIST)
        ...     )
        ...     # Build the model
        ...     model.build_model()
        ...     # Compile and train the model
        ...     training_history = model.compile_and_train(
        ...     train_data=X_train,  # Input training data
        ...     train_labels=y_train,  # Corresponding training labels
        ...     epochs=10,  # Number of training epochs
        ...     batch_size=32,  # Size of each training batch
        ...     validation_data=(X_val, y_val)  # Optional validation data
        ... )
        >>>

    """

    def __init__(self,
                 number_classes: int,
                 last_layer_activation: str,
                 loss_function: str,
                 optimizer_function: str,
                 dropout_rate: float,
                 intermediary_layer_activation: str,
                 input_dimension: tuple,
                 list_lstm_cells=None):
        """
        Initialize the DenseModel class.

        Args:
            @number_classes (int): The number of output classes for classification.
            @last_layer_activation (str): The activation function for the last layer (e.g., 'softmax' or 'sigmoid').
            @loss_function (str): The loss function to use (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer function to use (e.g., 'adam').
            @dropout_rate (float): The rate at which dropout will be applied to hidden layers.
            @intermediary_layer_activation (str): The activation function used in intermediary layers (e.g., 'relu').
            @input_dimension (tuple): The shape of the input data (e.g., (28, 28, 1) for MNIST).
            @list_lstm_cells (list, optional): A list representing the number of neurons in each
            LSTM layer. Default is None.
        """

        # If list_lstm_cells is not provided, use the default list of dense neurons.
        if list_lstm_cells is None:
            list_lstm_cells = DEFAULT_LIST_DENSE_NEURONS

        # Model initialization attributes.
        self.neural_network_model = None  # Placeholder for the Keras model.
        self.list_number_neurons = list_lstm_cells  # List of the number of neurons in each hidden layer.
        self.loss_function = loss_function  # Loss function for training.
        self.optimizer_function = optimizer_function  # Optimizer function for training.
        self.intermediary_layer_activation = intermediary_layer_activation  # Activation function for hidden layers.
        self.input_dimension = input_dimension  # Shape of the input data (e.g., images).
        self.number_classes = number_classes  # Number of output classes for classification.
        self.dropout_rate = dropout_rate  # Dropout rate for regularization.
        self.last_layer_activation = last_layer_activation  # Activation for the output layer.
        self.model_name = "MLP"  # Name of the model (default to 'MLP').

    def build_model(self) -> None:
        """
        Build the model architecture using Keras Functional API.

        This method defines the structure of the MLP model. It starts with an input layer,
        then creates a sequence of hidden layers (Dense + Dropout),
        and ends with an output layer for classification.

        The network is constructed in a modular way, allowing flexible configurations
        for the number of layers and neurons in each layer.

        The steps are as follows:
            1. Input layer: Accepts data with the specified input shape.
            2. Hidden layers: A series of Dense layers with specified activations and dropout regularization.
            3. Output layer: A Dense layer with the specified activation function for classification.

        The model is then created using Keras' Model API.
        """
        # Create input layer with the shape of the input data.
        inputs = Input(shape=self.input_dimension)

        # Initialize the flow of the neural network.
        neural_network_flow = inputs
        neural_network_flow = Flatten()(neural_network_flow)  # Flatten the input to a 1D vector.

        # Add hidden layers (Dense layers with activation and dropout).
        for _, number_neurons in enumerate(self.list_number_neurons):
            # Add a Dense layer with a specified number of neurons and activation function.
            neural_network_flow = Dense(number_neurons,
                                        activation=self.intermediary_layer_activation)(neural_network_flow)

            # Apply Dropout regularization after each hidden layer.
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        # Add the output layer with the number of classes and specified activation function.
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)

        # Create the model with input and output layers defined.
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compile and train the model.

        This method compiles the model with the specified optimizer and loss function,
        then trains the model using the provided training data.

        Args:
            train_data (tensorflow.Tensor): The input data for training (e.g., images or features).
            train_labels (tensorflow.Tensor): The target labels for training.
            epochs (int): The number of epochs for training.
            batch_size (int): The number of samples per batch for training.
            validation_data (tuple, optional): A tuple of validation data (input, labels). Default is None.

        Returns:
            tensorflow.keras.callbacks.History: The history object containing information about training progress.
        """
        # Compile the model with the specified loss function, optimizer, and metrics.
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model using the training data, labels, and validation data (if provided).
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)

        # Return the training history for further analysis.
        return training_history


