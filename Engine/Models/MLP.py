#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
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

    import tensorflow

    from tensorflow.keras import Model

    from tensorflow.keras.layers import LSTM

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import Bidirectional

    from Engine.Models.Process.MLP_Process import MLPProcess
    from tensorflow.keras.layers import GlobalAveragePooling1D

except ImportError as error:
    print(error)
    sys.exit(-1)



class DenseModel(MLPProcess):
    """
    @DenseModel

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

     Reference:
        "A Comprehensive Review on Multi-Layer Perceptrons" (IEEE, 2015)
         (https://ieeexplore.ieee.org/document/7267240)

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

    def __init__(self, arguments):


    # def __init__(self, number_classes: int, last_layer_activation: str, loss_function: str, optimizer_function: str,
    #             dropout_rate: float, intermediary_layer_activation: str, input_dimension: tuple, size_batch: int,
    #             number_splits: int, number_epochs: int, window_size_factor: int, decibel_scale_factor: int,
    #             hop_length: int, overlap: int, sample_rate: int, file_extension: str, list_lstm_cells=None):

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

#        # If list_lstm_cells is not provided, use the default list of dense neurons.
#        EvaluationProcess.__init__(arguments.batch_size, arguments.number_splits, arguments.number_epochs,
#                         arguments.mlp_optimizer_function, arguments.mlp_window_size_factor,
#                         arguments.mlp_decibel_scale_factor, arguments.mlp_hop_length, arguments.mlp_overlap,
#                         arguments.sample_rate, arguments.file_extension)

        # Model initialization attributes.
        super().__init__(arguments)

        self.neural_network_model = None  # Placeholder for the Keras model.
        self.list_number_neurons = arguments.mlp_list_dense_neurons  # List of the number of neurons in each hidden layer.
        self.loss_function = arguments.mlp_loss_function  # Loss function for training.
        self.optimizer_function = arguments.mlp_optimizer_function  # Optimizer function for training.
        self.intermediary_layer_activation = arguments.mlp_intermediary_layer_activation  # Activation function for hidden layers.
        self.input_dimension = arguments.mlp_input_dimension  # Shape of the input data (e.g., images).
        self.number_classes = arguments.number_classes  # Number of output classes for classification.
        self.dropout_rate = arguments.mlp_dropout_rate  # Dropout rate for regularization.
        self.last_layer_activation = arguments.mlp_last_layer_activation  # Activation for the output layer.
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
        self.neural_network_model.summary()

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
                                                         validation_data=validation_data
                                                         )
        # Return the training history for further analysis.
        return training_history


    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_number_neurons(self):
        return self._list_number_neurons

    @list_number_neurons.setter
    def list_number_neurons(self, value):
        self._list_number_neurons = value

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
    def intermediary_layer_activation(self):
        return self._intermediary_layer_activation

    @intermediary_layer_activation.setter
    def intermediary_layer_activation(self, value):
        self._intermediary_layer_activation = value

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