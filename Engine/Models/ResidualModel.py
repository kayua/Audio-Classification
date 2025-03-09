#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']

from Engine.Models.Process.Residual_Process import ResidualProcess

try:
    import sys

    import tensorflow

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input

    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import MaxPooling2D

except ImportError as error:

    print(error)
    sys.exit(-1)



class ResidualModel(ResidualProcess): #(EvaluationProcess):
    """
    @ResidualModel
        ResidualModel is a class that implements a residual convolutional neural
        network (CNN) for classification tasks.

        This model architecture incorporates residual blocks, which are used to combat the
        vanishing gradient problem in deep neural networks by allowing gradients to flow more
        easily through the layers. The network consists of convolutional layers, max pooling,
        dropout regularization, and dense layers, optimized for classification problems.
        Residual connections are added to improve performance, especially for deeper architectures.

        The ResidualModel can be customized with various hyperparameters, including the number
        of convolutional filters, size of filters, size of pooling layers, dropout rate, and
        activation functions. The model is typically used for image classification tasks but
        can be adapted for other types of input data.

    Reference for Residual Convolutional Neural Networks:
        Paim, K. O., Rohweder, R., Recamonde-Mendoza, M., Mansilha, R. B., & Cordeiro, W. (2024).
        Acoustic identification of Ae. aegypti mosquitoes using smartphone apps and residual
        convolutional neural networks. *Biomedical Signal Processing and Control,
        95, 106342. https://doi.org/10.1016/j.bspc.2024.106342

    Example:
        >>> # Instantiate the model
        ...     model = ResidualModel(
        ...     input_dimension=(128, 128, 3),  # Input shape of the image (e.g., 128x128 RGB image)
        ...     convolutional_padding='same',  # Padding type for convolution layers
        ...     intermediary_activation='relu',  # Activation function for intermediary layers
        ...     last_layer_activation='softmax',  # Activation function for output layer (e.g., 'softmax')
        ...     number_classes=10,  # Number of output classes (e.g., 10 for multi-class classification)
        ...     size_convolutional_filters=(3, 3),  # Size of convolutional filters (e.g., 3x3)
        ...     size_pooling=(2, 2),  # Size of max-pooling filters (e.g., 2x2)
        ...     filters_per_block=[32, 64, 128],  # Number of filters per block
        ...     loss_function='categorical_crossentropy',  # Loss function for multi-class classification
        ...     optimizer_function='adam',  # Optimizer used for training
        ...     dropout_rate=0.2  # Dropout rate to prevent overfitting
        ...     )
        ...
        ...     # Build the model
        ...     model.build_model()
        ...
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
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the residual CNN.
        @filters_per_block (list[int]): List containing the number of filters in each residual block.
        @input_shape (tuple): The shape of the input data (e.g., (128, 128, 3) for RGB images).
        @size_convolutional_filters (tuple): The size of the convolutional filters (e.g., (3, 3)).
        @size_pooling (tuple): The size of the max-pooling filters (e.g., (2, 2)).
        @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        @optimizer_function (str): The optimizer function used during model training (e.g., 'adam').
        @dropout_rate (float): The dropout rate used for regularization.
        @number_classes (int): The number of output classes for classification (e.g., 10 for multi-class classification).
        @last_layer_activation (str): Activation function for the output layer (e.g., 'softmax').
        @convolutional_padding (str): Padding strategy for convolution layers (e.g., 'same').
        @intermediary_activation (str): Activation function used in intermediary layers (e.g., 'relu').
        @model_name (str): The name of the model (default is "ResidualModel").
    """
    def __init__(self, arguments):



#    def __init__(self, input_dimension: tuple[int, int, int], convolutional_padding: str, intermediary_activation: str,
#                 last_layer_activation: str, number_classes: int, size_convolutional_filters: tuple[int, int],
#                 size_pooling: tuple[int, int], filters_per_block: list[int], loss_function: str,
#                 optimizer_function: str, dropout_rate: float, size_batch: int, number_splits: int, number_epochs: int,
#                 window_size_factor: int, decibel_scale_factor: int, hop_length: int, overlap: int, sample_rate: int,
#                 file_extension: str):

        """
        Initialize the ResidualModel class.

        Args:
            @input_dimension (tuple): The shape of the input data (e.g., (128, 128, 3) for RGB images).
            @convolutional_padding (str): Padding strategy for convolution layers (e.g., 'same' or 'valid').
            @intermediary_activation (str): Activation function used in intermediary layers (e.g., 'relu').
            @last_layer_activation (str): Activation function for the output layer (e.g., 'softmax').
            @number_classes (int): The number of output classes for classification.
            @size_convolutional_filters (tuple): The size of the convolutional filters (e.g., (3, 3)).
            @size_pooling (tuple): The size of the pooling filters (e.g., (2, 2)).
            @filters_per_block (list[int]): A list containing the number of filters for each residual block.
            @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer function used during model training (e.g., 'adam').
            @dropout_rate (float): The dropout rate for regularization.
        """

        # Initialize model parameters
#        super().__init__(size_batch, number_splits, number_epochs, optimizer_function, window_size_factor,
#                         decibel_scale_factor, hop_length, overlap, sample_rate, file_extension)

        ResidualProcess.__init__(self, arguments)
        self.neural_network_model = None  # Placeholder for the Keras model
        self.loss_function = arguments.residual_loss_function  # Loss function used during training
        self.size_pooling = arguments.residual_size_pooling  # Pooling size for down-sampling
        self.filters_per_block = arguments.residual_filters_per_block  # Number of filters in each block
        self.input_shape = arguments.residual_input_dimension  # Shape of the input data
        self.optimizer_function = arguments.residual_optimizer_function  # Optimizer used for training
        self.dropout_rate = arguments.residual_dropout_rate  # Dropout rate for regularization
        self.size_convolutional_filters = arguments.residual_size_convolutional_filters  # Size of convolutional filters
        self.number_classes = arguments.number_classes  # Number of output classes
        self.last_layer_activation = arguments.residual_last_layer_activation  # Activation function for the output layer
        self.convolutional_padding = arguments.residual_convolutional_padding  # Padding type for convolution layers
        self.intermediary_activation = arguments.residual_intermediary_activation  # Activation for intermediary layers
        self.model_name = "ResidualModel"  # Name of the model

    def build_model(self):
        """
        Build the model architecture using Keras Functional API.

        This method defines the structure of the residual convolutional network, incorporating
        residual connections to improve training performance. The model consists of:
            - Convolutional layers with specified padding and activation.
            - Residual connections that allow the output of each layer to be added to the input.
            - Max pooling for down-sampling.
            - Dropout layers for regularization.
            - Final dense layer for classification.

        The network architecture is as follows:
            1. Input layer: Accepts the input data with the specified shape.
            2. Residual blocks: A sequence of convolutional layers, followed by a residual connection and max-pooling.
            3. Final dense layer: The output layer with the specified number of classes and activation function.
        """

        # Create input layer with the shape of the input data.
        inputs = Input(shape=self.input_shape)
        neural_network_flow = inputs  # Initialize the flow of data through the network

        # Add residual blocks.
        for number_filters in self.filters_per_block:
            residual_flow = neural_network_flow  # Save the input for the residual connection

            # Add two convolutional layers followed by activation.
            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)
            neural_network_flow = Conv2D(number_filters, self.size_convolutional_filters,
                                         activation=self.intermediary_activation,
                                         padding=self.convolutional_padding)(neural_network_flow)

            # Add the residual connection (skip connection).
            neural_network_flow = Concatenate()([neural_network_flow, residual_flow])

            # Apply max pooling and dropout after each residual block.
            neural_network_flow = MaxPooling2D(self.size_pooling)(neural_network_flow)
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        # Flatten the output before passing it to the dense layer.
        neural_network_flow = Flatten()(neural_network_flow)

        # Add the output layer with the specified number of classes.
        neural_network_flow = Dense(self.number_classes,
                                    activation=self.last_layer_activation)(neural_network_flow)

        # Create the Keras model with the defined input and output layers.
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)
        self.neural_network_model.summary()

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:
        """
        Compile and train the model.

        This method compiles the model with the specified optimizer and loss function,
        and trains the model using the provided training data.

        Args:
            @train_data (tensorflow.Tensor): The input data for training (e.g., images).
            @train_labels (tensorflow.Tensor): The target labels for training.
            @epochs (int): The number of epochs for training.
            @batch_size (int): The batch size for training.
            @validation_data (tuple, optional): A tuple of validation data (input, labels). Default is None.

        Returns:
            tensorflow.keras.callbacks.History: The training history, containing information about the training process.
        """

        # Compile the model with the specified loss function, optimizer, and metrics.
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model with the provided data, labels, and validation data (if available).
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)

        # Return the training history object.
        return training_history

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
    def size_pooling(self):
        return self._size_pooling

    @size_pooling.setter
    def size_pooling(self, value):
        self._size_pooling = value

    @property
    def filters_per_block(self):
        return self._filters_per_block

    @filters_per_block.setter
    def filters_per_block(self, value):
        self._filters_per_block = value

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
