#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

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

    import tensorflow

    from tensorflow.keras import Model

    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input

    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from Engine.Models.Process.LSTM_Process import ProcessLSTM
    from tensorflow.keras.layers import GlobalAveragePooling1D


except ImportError as error:
    print(error)
    sys.exit(-1)



class AudioLSTM(ProcessLSTM): #(EvaluationProcess):
    """
    @AudioLSTM

        AudioLSTM is a class implementing a deep learning model based on Long Short-Term
        Memory (LSTM) networks for audio classification tasks. The model utilizes multiple
        LSTM layers followed by a dense layer for classification.
        This type of architecture is particularly suited for tasks involving sequential data,
        such as speech recognition, audio event detection, and other time-series classification tasks.

        The model consists of:
            - LSTM layers to capture temporal dependencies in the audio data.
            - Dropout layers for regularization to prevent overfitting.
            - A final dense layer with softmax (or other) activation for classification.

    Reference:
        Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
        *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

    Attributes:
        @neural_network_model (tensorflow.keras.Model): The Keras model representing the LSTM network.
        @list_lstm_cells (list[int]): List of the number of cells in each LSTM layer.
        @loss_function (str): The loss function used during model training (e.g., 'categorical_crossentropy').
        @optimizer_function (str): The optimizer used for model training (e.g., 'adam').
        @recurrent_activation (str): The activation function for the recurrent state (e.g., 'sigmoid').
        @intermediary_layer_activation (str): Activation function used in the intermediary layers (e.g., 'tanh').
        @input_dimension (tuple): The shape of the input data (e.g., (128, 80) for Mel spectrograms).
        @number_classes (int): The number of output classes for classification.
        @dropout_rate (float): The dropout rate used for regularization.
        @last_layer_activation (str): The activation function used in the output layer (e.g., 'softmax').
        @model_name (str): The name of the model (default is "LSTM").

    Example:
        >>> # Instantiate the model
        ...     model = AudioLSTM(
        ...     number_classes=10,  # Number of output classes (e.g., 10 for classification)
        ...     last_layer_activation='softmax',  # Activation function for the output layer (e.g., 'softmax')
        ...     loss_function='categorical_crossentropy',  # Loss function for training (e.g., 'categorical_crossentropy')
        ...     optimizer_function='adam',  # Optimizer for training (e.g., 'adam')
        ...     dropout_rate=0.2,  # Dropout rate for regularization
        ...     intermediary_layer_activation='tanh',  # Activation function for intermediary layers (e.g., 'tanh')
        ...     recurrent_activation='sigmoid',  # Activation function for the recurrent state (e.g., 'sigmoid')
        ...     input_dimension=(128, 80),  # Input dimension for the model (e.g., Mel spectrogram)
        ...     list_lstm_cells=[64, 128]  # List of LSTM cell sizes for each LSTM layer
        ...     )
        ...     # Build the model
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

    """

    def __init__(self, arguments):


#    def __init__(self, number_classes: int, last_layer_activation: str, loss_function: str, optimizer_function: str,
#                 dropout_rate: float, intermediary_layer_activation: str, recurrent_activation: str,
#                 input_dimension: tuple, size_batch: int, number_splits: int, number_epochs: int,
#                 window_size_factor: int, decibel_scale_factor: int, hop_length: int, overlap: int, sample_rate: int,
#                 file_extension: str, list_lstm_cells=None):

        """
        Initialize the AudioLSTM model with specified hyperparameters.

        Args:
            @number_classes (int): The number of output classes for classification tasks.
            @last_layer_activation (str): The activation function for the output layer (e.g., 'softmax').
            @loss_function (str): The loss function used for training the model (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer used for training the model (e.g., 'adam').
            @dropout_rate (float): The dropout rate for regularization.
            @intermediary_layer_activation (str): The activation function for intermediary layers (e.g., 'tanh').
            @recurrent_activation (str): The activation function for the recurrent state (e.g., 'sigmoid').
            @input_dimension (tuple): The input dimension for the model (e.g., (128, 80) for Mel spectrograms).
            @list_lstm_cells (list[int], optional): A list of the number of cells for each LSTM layer.
        """

#        super().__init__(size_batch, number_splits, number_epochs, optimizer_function, window_size_factor,
#                         decibel_scale_factor, hop_length, overlap, sample_rate, file_extension)

        # Initialize model parameters
        ProcessLSTM.__init__(self, arguments)

        self.neural_network_model = None
        self.list_lstm_cells = arguments.lstm_list_lstm_cells  # Number of cells in each LSTM layer
        self.loss_function = arguments.lstm_loss_function  # Loss function for training
        self.optimizer_function = arguments.lstm_optimizer_function  # Optimizer function
        self.recurrent_activation = arguments.lstm_recurrent_activation  # Recurrent activation function
        self.intermediary_layer_activation = arguments.lstm_intermediary_layer_activation  # Activation function for intermediary layers
        self.input_dimension = arguments.lstm_input_dimension  # Input data shape
        self.number_classes = arguments.number_classes  # Number of output classes for classification
        self.dropout_rate = arguments.lstm_dropout_rate  # Dropout rate for regularization
        self.last_layer_activation = arguments.lstm_last_layer_activation  # Activation for the output layer
        self.model_name = "LSTM"  # Model name

    def build_model(self) -> None:
        """
        Build the LSTM model architecture using Keras.

        The model consists of the following components:
            - Multiple LSTM layers to capture temporal dependencies.
            - Dropout layers for regularization.
            - A dense layer with softmax (or other) activation for classification.

        The model is designed for audio classification tasks where sequential dependencies need to be captured.
        """

        # Input layer
        inputs = Input(shape=self.input_dimension)

        neural_network_flow = inputs

        # Apply LSTM layers with specified cells and activation functions
        for _, cells in enumerate(self.list_lstm_cells):
            neural_network_flow = LSTM(cells, activation=self.intermediary_layer_activation,
                                       recurrent_activation=self.recurrent_activation,
                                       return_sequences=True)(neural_network_flow)
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        # Global average pooling after LSTM layers
        neural_network_flow = GlobalAveragePooling1D()(neural_network_flow)

        # Final dense layer for classification
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)

        # Create the model
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow)
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

        # Compile the model with the specified loss function and optimizer
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data,
                                                         callbacks=callbacks if callbacks else None
                                                         )
        return training_history


    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_lstm_cells(self):
        return self._list_lstm_cells

    @list_lstm_cells.setter
    def list_lstm_cells(self, value):
        self._list_lstm_cells = value

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
    def recurrent_activation(self):
        return self._recurrent_activation

    @recurrent_activation.setter
    def recurrent_activation(self, value):
        self._recurrent_activation = value

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