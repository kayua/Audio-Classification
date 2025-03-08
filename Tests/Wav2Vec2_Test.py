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

    import unittest
    import tensorflow

    from tensorflow.keras import layers
    from unittest.mock import MagicMock

    from Engine.Models.Wav2Vec2 import AudioWav2Vec2

except ImportError as error:
    print(error)
    sys.exit(-1)

class TestAudioWav2Vec2(unittest.TestCase):
    """
    Unit tests for the AudioWav2Vec2 class.
    """

    def setUp(self):
        """
        This method is called before each test.
        It initializes the AudioWav2Vec2 class with example parameters.
        """
        self.model = AudioWav2Vec2(
            number_classes=10,
            last_layer_activation='softmax',
            loss_function='categorical_crossentropy',
            optimizer_function='adam',
            quantization_units=4,
            key_dimension=64,
            dropout_rate=0.2,
            intermediary_layer_activation='relu',
            input_dimension=(128, 80),
            number_heads=8,
            kernel_size=3,
            list_filters_encoder=[64, 128, 256]
        )

    def test_initialization(self):
        """
        Tests the initialization of the AudioWav2Vec2 class to verify that parameters are correctly assigned.
        """
        self.assertEqual(self.model.number_classes, 10)
        self.assertEqual(self.model.last_layer_activation, 'softmax')
        self.assertEqual(self.model.loss_function, 'categorical_crossentropy')
        self.assertEqual(self.model.optimizer_function, 'adam')
        self.assertEqual(self.model.quantization_units, 4)
        self.assertEqual(self.model.key_dimension, 64)
        self.assertEqual(self.model.dropout_rate, 0.2)
        self.assertEqual(self.model.intermediary_layer_activation, 'relu')
        self.assertEqual(self.model.input_dimension, (128, 80))
        self.assertEqual(self.model.number_heads, 8)
        self.assertEqual(self.model.kernel_size, 3)
        self.assertEqual(self.model.list_filters_encoder, [64, 128, 256])

    def test_build_model(self):
        """
        Tests the build_model method to ensure that the model is built correctly without errors.
        """
        self.model.build_model()
        self.assertIsNotNone(self.model.neural_network_model)
        self.assertIsInstance(self.model.neural_network_model, tensorflow.keras.Model)

    def test_compile_and_train(self):
        """
        Tests the compile_and_train method to check if training occurs without errors.
        For simulation, we will use mock data.
        """
        # Creating mock data for testing
        X_train = tensorflow.random.normal((10, 128, 80))  # Example input data
        y_train = tensorflow.random.uniform((10,), maxval=10, dtype=tensorflow.int32)  # Example labels

        # Mocking some methods to test without actually training a model
        self.model.neural_network_model.fit = MagicMock()

        # Calling the method
        self.model.compile_and_train(X_train, y_train, epochs=1, batch_size=2)

        # Checking if the fit method was called
        self.model.neural_network_model.fit.assert_called_once_with(X_train, y_train, epochs=1, batch_size=2)

    def test_compile_and_train_no_validation(self):
        """
        Tests the compile_and_train method with validation data.
        """
        # Creating mock data for testing
        X_train = tensorflow.random.normal((10, 128, 80))  # Example input data
        y_train = tensorflow.random.uniform((10,), maxval=10, dtype=tensorflow.int32)  # Example labels
        X_val = tensorflow.random.normal((5, 128, 80))  # Example validation data
        y_val = tensorflow.random.uniform((5,), maxval=10, dtype=tensorflow.int32)  # Example validation labels

        # Mocking the fit method for testing without actual execution
        self.model.neural_network_model.fit = MagicMock()

        # Calling the method with validation data
        self.model.compile_and_train(X_train, y_train, epochs=1, batch_size=2, validation_data=(X_val, y_val))

        # Checking if the fit method was called with validation data
        self.model.neural_network_model.fit.assert_called_once_with(X_train, y_train, epochs=1, batch_size=2,
                                                                     validation_data=(X_val, y_val))

    def test_build_model_layers(self):
        """
        Tests the model building process to ensure that the layers are correctly added and the model is structured as expected.
        """
        self.model.build_model()
        model = self.model.neural_network_model

        # Verifying if the expected layers are added
        self.assertIn('conv1d', [layer.name for layer in model.layers])  # Convolutional layer
        self.assertIn('multi_head_attention', [layer.name for layer in model.layers])  # Multi-head attention layer
        self.assertIn('dense', [layer.name for layer in model.layers])  # Dense layer
        self.assertIn('quantization', [layer.name for layer in model.layers])  # Quantization layer

    def test_invalid_optimizer(self):
        """
        Tests the initialization with an invalid optimizer to ensure that an appropriate error is raised.
        """
        with self.assertRaises(ValueError):
            self.model.optimizer_function = 'invalid_optimizer'
            self.model.build_model()

if __name__ == '__main__':
    unittest.main()