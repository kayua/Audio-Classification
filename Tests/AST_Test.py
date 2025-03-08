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

    import numpy

    import unittest
    import tensorflow

    from tensorflow.keras import backend

    from Engine.Models.AST import AudioSpectrogramTransformer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

class TestAudioSpectrogramTransformerModel(unittest.TestCase):
    """
    Unit test class for the AudioSpectrogramTransformer model.

    The tests validate the behavior of the AudioSpectrogramTransformer model:
    - Initialization with correct parameters
    - Model building
    - Compilation and training
    - Error handling for invalid inputs and configurations
    """

    def setUp(self):
        """
        Set up test parameters for the AudioSpectrogramTransformer model.
        This method will be called before every test.
        """
        self.test_params = {
            'projection_dimension': 512,
            'head_size': 64,
            'num_heads': 8,
            'number_blocks': 12,
            'number_classes': 10,
            'patch_size': (32, 32),
            'dropout': 0.1,
            'intermediary_activation': 'relu',
            'loss_function': 'categorical_crossentropy',
            'last_activation_layer': 'softmax',
            'optimizer_function': 'adam',
            'normalization_epsilon': 1e-6,
            'number_filters_spectrogram': 64
        }
        self.model = AudioSpectrogramTransformer(**self.test_params)

    def test_model_initialization(self):
        """
        Test that the AudioSpectrogramTransformer model initializes correctly.
        """
        self.assertEqual(self.model.head_size, self.test_params['head_size'])
        self.assertEqual(self.model.number_heads, self.test_params['num_heads'])
        self.assertEqual(self.model.number_blocks, self.test_params['number_blocks'])
        self.assertEqual(self.model.number_classes, self.test_params['number_classes'])
        self.assertEqual(self.model.patch_size, self.test_params['patch_size'])
        self.assertEqual(self.model.dropout, self.test_params['dropout'])
        self.assertEqual(self.model.optimizer_function, self.test_params['optimizer_function'])
        self.assertEqual(self.model.loss_function, self.test_params['loss_function'])
        self.assertEqual(self.model.normalization_epsilon, self.test_params['normalization_epsilon'])
        self.assertEqual(self.model.last_activation_layer, self.test_params['last_activation_layer'])
        self.assertEqual(self.model.projection_dimension, self.test_params['projection_dimension'])
        self.assertEqual(self.model.intermediary_activation, self.test_params['intermediary_activation'])
        self.assertEqual(self.model.number_filters_spectrogram, self.test_params['number_filters_spectrogram'])

    def test_model_building(self):
        """
        Test that the AudioSpectrogramTransformer model can be built without errors.
        """
        try:
            self.model.build_model(number_patches=10)
        except Exception as e:
            self.fail(f"Model build failed with exception: {e}")

    def test_model_compilation(self):
        """
        Test that the AudioSpectrogramTransformer model compiles without errors.
        """
        self.model.build_model(number_patches=10)
        try:
            self.model.neural_network_model.compile(optimizer=self.test_params['optimizer_function'],
                                                    loss=self.test_params['loss_function'],
                                                    metrics=['accuracy'])
        except Exception as e:
            self.fail(f"Model compilation failed with exception: {e}")

    def test_compile_and_train(self):
        """
        Test that the AudioSpectrogramTransformer model can be compiled and trained with dummy data.
        """
        # Create dummy data
        X_train = numpy.random.rand(10, 10, 32, 32).astype(numpy.float32)  # 10 samples, 10 patches, 32x32 spectrograms
        y_train = numpy.random.randint(0, 10, size=(10, 1)).astype(numpy.int32)
        X_val = numpy.random.rand(5, 10, 32, 32).astype(numpy.float32)
        y_val = numpy.random.randint(0, 10, size=(5, 1)).astype(numpy.int32)

        self.model.build_model(number_patches=10)
        self.model.neural_network_model.compile(optimizer=self.test_params['optimizer_function'],
                                                loss=self.test_params['loss_function'],
                                                metrics=['accuracy'])

        try:
            history = self.model.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=2,
                batch_size=2,
                validation_data=(X_val, y_val)
            )
            self.assertIsInstance(history, tensorflow.keras.callbacks.History)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_invalid_input_dimension(self):
        """
        Test that the model raises an exception if the input dimension is invalid.
        """
        invalid_params = self.test_params.copy()
        invalid_params['patch_size'] = (64,)  # Invalid patch size (must be a tuple with two elements)

        with self.assertRaises(ValueError):
            model = AudioSpectrogramTransformer(**invalid_params)
            model.build_model(number_patches=10)

    def test_invalid_loss_function(self):
        """
        Test that the model raises an exception if an invalid loss function is provided.
        """
        invalid_params = self.test_params.copy()
        invalid_params['loss_function'] = 'invalid_loss_function'  # Invalid loss function

        with self.assertRaises(ValueError):
            model = AudioSpectrogramTransformer(**invalid_params)
            model.build_model(number_patches=10)

    def test_invalid_number_classes(self):
        """
        Test that the model raises an exception if the number of classes is invalid.
        """
        invalid_params = self.test_params.copy()
        invalid_params['number_classes'] = -1  # Invalid number of classes (cannot be negative)

        with self.assertRaises(ValueError):
            model = AudioSpectrogramTransformer(**invalid_params)
            model.build_model(number_patches=10)

    def tearDown(self):
        """
        Clean up after each test method to free resources.
        This method will be called after every test.
        """
        backend.clear_session()


if __name__ == '__main__':
    unittest.main()