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

    from Engine.Models.Conformer import Conformer

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

class TestConformerModel(unittest.TestCase):
    """
    Unit test class for the Conformer model.

    The tests validate the behavior of the Conformer model:
    - Initialization with correct parameters
    - Model building
    - Compilation and training
    - Error handling for invalid inputs and configurations
    """

    def setUp(self):
        """
        Set up test parameters for the Conformer model.
        This method will be called before every test.
        """
        self.test_params = {
            'number_conformer_blocks': 2,
            'embedding_dimension': 512,
            'number_heads': 8,
            'size_kernel': (3, 3),
            'number_classes': 10,
            'last_layer_activation': 'softmax',
            'loss_function': 'categorical_crossentropy',
            'optimizer_function': 'adam',
            'number_filters_spectrogram': 64,
            'dropout_rate': 0.2,
            'input_dimension': (128, 80)
        }
        self.model = Conformer(**self.test_params)

    def test_model_initialization(self):
        """
        Test that the Conformer model initializes correctly.
        """
        self.assertEqual(self.model.number_conformer_blocks, self.test_params['number_conformer_blocks'])
        self.assertEqual(self.model.embedding_dimension, self.test_params['embedding_dimension'])
        self.assertEqual(self.model.number_heads, self.test_params['number_heads'])
        self.assertEqual(self.model.size_kernel, self.test_params['size_kernel'])
        self.assertEqual(self.model.number_classes, self.test_params['number_classes'])
        self.assertEqual(self.model.last_layer_activation, self.test_params['last_layer_activation'])
        self.assertEqual(self.model.loss_function, self.test_params['loss_function'])
        self.assertEqual(self.model.optimizer_function, self.test_params['optimizer_function'])
        self.assertEqual(self.model.number_filters_spectrogram, self.test_params['number_filters_spectrogram'])
        self.assertEqual(self.model.dropout_rate, self.test_params['dropout_rate'])
        self.assertEqual(self.model.input_dimension, self.test_params['input_dimension'])

    def test_model_building(self):
        """
        Test that the Conformer model can be built without errors.
        """
        try:
            self.model.build_model()
        except Exception as e:
            self.fail(f"Model build failed with exception: {e}")

    def test_model_compilation(self):
        """
        Test that the Conformer model compiles without errors.
        """
        self.model.build_model()
        try:
            self.model.compile_model()
        except Exception as e:
            self.fail(f"Model compilation failed with exception: {e}")

    def test_compile_and_train(self):
        """
        Test that the model can be compiled and trained with dummy data.
        """
        # Create dummy data
        X_train = numpy.random.rand(10, 128, 80).astype(numpy.float32)
        y_train = numpy.random.randint(0, 10, size=(10, 1)).astype(numpy.int32)
        X_val = numpy.random.rand(5, 128, 80).astype(numpy.float32)
        y_val = numpy.random.randint(0, 10, size=(5, 1)).astype(numpy.int32)

        self.model.build_model()
        self.model.compile_model()

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
        invalid_params['input_dimension'] = (128,)  # Invalid input shape

        with self.assertRaises(ValueError):
            model = Conformer(**invalid_params)
            model.build_model()

    def test_invalid_number_conformer_blocks(self):
        """
        Test that the model raises an exception if the number of Conformer blocks is invalid (non-positive).
        """
        invalid_params = self.test_params.copy()
        invalid_params['number_conformer_blocks'] = -1  # Invalid number of blocks

        with self.assertRaises(ValueError):
            model = Conformer(**invalid_params)
            model.build_model()

    def test_invalid_loss_function(self):
        """
        Test that the model raises an exception if an invalid loss function is provided.
        """
        invalid_params = self.test_params.copy()
        invalid_params['loss_function'] = 'invalid_loss_function'  # Invalid loss function

        with self.assertRaises(ValueError):
            model = Conformer(**invalid_params)
            model.build_model()

    def tearDown(self):
        """
        Clean up after each test method to free resources.
        This method will be called after every test.
        """
        backend.clear_session()


if __name__ == '__main__':
    unittest.main()