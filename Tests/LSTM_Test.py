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

    from Engine.Models.LSTM import AudioLSTM

except ImportError as error:
    print(error)
    sys.exit(-1)

class TestAudioLSTM(unittest.TestCase):
    """
    Test class for the AudioLSTM model. This class is used to test the correct
    implementation and functionality of the AudioLSTM class, ensuring it behaves
    as expected during initialization, model building, and training.

    Example usage:
        >>> test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioLSTM)
        >>> unittest.TextTestRunner().run(test_suite)
    """

    def setUp(self):
        """
        Initializes the AudioLSTM model instance with the necessary parameters
        before each test.

        This method is called before each test case to ensure that the model
        is properly set up.
        """
        self.model = AudioLSTM(
            number_classes=10,  # Number of output classes
            last_layer_activation='softmax',  # Activation function for the last layer
            loss_function='categorical_crossentropy',  # Loss function used during training
            optimizer_function='adam',  # Optimizer function
            dropout_rate=0.2,  # Dropout rate for regularization
            intermediary_layer_activation='tanh',  # Activation function for intermediary layers
            recurrent_activation='sigmoid',  # Recurrent activation function for LSTM
            input_dimension=(128, 80),  # Input data shape (e.g., Mel spectrogram)
            list_lstm_cells=[64, 128]  # LSTM cell sizes for each LSTM layer
        )

    def test_model_initialization(self):
        """
        Test the initialization of the AudioLSTM model. This test checks if the
        model parameters are correctly assigned during initialization.

        Expected behavior:
            - The number of output classes should be 10.
            - The loss function should be 'categorical_crossentropy'.
            - The optimizer should be 'adam'.
            - The dropout rate should be 0.2.
            - The input dimension should be (128, 80).
            - The LSTM layers should be initialized with [64, 128] cells.
        """
        self.assertEqual(self.model.number_classes, 10, "The number of output classes should be 10.")
        self.assertEqual(self.model.loss_function, 'categorical_crossentropy',
                         "The loss function should be 'categorical_crossentropy'.")
        self.assertEqual(self.model.optimizer_function, 'adam', "The optimizer should be 'adam'.")
        self.assertEqual(self.model.dropout_rate, 0.2, "The dropout rate should be 0.2.")
        self.assertEqual(self.model.input_dimension, (128, 80), "The input dimension should be (128, 80).")
        self.assertEqual(self.model.list_lstm_cells, [64, 128], "The LSTM cell sizes should be [64, 128].")
        self.assertIsNone(self.model.neural_network_model, "The neural network model should be None before being built.")

    def test_build_model(self):
        """
        Test the model building process. This test ensures that the model
        architecture is correctly constructed without errors when calling the
        `build_model` method.

        Expected behavior:
            - The model should be successfully built.
            - The number of layers in the model should be correct (LSTM layers + dropout + pooling + dense).
        """
        self.model.build_model()
        self.assertIsNotNone(self.model.neural_network_model, "The neural network model should not be None after building.")
        # The model should consist of 6 layers: LSTM layers (2), Dropout (2), Pooling, and Dense
        self.assertEqual(len(self.model.neural_network_model.layers), 6, "The number of layers in the model should be 6.")

    def test_compile_and_train(self):
        """
        Test the compilation and training process of the AudioLSTM model. This test
        ensures that the model can be compiled and trained without errors and that
        the returned training history contains the expected accuracy metric.

        Expected behavior:
            - The model should be compiled without errors.
            - The training history should be returned.
            - The history object should contain accuracy as one of the metrics.
        """
        self.model.build_model()

        # Create fake training data for testing purposes
        X_train = tensorflow.random.normal((100, 128, 80))  # 100 samples with input shape (128, 80)
        y_train = tensorflow.random.uniform((100,), minval=0, maxval=10, dtype=tf.int32)  # 100 labels for 10 classes

        # Train the model
        try:
            history = self.model.compile_and_train(X_train, y_train, epochs=1, batch_size=32)
            self.assertIsInstance(history, tensorflow.keras.callbacks.History,
                                  "The returned history should be an instance of tf.keras.callbacks.History.")
            self.assertTrue('accuracy' in history.history, "The training history should contain 'accuracy' as a metric.")
        except Exception as e:
            self.fail(f"Model training failed with exception: {str(e)}")

    def test_invalid_loss_function(self):
        """
        Test that an invalid loss function raises an error during initialization.

        Expected behavior:
            - The model should raise a ValueError when an invalid loss function is provided.
        """
        with self.assertRaises(ValueError, msg="An invalid loss function should raise a ValueError."):
            model = AudioLSTM(
                number_classes=10,
                last_layer_activation='softmax',
                loss_function='invalid_loss',  # Invalid loss function
                optimizer_function='adam',
                dropout_rate=0.2,
                intermediary_layer_activation='tanh',
                recurrent_activation='sigmoid',
                input_dimension=(128, 80),
                list_lstm_cells=[64, 128]
            )

    def test_invalid_optimizer_function(self):
        """
        Test that an invalid optimizer function raises an error during initialization.

        Expected behavior:
            - The model should raise a ValueError when an invalid optimizer function is provided.
        """
        with self.assertRaises(ValueError, msg="An invalid optimizer function should raise a ValueError."):
            model = AudioLSTM(
                number_classes=10,
                last_layer_activation='softmax',
                loss_function='categorical_crossentropy',
                optimizer_function='invalid_optimizer',  # Invalid optimizer
                dropout_rate=0.2,
                intermediary_layer_activation='tanh',
                recurrent_activation='sigmoid',
                input_dimension=(128, 80),
                list_lstm_cells=[64, 128]
            )

if __name__ == '__main__':
    unittest.main()