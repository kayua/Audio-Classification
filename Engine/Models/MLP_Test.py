import unittest
import tensorflow as tf

from Engine.Models.MLP import DenseModel


class TestDenseModel(unittest.TestCase):
    """
    Test class for the DenseModel class. This class is used to test the correct
    implementation and functionality of the DenseModel, ensuring it behaves
    as expected during initialization, model building, and training.
    """

    def setUp(self):
        """
        Initializes the DenseModel instance with necessary parameters
        before each test case.
        """
        self.model = DenseModel(
            number_classes=10,  # Number of output classes
            last_layer_activation='softmax',  # Activation function for output layer
            loss_function='categorical_crossentropy',  # Loss function
            optimizer_function='adam',  # Optimizer function
            dropout_rate=0.2,  # Dropout rate
            intermediary_layer_activation='relu',  # Activation function for hidden layers
            input_dimension=(28, 28, 1)  # Input shape for images (e.g., MNIST dataset)
        )

    def test_model_initialization(self):
        """
        Test the initialization of the DenseModel class. This checks if the model
        has been initialized with the correct parameters.
        """
        self.assertEqual(self.model.number_classes, 10, "The number of output classes should be 10.")
        self.assertEqual(self.model.loss_function, 'categorical_crossentropy',
                         "The loss function should be 'categorical_crossentropy'.")
        self.assertEqual(self.model.optimizer_function, 'adam', "The optimizer should be 'adam'.")
        self.assertEqual(self.model.dropout_rate, 0.2, "The dropout rate should be 0.2.")
        self.assertEqual(self.model.input_dimension, (28, 28, 1), "The input dimension should be (28, 28, 1).")
        self.assertEqual(self.model.list_number_neurons, [128, 128],
                         "The number of neurons in hidden layers should be the default list.")
        self.assertIsNone(self.model.neural_network_model, "The model should not be built yet.")

    def test_build_model(self):
        """
        Test the model building process. This ensures that the model is properly
        built without errors when calling the `build_model` method.
        """
        self.model.build_model()
        self.assertIsNotNone(self.model.neural_network_model, "The model should be built.")
        # The model should have at least one Dense layer and one Dropout layer
        self.assertGreater(len(self.model.neural_network_model.layers), 0, "The model should have layers.")
        self.assertTrue(isinstance(self.model.neural_network_model, tf.keras.Model), "The model should be a Keras Model.")

    def test_compile_and_train(self):
        """
        Test the model compilation and training process. This ensures that the model can be compiled
        and trained without errors using the `compile_and_train` method.
        """
        self.model.build_model()

        # Generate fake training data for testing purposes
        X_train = tf.random.normal((100, 28, 28, 1))  # 100 samples with input shape (28, 28, 1)
        y_train = tf.random.uniform((100,), minval=0, maxval=10, dtype=tf.int32)  # 100 labels for 10 classes

        # Train the model and verify the result
        try:
            history = self.model.compile_and_train(X_train, y_train, epochs=1, batch_size=32)
            self.assertIsInstance(history, tf.keras.callbacks.History,
                                  "The training history should be returned as an instance"
                                  " of tf.keras.callbacks.History.")
            self.assertTrue('accuracy' in history.history, "The training history should contain 'accuracy'.")
        except Exception as e:
            self.fail(f"Model training failed with exception: {str(e)}")

    def test_invalid_loss_function(self):
        """
        Test that an invalid loss function raises an error during initialization.
        """
        with self.assertRaises(ValueError, msg="An invalid loss function should raise a ValueError."):
            model = DenseModel(
                number_classes=10,
                last_layer_activation='softmax',
                loss_function='invalid_loss',  # Invalid loss function
                optimizer_function='adam',
                dropout_rate=0.2,
                intermediary_layer_activation='relu',
                input_dimension=(28, 28, 1)
            )

    def test_invalid_optimizer_function(self):
        """
        Test that an invalid optimizer function raises an error during initialization.
        """
        with self.assertRaises(ValueError, msg="An invalid optimizer function should raise a ValueError."):
            model = DenseModel(
                number_classes=10,
                last_layer_activation='softmax',
                loss_function='categorical_crossentropy',
                optimizer_function='invalid_optimizer',  # Invalid optimizer
                dropout_rate=0.2,
                intermediary_layer_activation='relu',
                input_dimension=(28, 28, 1)
            )

if __name__ == '__main__':
    unittest.main()