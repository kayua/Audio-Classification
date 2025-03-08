import unittest
import tensorflow
import numpy

from Engine.Models.ResidualModel import ResidualModel


class TestResidualModel(unittest.TestCase):

    def setUp(self):
        """
        Setup method to configure test parameters. Initializes model configuration
        and generates random training and testing data.
        """
        self.input_dimension = (32, 32, 3)  # Example of RGB image 32x32
        self.convolutional_padding = 'same'
        self.intermediary_activation = 'relu'
        self.last_layer_activation = 'softmax'
        self.number_classes = 10  # Example with 10 classes (CIFAR-10)
        self.size_convolutional_filters = (3, 3)
        self.size_pooling = (2, 2)
        self.filters_per_block = [32, 64, 128]
        self.loss_function = 'categorical_crossentropy'
        self.optimizer_function = 'adam'
        self.dropout_rate = 0.2

        # Generate random data for training (e.g., images of shape (32, 32, 3))
        self.X_train = numpy.random.rand(100, 32, 32, 3)  # 100 random images for training
        self.y_train = numpy.random.randint(0, self.number_classes, 100)  # 100 random labels
        self.y_train = tensorflow.keras.utils.to_categorical(self.y_train, self.number_classes)

        # Generate random data for validation (optional)
        self.X_val = numpy.random.rand(20, 32, 32, 3)  # 20 random images for validation
        self.y_val = numpy.random.randint(0, self.number_classes, 20)  # 20 random labels
        self.y_val = tensorflow.keras.utils.to_categorical(self.y_val, self.number_classes)

    def test_initialization(self):
        """
        Test the initialization of the ResidualModel class.
        Verifies if the parameters are set correctly.
        """
        model = ResidualModel(
            input_dimension=self.input_dimension,
            convolutional_padding=self.convolutional_padding,
            intermediary_activation=self.intermediary_activation,
            last_layer_activation=self.last_layer_activation,
            number_classes=self.number_classes,
            size_convolutional_filters=self.size_convolutional_filters,
            size_pooling=self.size_pooling,
            filters_per_block=self.filters_per_block,
            loss_function=self.loss_function,
            optimizer_function=self.optimizer_function,
            dropout_rate=self.dropout_rate
        )

        # Assert the model has been initialized properly
        self.assertEqual(model.input_shape, self.input_dimension)
        self.assertEqual(model.number_classes, self.number_classes)
        self.assertEqual(model.loss_function, self.loss_function)
        self.assertEqual(model.optimizer_function, self.optimizer_function)

    def test_model_building(self):
        """
        Test if the model can be built without errors.
        Verifies if the build_model method works correctly.
        """
        model = ResidualModel(
            input_dimension=self.input_dimension,
            convolutional_padding=self.convolutional_padding,
            intermediary_activation=self.intermediary_activation,
            last_layer_activation=self.last_layer_activation,
            number_classes=self.number_classes,
            size_convolutional_filters=self.size_convolutional_filters,
            size_pooling=self.size_pooling,
            filters_per_block=self.filters_per_block,
            loss_function=self.loss_function,
            optimizer_function=self.optimizer_function,
            dropout_rate=self.dropout_rate
        )

        # Build the model
        model.build_model()

        # Assert the model has been built correctly
        self.assertIsNotNone(model.neural_network_model)
        self.assertEqual(len(model.filters_per_block), len(self.filters_per_block))

    def test_model_training(self):
        """
        Test if the model can be compiled and trained on random data without errors.
        Verifies if the compile_and_train method works correctly.
        """
        model = ResidualModel(
            input_dimension=self.input_dimension,
            convolutional_padding=self.convolutional_padding,
            intermediary_activation=self.intermediary_activation,
            last_layer_activation=self.last_layer_activation,
            number_classes=self.number_classes,
            size_convolutional_filters=self.size_convolutional_filters,
            size_pooling=self.size_pooling,
            filters_per_block=self.filters_per_block,
            loss_function=self.loss_function,
            optimizer_function=self.optimizer_function,
            dropout_rate=self.dropout_rate
        )

        # Build the model
        model.build_model()

        # Compile and train the model
        history = model.compile_and_train(
            train_data=self.X_train,
            train_labels=self.y_train,
            epochs=3,  # Running for just a few epochs for testing purposes
            batch_size=16,
            validation_data=(self.X_val, self.y_val)
        )

        # Assert that the model has trained and returned a history object
        self.assertIsInstance(history, tensorflow.keras.callbacks.History)
        self.assertTrue(len(history.history['loss']) > 0)
        self.assertTrue(len(history.history['accuracy']) > 0)


if __name__ == '__main__':
    unittest.main()