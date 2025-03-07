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
    import logging
    import tensorflow

    from tqdm import tqdm

    from sklearn.utils import resample

    from tensorflow.keras import Model

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import GlobalAveragePooling1D

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    from Modules.Layers.ConformerBlock import ConformerBlock
    from Modules.Evaluation.MetricsCalculator import MetricsCalculator
    from Modules.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

DEFAULT_INPUT_DIMENSION = (80, 40)
DEFAULT_NUMBER_CONFORMER_BLOCKS = 4
DEFAULT_EMBEDDING_DIMENSION = 64
DEFAULT_NUMBER_HEADS = 4
DEFAULT_MAX_LENGTH = 100
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT_DECAY = 0.2
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_SIZE_KERNEL = 3
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_SIZE_BATCH = 32
DEFAULT_OVERLAP = 2
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 80
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_OPTIMIZER_FUNCTION = 'adam'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'


class TransposeLayer(Layer):
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.channels_permutation = perm

    def call(self, inputs):
        return tensorflow.transpose(inputs, perm=self.channels_permutation)

    def compute_output_shape(self, input_shape):
        return [input_shape[dim] for dim in self.channels_permutation]

    def get_config(self):
        config = super(TransposeLayer, self).get_config()
        config.update({
            'perm': self.channels_permutation
        })
        return config


class Conformer(MetricsCalculator):

    def __init__(self,
                 number_conformer_blocks: int = DEFAULT_NUMBER_CONFORMER_BLOCKS,
                 embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
                 number_heads: int = DEFAULT_NUMBER_HEADS,
                 size_kernel: tuple = DEFAULT_KERNEL_SIZE,
                 number_classes: int = DEFAULT_NUMBER_CLASSES,
                 last_layer_activation: str = DEFAULT_LAST_LAYER_ACTIVATION,
                 size_batch: int = DEFAULT_SIZE_BATCH,
                 number_splits: int = DEFAULT_NUMBER_SPLITS,
                 number_epochs: int = DEFAULT_NUMBER_EPOCHS,
                 loss_function: str = DEFAULT_LOSS_FUNCTION,
                 optimizer_function: str = DEFAULT_OPTIMIZER_FUNCTION,
                 window_size_factor: int = DEFAULT_WINDOW_SIZE_FACTOR,
                 decibel_scale_factor: int = DEFAULT_DECIBEL_SCALE_FACTOR,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 window_size_fft: int = DEFAULT_WINDOW_SIZE,
                 number_filters_spectrogram: int = DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
                 overlap: int = DEFAULT_OVERLAP,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 dropout_rate: float = DEFAULT_DROPOUT_RATE,
                 file_extension: str = DEFAULT_FILE_EXTENSION,
                 input_dimension: tuple = DEFAULT_INPUT_DIMENSION):

        self.neural_network_model = None
        self.size_batch = size_batch
        self.number_splits = number_splits
        self.number_epochs = number_epochs
        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.window_size_factor = window_size_factor
        self.decibel_scale_factor = decibel_scale_factor
        self.hop_length = hop_length
        self.window_size_fft = window_size_fft
        self.number_filters_spectrogram = number_filters_spectrogram
        self.overlap = overlap
        self.window_size = hop_length * (self.window_size_factor - 1)
        self.sample_rate = sample_rate
        self.file_extension = file_extension
        self.input_dimension = input_dimension
        self.number_conformer_blocks = number_conformer_blocks
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.number_classes = number_classes
        self.kernel_size = size_kernel
        self.dropout_rate = dropout_rate
        self.model_name = "Conformer"
        self.last_layer_activation = last_layer_activation

    def build_model(self) -> None:

        inputs = Input(shape=self.input_dimension)
        neural_network_flow = ConvolutionalSubsampling()(inputs)
        neural_network_flow = TransposeLayer(perm=[0, 2, 1])(neural_network_flow)
        neural_network_flow = Dense(self.embedding_dimension)(neural_network_flow)

        for _ in range(self.number_conformer_blocks):
            neural_network_flow = ConformerBlock(self.embedding_dimension,
                                                 self.number_heads,
                                                 self.input_dimension[0] // 2,
                                                 self.kernel_size,
                                                 self.dropout_rate)(neural_network_flow)

        neural_network_flow = GlobalAveragePooling1D()(neural_network_flow)
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)
        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor, epochs: int,
                          batch_size: int, validation_data: tuple = None) -> tensorflow.keras.callbacks.History:

        # Compile the model with the specified optimizer, loss function, and metrics
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

        # Train the model with the training data and labels, and optionally validation data
        training_history = self.neural_network_model.fit(train_data, train_labels, epochs=epochs,
                                                         batch_size=batch_size,
                                                         validation_data=validation_data)
        return training_history


    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:

        logging.info("Starting to load data...")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        # Check if directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Collect all class directories
        logging.info(f"Reading subdirectories in '{sub_directories}'...")
        for class_dir in os.listdir(sub_directories):
            class_path = os.path.join(sub_directories, class_dir)
            if os.path.isdir(class_path):
                list_class_path.append(class_path)

        logging.info(f"Found {len(list_class_path)} class directories.")

        # Process each audio file in subdirectories
        for sub_directory in list_class_path:
            logging.info(f"Processing class directory: {sub_directory}...")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                try:
                    signal, _ = librosa.load(file_name, sr=self.sample_rate)
                    label = file_name.split('/')[-2].split('_')[0]

                    for (start, end) in self.windows(signal, self.window_size, self.overlap):
                        if len(signal[start:end]) == self.window_size:
                            signal_window = signal[start:end]

                            # Generate mel spectrogram
                            spectrogram = librosa.feature.melspectrogram(
                                y=signal_window,
                                n_mels=self.number_filters_spectrogram,
                                sr=self.sample_rate,
                                n_fft=self.window_size_fft,
                                hop_length=self.hop_length
                            )

                            # Convert spectrogram to decibels
                            spectrogram_decibel_scale = librosa.power_to_db(spectrogram, ref=numpy.max)
                            spectrogram_decibel_scale = (spectrogram_decibel_scale / self.decibel_scale_factor) + 1

                            # Append spectrogram and label
                            list_spectrogram.append(spectrogram_decibel_scale)
                            list_labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        # Reshape the feature array to the expected dimensions
        array_features = numpy.array(list_spectrogram).reshape(
            len(list_spectrogram),
            self.number_filters_spectrogram,
            self.window_size // self.hop_length,  # Time frames depend on hop length
            1
        )

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info(f"Loaded {len(array_features)} spectrogram features.")
        logging.info("Data loading complete.")

        return numpy.array(array_features, dtype=numpy.float32), array_labels


    def compile_model(self) -> None:
        """
        Compiles the CNN model with the specified loss function and optimizer.

        This method prepares the model for training by setting the optimizer, loss function, and metrics.
        """
        self.neural_network_model.compile(optimizer=self.optimizer_function, loss=self.loss_function,
                                          metrics=['accuracy'])

    def train(self, dataset_directory, number_epochs, batch_size, number_splits,
              loss, sample_rate, overlap, number_classes, arguments) -> tuple:
        """
        Trains the model using cross-validation.

        Parameters
        ----------
        dataset_directory : str
            Directory containing the training data.
        number_epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size for training.
        number_splits : int, optional
            Number of splits for cross-validation.

        Returns
        -------
        tuple
            A tuple containing the mean metrics, the training history, the mean confusion matrix,
            and the predicted probabilities along with the ground truth labels.
        """
        # Use default values if not provided
        self.number_epochs = number_epochs or self.number_epochs
        self.number_splits = number_splits or self.number_splits
        self.size_batch = batch_size or self.size_batch
        self.loss_function = loss or self.loss_function
        self.sample_rate = sample_rate or self.sample_rate
        self.overlap = overlap or self.overlap
        self.number_classes = number_classes or self.number_classes

        self.window_size_factor = arguments.conformer_window_size_factor
        self.decibel_scale_factor = arguments.conformer_decibel_scale_factor
        self.hop_length = arguments.conformer_hop_length
        self.number_filters_spectrogram = arguments.conformer_number_filters_spectrogram
        self.overlap = arguments.conformer_overlap
        self.window_size = self.hop_length * (self.window_size_factor - 1)
        self.number_conformer_blocks = arguments.conformer_number_conformer_blocks
        self.embedding_dimension = arguments.conformer_embedding_dimension
        self.number_heads = arguments.conformer_number_heads
        self.kernel_size = arguments.conformer_size_kernel
        self.dropout_rate = arguments.conformer_dropout_rate

        history_model = None
        features, labels = self.load_data(dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data into train/val and test sets
        features_train_val, features_test, labels_train_val, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance training/validation set
        features_train_val, labels_train_val = balance_classes(features_train_val, labels_train_val)

        # Stratified k-fold cross-validation on the training/validation set
        instance_k_fold = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
        probabilities_list = []
        real_labels_list = []

        for train_indexes, val_indexes in instance_k_fold.split(features_train_val, labels_train_val):
            features_train, features_val = features_train_val[train_indexes], features_train_val[val_indexes]
            labels_train, labels_val = labels_train_val[train_indexes], labels_train_val[val_indexes]

            # Balance the training set for this fold
            features_train, labels_train = balance_classes(features_train, labels_train)

            self.build_model()
            self.neural_network_model.summary()

            history_model = self.compile_and_train(features_train, labels_train, epochs=self.number_epochs,
                                                   batch_size=self.size_batch,
                                                   validation_data=(features_val, labels_val))

            model_predictions = self.neural_network_model.predict(features_val)
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_val)

            # Calculate and store the metrics for this fold
            metrics, confusion_matrix = self.calculate_metrics(predicted_labels, labels_val, predicted_labels)
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)

        # Calculate mean metrics across all folds
        mean_metrics = {
            'model_name': self.model_name,
            'Acc.': {'value': numpy.mean([metric['Accuracy'] for metric in metrics_list]),
                     'std': numpy.std([metric['Accuracy'] for metric in metrics_list])},
            'Prec.': {'value': numpy.mean([metric['Precision'] for metric in metrics_list]),
                      'std': numpy.std([metric['Precision'] for metric in metrics_list])},
            'Rec.': {'value': numpy.mean([metric['Recall'] for metric in metrics_list]),
                     'std': numpy.std([metric['Recall'] for metric in metrics_list])},
            'F1.': {'value': numpy.mean([metric['F1-Score'] for metric in metrics_list]),
                    'std': numpy.std([metric['F1-Score'] for metric in metrics_list])},
        }

        probabilities_predicted = {
            'model_name': self.model_name,
            'predicted': numpy.concatenate(probabilities_list),
            'ground_truth': numpy.concatenate(real_labels_list)
        }

        confusion_matrix_array = numpy.array(confusion_matriz_list)
        mean_confusion_matrix = numpy.mean(confusion_matrix_array, axis=0)
        mean_confusion_matrix = numpy.round(mean_confusion_matrix).astype(numpy.int32).tolist()

        mean_confusion_matrices = {
            "confusion_matrix": mean_confusion_matrix,
            "class_names": ['Class {}'.format(i) for i in range(self.number_classes)],
            "title": self.model_name
        }

        return (mean_metrics, {"Name": self.model_name, "History": history_model.history}, mean_confusion_matrices,
                probabilities_predicted)

