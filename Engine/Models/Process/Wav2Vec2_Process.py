"""
===================================================================================
WAV2VEC2 PROCESS - VERSÃO ATUALIZADA COM XAI CORRETO
===================================================================================
Atualizado para usar métodos XAI apropriados para Transformers
"""

try:
    import os
    import sys
    import glob
    import numpy
    import logging
    import librosa
    from tqdm import tqdm
    from Tools.Metrics import Metrics
    from tensorflow.keras import Model
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    from Engine.Processing.ClassBalance import ClassBalancer
    from Engine.Models.Process.Base_Process import BaseProcess
    from Engine.Processing.WindowGenerator import WindowGenerator
except ImportError as error:
    print(error)
    sys.exit(-1)


class Wav2Vec2Process(ClassBalancer, WindowGenerator, BaseProcess, Metrics):
    """
    Wav2Vec2 training process with CORRECTED XAI methods.

    MUDANÇAS:
    - Removido parâmetro xai_method (sempre usa comparação dos 3 métodos)
    - Adicionado xai_methods para selecionar quais métodos usar
    - Melhorado logging de XAI
    """

    def __init__(self, arguments):
        self.neural_network_model = None
        self.batch_size = arguments.batch_size
        self.number_splits = arguments.number_splits
        self.number_epochs = arguments.number_epochs
        self.loss_function = arguments.wav_to_vec_loss_function
        self.optimizer_function = arguments.wav_to_vec_optimizer_function
        self.window_size_factor = arguments.wav_to_vec_window_size_factor
        self.decibel_scale_factor = arguments.wav_to_vec_decibel_scale_factor
        self.hop_length = arguments.wav_to_vec_hop_length
        self.overlap = arguments.wav_to_vec_overlap
        self.window_size = self.hop_length * self.window_size_factor
        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension
        self.input_dimension = arguments.wav_to_vec_input_dimension
        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory

        # XAI configuration - UPDATED
        self.generate_xai = getattr(arguments, 'generate_xai', True)
        self.num_xai_samples = getattr(arguments, 'num_xai_samples', 30)
        self.xai_output_dir = getattr(arguments, 'xai_output_dir', './wav2vec2_xai_outputs')

        # NEW: Select which XAI methods to use
        # Options: 'all', 'attention', 'integrated_gradients', 'saliency'
        self.xai_methods = getattr(arguments, 'xai_methods', 'all')

        self.freeze_encoder = getattr(arguments, 'freeze_encoder', False)
        self.xai_only_last_fold = getattr(arguments, 'xai_only_last_fold', True)

        WindowGenerator.__init__(self, self.window_size, self.overlap)

        # Log XAI configuration
        logging.info(f"\n{'=' * 60}")
        logging.info("XAI CONFIGURATION (CORRECTED METHODS)")
        logging.info(f"{'=' * 60}")
        logging.info(f"Generate XAI: {self.generate_xai}")
        logging.info(f"XAI samples: {self.num_xai_samples}")
        logging.info(f"XAI output dir: {self.xai_output_dir}")
        logging.info(f"XAI methods: {self.xai_methods}")
        logging.info(f"XAI only last fold: {self.xai_only_last_fold}")
        logging.info(f"{'=' * 60}\n")

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """Load audio data (unchanged)."""
        logging.info("Starting data loading process.")
        list_spectrogram, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        for _, sub_directory in enumerate(list_class_path):
            logging.info(f"Processing directory: {sub_directory}")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)
                label = self.__get_label__(file_name)

                for (start, end) in self.generate_windows(raw_signal):
                    if len(raw_signal[start:end]) == self.window_size:
                        list_spectrogram.append(
                            self.normalization_signal(raw_signal[start:end])
                        )
                        list_labels.append(label)

        array_features = numpy.array(list_spectrogram, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)
        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Data loading completed successfully.")
        logging.info(f"Total samples loaded: {len(array_labels)}")

        return array_features, array_labels

    def train(self) -> tuple:
        """Train with corrected XAI methods."""
        # Load data
        features, labels = self.load_data(self.dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data
        features_train_validation, features_test, labels_train_validation, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        features_test, labels_test = self.balance_class(features_test, labels_test)

        # K-fold
        instance_k_fold = StratifiedKFold(
            n_splits=self.number_splits,
            shuffle=True,
            random_state=42
        )

        history_model = None
        probabilities_list = []
        real_labels_list = []

        logging.info(f"\n{'=' * 80}")
        logging.info(f"STARTING K-FOLD CROSS-VALIDATION (k={self.number_splits})")
        logging.info(f"{'=' * 80}\n")

        for fold_idx, (train_indexes, validation_indexes) in enumerate(
                instance_k_fold.split(features_train_validation, labels_train_validation),
                start=1
        ):
            logging.info(f"\n{'=' * 80}")
            logging.info(f"FOLD {fold_idx}/{self.number_splits}")
            logging.info(f"{'=' * 80}")

            # Split fold
            features_train = features_train_validation[train_indexes]
            features_validation = features_train_validation[validation_indexes]
            labels_train = labels_train_validation[train_indexes]
            labels_validation = labels_train_validation[validation_indexes]

            # Balance
            features_train, labels_train = self.balance_class(features_train, labels_train)

            logging.info(f"Training samples: {len(features_train)}")
            logging.info(f"Validation samples: {len(features_validation)}")
            logging.info(f"Test samples: {len(features_test)}")

            # Build model
            self.build_model()
            self.neural_network_model.summary()

            # XAI for this fold?
            should_generate_xai = self.generate_xai
            if self.xai_only_last_fold:
                should_generate_xai = should_generate_xai and (fold_idx == self.number_splits)

            logging.info(f"\n{'=' * 60}")
            logging.info(f"XAI FOR FOLD {fold_idx}")
            logging.info(f"{'=' * 60}")
            logging.info(f"Will generate XAI: {should_generate_xai}")
            if should_generate_xai:
                logging.info(f"Methods: {self.xai_methods}")
            logging.info(f"{'=' * 60}\n")

            # Train with CORRECTED parameters
            history_model = self.compile_and_train(
                train_data=features_train,
                train_labels=labels_train,
                epochs=self.number_epochs,
                batch_size=self.batch_size,
                validation_data=(features_validation, labels_validation),
                generate_xai=should_generate_xai,
                num_xai_samples=self.num_xai_samples,
                xai_output_dir=f"{self.xai_output_dir}/fold_{fold_idx}",
                xai_methods=self.xai_methods,  # UPDATED
                freeze_encoder=self.freeze_encoder
            )

            # Evaluate
            logging.info(f"\nEvaluating fold {fold_idx} on test set...")
            model_predictions = self.neural_network_model.predict(
                features_test,
                batch_size=self.batch_size,
                verbose=1
            )
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_test)

            metrics, confusion_matrix = self.calculate_metrics(
                predicted_labels,
                labels_test
            )
            metrics_list.append(metrics)
            confusion_matriz_list.append(confusion_matrix)

            # Log results
            logging.info(f"\nFold {fold_idx} Results:")
            logging.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            logging.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logging.info(f"  Recall: {metrics.get('recall', 0):.4f}")
            logging.info(f"  F1-Score: {metrics.get('f1', 0):.4f}")

        # Summary
        logging.info(f"\n{'=' * 80}")
        logging.info("CROSS-VALIDATION COMPLETED")
        logging.info(f"{'=' * 80}")

        avg_accuracy = numpy.mean([m.get('accuracy', 0) for m in metrics_list])
        avg_f1 = numpy.mean([m.get('f1', 0) for m in metrics_list])

        logging.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logging.info(f"Average F1-Score: {avg_f1:.4f}\n")

        return self.__cast_to_dic__(
            metrics_list,
            probabilities_list,
            real_labels_list,
            confusion_matriz_list,
            history_model
        )