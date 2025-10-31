#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{2}.{0}.{1}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/27'
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
    Wav2Vec2 training process CORRECTED for raw waveform input.

    CONFIGURED FOR: 2.5 seconds @ 8kHz = 20000 samples

    Key differences from original:
    - Loads RAW WAVEFORMS instead of spectrograms
    - FIXED audio length: 2.5s @ 8kHz = 20000 samples
    - Proper data shape for Wav2Vec2 (N, 20000, 1)
    - Compatible with two-phase training
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

        # Calculate window_size from hop_length and factor
        self.window_size = self.hop_length * self.window_size_factor

        self.sample_rate = arguments.sample_rate
        self.file_extension = arguments.file_extension

        # NEW: Option to use full audio or windowing
        # use_full_audio=True: Process entire audio file (no windowing)
        # use_full_audio=False: Use windowing (original behavior)
        self.use_full_audio = False

        # CRITICAL: Input dimension calculation
        # FIXED: 2.5 seconds audio duration
        if self.use_full_audio:
            # Calculate for 2.5 seconds
            # For 8kHz: 2.5 × 8000 = 20000 samples
            # For 16kHz: 2.5 × 16000 = 40000 samples
            audio_duration_seconds = 2.5
            default_max_length = int(audio_duration_seconds * self.sample_rate)

            # Allow override via arguments, but default to 2.5s
            self.max_audio_length = getattr(arguments, 'wav_to_vec_max_audio_length', default_max_length)

            # Set input dimension
            self.input_dimension = (self.max_audio_length, 1)

            logging.info(
                f"✓ Full audio mode: FIXED length = {self.max_audio_length} samples "
                f"({self.max_audio_length / self.sample_rate:.2f}s @ {self.sample_rate}Hz)"
            )
        else:
            # For windowing, use calculated window_size
            self.input_dimension = (self.window_size, 1)

        self.number_classes = arguments.number_classes
        self.dataset_directory = arguments.dataset_directory

        self.freeze_encoder = getattr(arguments, 'freeze_encoder', False)

        WindowGenerator.__init__(self, self.window_size, self.overlap)

        # Log the configuration
        logging.info("=" * 80)
        logging.info("WAV2VEC2 PROCESS INITIALIZED (2.5s AUDIO)")
        logging.info("=" * 80)
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"K-folds: {self.number_splits}")
        logging.info(f"Epochs: {self.number_epochs}")
        logging.info(f"Sample rate: {self.sample_rate} Hz")
        logging.info(f"Use full audio: {self.use_full_audio}")
        #logging.info(f"Audio duration: {self.max_audio_length / self.sample_rate:.2f} seconds")
        logging.info(f"Input dimension: {self.input_dimension} (FIXED)")
        logging.info(f"Number of classes: {self.number_classes}")
        logging.info("=" * 80)

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Load RAW WAVEFORMS for Wav2Vec2.

        Two modes:
        1. use_full_audio=True: Load complete audio files (FIXED 2.5s length)
        2. use_full_audio=False: Load audio windows

        Returns:
            tuple: (waveforms, labels)
                - waveforms: numpy array of shape (N, 20000, 1) for 8kHz
                - labels: numpy array of shape (N,)
        """
        if self.use_full_audio:
            logging.info(
                f"Starting data loading process (FULL AUDIO - FIXED {self.max_audio_length / self.sample_rate:.2f}s).")
        else:
            logging.info("Starting data loading process (RAW WAVEFORMS with windowing).")

        list_waveforms, list_labels, list_class_path = [], [], []
        file_extension = file_extension or self.file_extension

        list_class_path = self.__create_dir__(sub_directories, list_class_path)

        # For full audio mode, use the pre-defined max_audio_length
        # NO auto-detection needed - we use FIXED length
        if self.use_full_audio:
            max_length = self.max_audio_length
            logging.info(f"Using FIXED audio length: {max_length} samples ({max_length / self.sample_rate:.2f}s)")
        else:
            max_length = self.window_size

        # Load data
        for _, sub_directory in enumerate(list_class_path):
            logging.info(f"Processing directory: {sub_directory}")

            for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):
                raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)
                label = self.__get_label__(file_name)

                if self.use_full_audio:
                    # Process FULL audio file with FIXED length
                    # Normalize the waveform
                    waveform = self.normalization_signal(raw_signal)

                    # Pad or truncate to FIXED max_length
                    if len(waveform) < max_length:
                        # Pad with zeros
                        waveform = numpy.pad(waveform, (0, max_length - len(waveform)), mode='constant')
                    elif len(waveform) > max_length:
                        # Truncate
                        waveform = waveform[:max_length]
                    # else: exactly the right length, use as is

                    list_waveforms.append(waveform)
                    list_labels.append(label)

                else:
                    # Use windowing (original behavior)
                    for (start, end) in self.generate_windows(raw_signal):
                        if len(raw_signal[start:end]) == self.window_size:
                            # Store RAW WAVEFORM (normalized)
                            waveform_segment = self.normalization_signal(raw_signal[start:end])
                            list_waveforms.append(waveform_segment)
                            list_labels.append(label)

        # Convert to numpy arrays
        # Shape: (N, audio_length, 1) - raw waveforms with channel dimension
        array_features = numpy.array(list_waveforms, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)
        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Data loading completed successfully.")
        logging.info(f"Total samples loaded: {len(array_labels)}")
        logging.info(f"Feature shape: {array_features.shape}")

        if self.use_full_audio:
            logging.info(f"  ↳ FULL AUDIO mode: Each sample is {self.max_audio_length / self.sample_rate:.2f}s")
            logging.info(f"  ↳ All audio files padded/truncated to {self.max_audio_length} samples")
        else:
            logging.info(f"  ↳ WINDOWING mode: Each sample is a {self.window_size}-sample window")

        logging.info(f"  ↳ This is RAW WAVEFORM, not spectrogram!")

        return array_features, array_labels

    def train(self) -> tuple:
        """
        Train Wav2Vec2 with K-fold cross-validation.

        This method follows the same pattern as ResidualProcess but adapted
        for Wav2Vec2's two-phase training (pre-training + fine-tuning).

        Returns:
            tuple: (metrics_dict, probabilities_list, real_labels_list,
                   confusion_matrix_list, history)
        """
        logging.info(f"\n{'=' * 80}")
        logging.info(f"STARTING K-FOLD CROSS-VALIDATION (k={self.number_splits})")
        logging.info(f"{'=' * 80}\n")

        # Load data (RAW WAVEFORMS!)
        # Data will be loaded with FIXED length (20000 samples @ 8kHz)
        features, labels = self.load_data(self.dataset_directory)
        metrics_list, confusion_matriz_list = [], []
        labels = numpy.array(labels).astype(float)

        # Split data
        features_train_validation, features_test, labels_train_validation, labels_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Balance test set
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

            # Balance training set
            features_train, labels_train = self.balance_class(features_train, labels_train)

            logging.info(f"Training samples: {len(features_train)}")
            logging.info(f"Validation samples: {len(features_validation)}")
            logging.info(f"Test samples: {len(features_test)}")

            # Build model (input_dimension is FIXED at 20000 samples)
            self.build_model()
            self.neural_network_model.summary()

            # Train with two-phase approach
            # The compile_and_train method in AudioWav2Vec2 handles:
            # 1. Pre-training with contrastive loss
            # 2. Fine-tuning with supervised loss
            history_model = self.compile_and_train(train_data=features_train,
                                                   train_labels=labels_train,
                                                   epochs=self.number_epochs,
                                                   batch_size=self.batch_size,
                                                   validation_data=(features_validation, labels_validation))

            # Evaluate on test set
            logging.info(f"\nEvaluating fold {fold_idx} on test set...")
            model_predictions = self.neural_network_model.predict(
                features_test,
                batch_size=self.batch_size,
                verbose=1
            )
            predicted_labels = numpy.argmax(model_predictions, axis=1)

            probabilities_list.append(model_predictions)
            real_labels_list.append(labels_test)

            # Calculate metrics
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

