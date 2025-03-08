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

    import logging
    import librosa

    from tqdm import tqdm

    from Engine.Processing.PathTools import PathTools

    from Engine.Processing.WindowGenerator import WindowGenerator
    from Engine.Transformations.SpectrogramPatcher import SpectrogramPatcher

except ImportError as error:
    print(error)
    sys.exit(-1)

class PatcherSpectrogramFeature(SpectrogramPatcher, WindowGenerator, PathTools):

    def __init__(self, sample_rate: int, window_size: int, overlap: int, number_filters_spectrogram: int,
                 window_size_fft: int, hop_length: int, decibel_scale_factor: int, patch_size: tuple[int, int]):

        # Store all the parameters as attributes
        super().__init__(patch_size)
        self.audio_duration = None
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.number_filters_spectrogram = number_filters_spectrogram
        self.window_size_fft = window_size_fft
        self.hop_length = hop_length
        self.decibel_scale_factor = decibel_scale_factor

    def load_data_patcher_spectrogram_format(self, sub_directories: str, file_extension: str = "*.wav",
                                             stack_segments=False) -> tuple:

        logging.info("Starting to load data...")

        # If no extension is provided, use default "*.wav"
        file_extension = file_extension or "*.wav"

        # Check if the directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Get class paths (subdirectories representing classes)
        class_paths = self._get_class_paths(sub_directories)

        # Process all files to extract features and labels
        list_all_spectrogram, list_all_labels = self._process_files(class_paths, file_extension)

        # Prepare the final output (features and labels)
        return self._prepare_output(list_all_spectrogram, list_all_labels, stack_segments)

    def _process_files(self, class_paths: list, file_extension: str) -> tuple:
        """
        Processes all files in the class directories to extract their features and labels.

        Args:
            class_paths (list): List of paths to the class subdirectories.
            file_extension (str): The extension of the files to process.

        Returns:
            tuple: A tuple containing:
                - list: A list of extracted spectrogram features.
                - list: A list of corresponding labels.
        """
        list_spectrogram, list_labels = [], []

        # Loop through each class directory
        for class_path in class_paths:
            logging.info(f"Processing class directory: {class_path}...")
            for file_name in tqdm(glob.glob(os.path.join(class_path, file_extension))):
                try:
                    # Load the audio file and extract its label
                    signal, label = self._load_audio_and_extract_label(file_name)

                    # Extract the spectrogram for the audio signal
                    list_spectrogram.extend(self._extract_spectrogram(signal))

                    # Extend the labels to match the number of spectrogram segments
                    list_labels.extend([label] * len(list_spectrogram))
                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        return list_spectrogram, list_labels

    def _load_audio_and_extract_label(self, file_name: str) -> tuple:
        """
        Loads an audio file and extracts the corresponding label from the directory structure.

        Args:
            file_name (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the audio signal (numpy array) and the label (str).
        """
        # Load the audio signal with librosa
        raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)

        # Extract the label based on the directory structure
        signal_label = os.path.basename(os.path.dirname(file_name)).split('_')[0]

        # Calculate the maximum length of the signal based on the desired duration
        max_length = int(self.sample_rate * self.audio_duration)

        # Pad the signal if it's shorter than the required length
        if len(raw_signal) < max_length:
            padding = max_length - len(raw_signal)
            raw_signal = numpy.pad(raw_signal, (0, padding), 'constant')

        # Truncate the signal to the maximum length
        raw_signal = raw_signal[:max_length]

        return raw_signal, signal_label


    def _extract_spectrogram(self, signal: numpy.ndarray) -> list:
        """
        Extracts Mel spectrograms from the audio signal by generating windows.

        Args:
            signal (numpy.ndarray): The raw audio signal.

        Returns:
            list: A list of Mel spectrograms for each window of the signal.
        """
        list_spectrogram = []

        # Generate windows of the audio signal for processing
        for start, end in self.generate_windows(len(signal)):
            if len(signal[start:end]) == self.window_size:
                # Generate a Mel spectrogram for each window
                list_spectrogram.append(self._generate_mel_spectrogram(signal[start:end]))

        return list_spectrogram

    def _generate_mel_spectrogram(self, signal_window: numpy.ndarray) -> numpy.ndarray:
        """
        Generates the Mel spectrogram for a given audio window.

        Args:
            signal_window (numpy.ndarray): A segment of the audio signal.

        Returns:
            numpy.ndarray: The Mel spectrogram for the segment.
        """
        melody_spectrogram = librosa.feature.melspectrogram(
            y=signal_window,
            sr=self.sample_rate,
            n_mels=self.number_filters_spectrogram,
            n_fft=self.window_size_fft,
            hop_length=self.hop_length
        )
        # Convert to dB and normalize
        spectrogram_decibel_scale =\
            (librosa.power_to_db(melody_spectrogram, ref=numpy.max) / self.decibel_scale_factor) + 1

        return self.split_into_patches(spectrogram_decibel_scale)



    def _prepare_output(self, list_spectrogram: list, labels: list, stack_segments: bool) -> tuple:
        """
        Prepares the spectrogram features and labels, reshaping them for model input.

        Args:
            list_spectrogram (list): A list of spectrograms extracted from the audio files.
            labels (list): A list of labels corresponding to the spectrograms.
            stack_segments (bool): Whether to stack the spectrogram segments or not.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The reshaped spectrogram features.
                - numpy.ndarray: The labels as a numpy array.
        """
        # Reshape spectrograms into the format [samples, filters, time_steps, 1]
        array_features = numpy.array(list_spectrogram).reshape(
            len(list_spectrogram),
            self.number_filters_spectrogram,
            self.window_size // self.hop_length,
            1
        )
        array_labels = numpy.array(labels, dtype=numpy.int32)

        # Stack the segments if requested
        if stack_segments:
            new_shape = list(array_features.shape)
            new_shape[1] += 1  # Increase the number of filters by 1 for stacking
            stacked_features = numpy.zeros(new_shape)
            stacked_features[:, :self.number_filters_spectrogram, :, :] = array_features
            return stacked_features.astype(numpy.float32), array_labels

        return array_features.astype(numpy.float32), array_labels