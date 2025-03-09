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

except ImportError as error:
    print(error)
    sys.exit(-1)



class SpectrogramFeature(WindowGenerator, PathTools):
    """
    A class for generating spectrogram features from audio signals for machine
    learning tasks.

    This class processes audio data and extracts Mel spectrograms as features,
    which can be used for tasks like classification or recognition. It includes
    methods for loading audio data, generating spectrograms, and preparing data
     for model training.

    Attributes:
        @sample_rate (int): The sample rate at which to load audio files.
        @window_size (int): The window size used for segmenting the audio signal.
        @overlap (int): The amount of overlap between adjacent windows.
        @number_filters_spectrogram (int): The number of Mel filters to use in the spectrogram.
        @window_size_fft (int): The window size for the Fast Fourier Transform (FFT).
        @hop_length (int): The hop length for computing the spectrogram.
        @decibel_scale_factor (int): The scale factor for decibel normalization of
        the spectrogram.

    Methods:
        @load_data(sub_directories: str, file_extension: str, stack_segments: bool) -> tuple:
            Loads and processes the audio files from the specified directories,
             then extracts and normalizes spectrogram features.

        @get_class_paths(sub_directories: str) -> list:
            Retrieves the paths of all subdirectories representing the different
            classes (categories).

        @process_files(class_paths: list, file_extension: str) -> tuple:
            Processes all files within the class directories, extracts their features,
            and associates them with their labels.

        @load_audio_and_extract_label(file_name: str) -> tuple:
            Loads the audio signal from a file and extracts the label based on the
            directory structure.

        @extract_spectrogram(signal: numpy.ndarray) -> list:
            Extracts Mel spectrograms from the audio signal.

        @generate_mel_spectrogram(signal_window: numpy.ndarray) -> numpy.ndarray:
            Generates the Mel spectrogram from a given audio window.

        @prepare_output(list_spectrogram: list, labels: list, stack_segments: bool) -> tuple:
            Prepares the spectrogram features and labels, reshaping them for model input.
            Optionally stacks the segments.

    Example:
    >>> # Initialize the SpectrogramFeature instance
    ...     spectrogram_feature = SpectrogramFeature(
    ...     sample_rate=22050,               # Standard sample rate for audio
    ...     window_size=1024,                # Size of the sliding window for segmenting audio
    ...     overlap=512,                     # Overlap between consecutive windows
    ...     number_filters_spectrogram=40,   # Number of Mel filters for spectrogram
    ...     window_size_fft=2048,            # Window size for FFT
    ...     hop_length=512,                  # Hop length for spectrogram calculation
    ...     decibel_scale_factor=80         # Factor to scale decibel values
    ...     )
    ...
    ...     # Load and process the data from a directory containing audio files
    ...     features, labels = spectrogram_feature.load_data_patcher_spectrogram_format(
    ...     sub_directories="path_to_audio_data",  # Directory with subdirectories representing classes
    ...     file_extension=".wav",                # Extension of the audio files
    ...     stack_segments=False                  # Whether to stack segments (optional)
    ...     )
    ...
    ...     # features: Spectrogram data as a numpy array
    ...     # labels: Corresponding class labels as a numpy array
    ...     print(f"Loaded {len(features)} spectrograms and {len(labels)} labels.")
    >>>
    """

    def __init__(self, sample_rate: int, window_size: int, overlap: int, number_filters_spectrogram: int,
                 window_size_fft: int, hop_length: int, decibel_scale_factor: int):
        """
        Initializes the SpectrogramFeature class.

        Args:
            sample_rate (int): The sample rate to load audio files.
            window_size (int): The window size for segmenting the audio signal.
            overlap (int): The overlap between consecutive windows.
            number_filters_spectrogram (int): The number of Mel filters in the spectrogram.
            window_size_fft (int): The FFT window size.
            hop_length (int): The hop length for computing the spectrogram.
            decibel_scale_factor (int): The decibel scale factor for normalization.
        """

        # Store all the parameters as attributes
        WindowGenerator.__init__(self, window_size, overlap)

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.number_filters_spectrogram = number_filters_spectrogram
        self.window_size_fft = window_size_fft
        self.hop_length = hop_length
        self.decibel_scale_factor = decibel_scale_factor

    def load_data_spectrogram_format(self, sub_directories: str, file_extension: str = "*.wav", stack_segments=False) -> tuple:
        """
        Loads and processes audio files, extracts Mel spectrogram features, and prepares data for training.

        Args:
            sub_directories (str): The path to the parent directory containing class subdirectories.
            file_extension (str, optional): The file extension to look for in the directory (default is "*.wav").
            stack_segments (bool, optional): Whether to stack the spectrogram segments (default is False).

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The spectrogram features as a numpy array.
                - numpy.ndarray: The labels corresponding to the spectrograms.
        """
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
        list_all_spectrogram, list_all_labels = self._spectrogram_feature_process_files(class_paths, file_extension)

        # Prepare the final output (features and labels)
        return self._spectrogram_feature_prepare_output(list_all_spectrogram, list_all_labels, stack_segments)

    def _spectrogram_feature_process_files(self, class_paths: list, file_extension: str) -> tuple:
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
                    signal, label = self._spectrogram_feature_load_audio_and_extract_label(file_name)

                    # Extract the spectrogram for the audio signal
                    list_spectrogram.extend(self._spectrogram_feature_extract_spectrogram(signal))

                    # Extend the labels to match the number of spectrogram segments
                    list_labels.extend([label] * len(list_spectrogram))
                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        return list_spectrogram, list_labels

    def _spectrogram_feature_load_audio_and_extract_label(self, file_name: str) -> tuple:
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

        return raw_signal, signal_label


    def _spectrogram_feature_extract_spectrogram(self, signal: numpy.ndarray) -> list:
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
                list_spectrogram.append(self._spectrogram_feature_generate_mel_spectrogram(signal[start:end]))

        return list_spectrogram

    def _spectrogram_feature_generate_mel_spectrogram(self, signal_window: numpy.ndarray) -> numpy.ndarray:
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
        return (librosa.power_to_db(melody_spectrogram, ref=numpy.max) / self.decibel_scale_factor) + 1



    def _spectrogram_feature_prepare_output(self, list_spectrogram: list, labels: list, stack_segments: bool) -> tuple:
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