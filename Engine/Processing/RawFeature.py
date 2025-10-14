#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
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

    import numpy

    import logging
    import librosa
    import glob

    from tqdm import tqdm

    from Engine.Processing.WindowGenerator import WindowGenerator

except ImportError as error:
    print(error)
    sys.exit(-1)

class RawDataLoader(WindowGenerator):
    """
    A class for loading and processing audio data for machine learning models.

    This class inherits from the `WindowGenerator` class and is designed to load and
    preprocess audio data, typically for tasks such as classification. It supports
    reading audio files from a directory, extracting raw features, and
    normalizing the data. It also generates sliding windows over the audio signal
    for training purposes.

    Attributes:
        @sample_rate (int): The sample rate for loading the audio files.
        @window_size (int): The size of the window for segmenting the audio signal.
        @overlap (int): The amount of overlap between adjacent windows.
        @window_size_factor (int): The factor for segmenting the audio into smaller windows during preprocessing.
        @file_extension (str): The file extension of the audio files to load (e.g., '.wav').

    Methods:
        @load_data(sub_directories: str, file_extension: Optional[str] = None) -> tuple:
            Loads the audio data from the specified directories and processes it into spectrogram features and labels.
        @get_class_paths(parent_directory: str) -> list:
            Returns the list of class directories in the parent directory.
        @process_all_classes(class_paths: list, file_extension: str) -> tuple:
            Processes all classes by iterating over their respective directories.
        @process_class_directory(class_path: str, file_extension: str) -> tuple:
            Processes a single class directory by loading and processing its audio files.
        @process_file(file_name: str) -> tuple:
            Processes a single audio file, extracting features and labels from it.
        @extract_label_from_path(file_path: str) -> int:
            Extracts the class label from the file path based on the directory structure.
        @segment_and_normalize(segment: numpy.ndarray) -> numpy.ndarray:
            Segments and normalizes an audio segment (patch) for further processing.

    Example:
        >>> python
        ...     # Initialize the DataLoader instance
        ...     data_loader = RawDataLoader(
        ...     raw_feature_sample_rate=22050,   # Standard audio sample rate
        ...     raw_feature_window_size=1024,    # Window size for segmentation
        ...     raw_feature_overlap=512,         # Overlap between windows
        ...     raw_feature_window_size_factor=2,  # Factor for splitting segments into smaller windows
        ...     raw_feature_file_extension='.wav'  # File extension for audio files
        ...     )
        ...
        ...     # Load data from the specified directory
        ...     features, labels = data_loader.load_data_patcher_spectrogram_format(sub_directories="path_to_audio_data")
        ...
        ...     # features: numpy array of spectrograms (processed audio data)
        ...     # labels: numpy array of labels corresponding to each spectrogram
        ...     print(f"Loaded {len(features)} feature arrays and {len(labels)} labels.")
        >>>
    """

    def __init__(self,
                 raw_feature_sample_rate: int,
                 raw_feature_window_size: int,
                 raw_feature_overlap: int,
                 raw_feature_window_size_factor: int,
                 raw_feature_file_extension: str):

        super().__init__(raw_feature_window_size, raw_feature_overlap)
        # Store the parameters
        self._raw_feature_sample_rate = raw_feature_sample_rate
        self._raw_feature_window_size = raw_feature_window_size
        self._raw_feature_overlap = raw_feature_overlap
        self._raw_feature_window_size_factor = raw_feature_window_size_factor
        self._raw_feature_file_extension = raw_feature_file_extension

    def load_data_raw_format(self, sub_directories: str = None, file_extension: str = None) -> tuple:
        """
        Loads the audio data from the specified subdirectories, processes the audio files
        into feature, and returns the features and their corresponding labels.

        Args:
            sub_directories (str): The directory containing subdirectories with audio data.
            file_extension (Optional[str]): The file extension for the audio files (optional,
             defaults to the instance's file_extension).

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The array of spectrogram features.
                - numpy.ndarray: The array of corresponding labels.
        """
        logging.info("Starting to load data...")

        # Use the file_extension passed to the function or the one stored in the instance
        file_extension = file_extension or self._raw_feature_file_extension

        # Check if the provided directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Get the paths to the class directories
        class_paths = self._get_class_paths(sub_directories)
        logging.info(f"Found {len(class_paths)} classes.")

        # Process each class directory to obtain feature and labels
        list_all_feature, list_all_labels = self._raw_feature_process_all_classes(class_paths, file_extension)

        # Convert the feature list into a numpy array
        array_features = numpy.array(list_all_feature, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)  # Add channel dimension for compatibility with models

        logging.info(f"Loaded {len(array_features)} feature arrays.")
        logging.info("Data loading complete.")

        return array_features, numpy.array(list_all_labels, dtype=numpy.int32)

    @staticmethod
    def _get_class_paths(sub_directories: str) -> list:
        """
        Retrieves the paths of subdirectories representing different classes.

        Args:
            sub_directories (str): The parent directory containing subdirectories for each class.

        Returns:
            list: A list of paths to the class subdirectories.
        """
        logging.info(f"Reading subdirectories in '{sub_directories}'...")

        # Get all class subdirectories
        class_paths = [os.path.join(sub_directories, sub_dir_name) for sub_dir_name in os.listdir(sub_directories) if
                       os.path.isdir(os.path.join(sub_directories, sub_dir_name))]

        logging.info(f"Found {len(class_paths)} class directories.")
        return class_paths


    def _raw_feature_process_all_classes(self, class_paths: list, file_extension: str) -> tuple:
        """
        Processes all class directories, loading and processing their respective files.

        Args:
            class_paths (list): List of paths to the class directories.
            file_extension (str): The file extension of the audio files to load.

        Returns:
            tuple: A tuple containing:
                - list: A list of feature extracted from the audio files.
                - list: A list of labels corresponding to the feature.
        """
        list_feature, list_labels = [], []

        # Iterate through each class directory and process its files
        for class_path in class_paths:
            logging.info(f"Processing class directory: {class_path}...")
            feature_processed, labels = self._process_class_directory(class_path, file_extension)
            list_feature.extend(feature_processed)
            list_labels.extend(labels)

        return list_feature, list_labels

    def _process_class_directory(self, class_path: str, file_extension: str) -> tuple:
        """
        Processes a single class directory, loading and processing all audio files.

        Args:
            class_path (str): Path to the class directory.
            file_extension (str): The file extension of the audio files.

        Returns:
            tuple: A tuple containing:
                - list: A list of feature extracted from the files in the class directory.
                - list: A list of labels corresponding to the feature.
        """
        list_feature, list_labels = [], []

        # Loop through all audio files in the directory using glob pattern
        for file_name in tqdm(glob.glob(os.path.join(class_path, file_extension))):
            try:
                # Process each file and extract feature and labels
                file_spectrogram, file_labels = self._process_file(file_name)
                list_feature.extend(file_spectrogram)
                list_labels.extend(file_labels)
            except Exception as e:
                logging.error(f"Error processing file '{file_name}': {e}")

        return list_feature, list_labels

    @staticmethod
    def _extract_label_from_path(file_path: str) -> int:
        """
        Extracts the class label from the file path based on the directory structure.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            int: The extracted class label.
        """
        label_from_filename = file_path.split('/')[-2].split('_')[0]
        return int(label_from_filename)

    def _raw_feature_process_file(self, file_name: str) -> tuple:
        """
        Processes a single audio file, extracting its feature and corresponding label.

        Args:
            file_name (str): Path to the audio file.

        Returns:
            tuple: A tuple containing:
                - list: A list of feature segments extracted from the file.
                - list: A list of labels corresponding to the feature segments.
        """
        # Load the audio file using librosa at the specified sample rate
        raw_signal, _ = librosa.load(file_name, sr=self._raw_feature_sample_rate)

        # Extract the label from the file path (based on directory structure)
        sound_label = self._extract_label_from_path(file_name)

        list_spectrogram, labels_feature = [], []

        # Generate windows and process each window of the raw audio signal
        for (start, end) in self.generate_windows(raw_signal):
            # Only process windows that match the desired window size
            if len(raw_signal[start:end]) == self._raw_feature_window_size:
                normalized_segment = self._raw_feature_segment_and_normalize(raw_signal[start:end])
                list_spectrogram.append(normalized_segment)
                labels_feature.append(sound_label)

        return list_spectrogram, labels_feature

    def _raw_feature_segment_and_normalize(self, segment: numpy.ndarray) -> numpy.ndarray:
        """
        Segments and normalizes an audio segment into smaller patches for training.

        Args:
            segment (numpy.ndarray): A segment of the audio signal.

        Returns:
            numpy.ndarray: The normalized patches.
        """
        # Calculate the size of each local window
        local_window = len(segment) // self._raw_feature_window_size_factor

        # Split the segment into smaller patches
        patches = [segment[i:i + local_window] for i in range(0, len(segment), local_window)]
        patches = numpy.abs(numpy.array(patches))  # Ensure absolute values to avoid negative values

        # Normalize the patches to the range [0, 1]
        signal_min = numpy.min(patches)
        signal_max = numpy.max(patches)

        if signal_max != signal_min:
            normalized_patches = (patches - signal_min) / (signal_max - signal_min)
        else:
            normalized_patches = numpy.zeros_like(patches)  # Avoid division by zero if max == min

        return normalized_patches
