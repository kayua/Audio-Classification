import os
import glob
import numpy
import librosa
import logging

from tqdm import tqdm

from Engine.Processing.WindowGenerator import WindowGenerator


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
        load_data(sub_directories: str, file_extension: Optional[str] = None) -> tuple:
            Loads the audio data from the specified directories and processes it into spectrogram features and labels.
        get_class_paths(parent_directory: str) -> list:
            Returns the list of class directories in the parent directory.
        process_all_classes(class_paths: list, file_extension: str) -> tuple:
            Processes all classes by iterating over their respective directories.
        process_class_directory(class_path: str, file_extension: str) -> tuple:
            Processes a single class directory by loading and processing its audio files.
        process_file(file_name: str) -> tuple:
            Processes a single audio file, extracting features and labels from it.
        extract_label_from_path(file_path: str) -> int:
            Extracts the class label from the file path based on the directory structure.
        segment_and_normalize(segment: numpy.ndarray) -> numpy.ndarray:
            Segments and normalizes an audio segment (patch) for further processing.

    Example:
        >>> python
        ...     # Initialize the DataLoader instance
        ...     data_loader = RawDataLoader(
        ...     sample_rate=22050,   # Standard audio sample rate
        ...     window_size=1024,    # Window size for segmentation
        ...     overlap=512,         # Overlap between windows
        ...     window_size_factor=2,  # Factor for splitting segments into smaller windows
        ...     file_extension='.wav'  # File extension for audio files
        ...     )
        ...
        ...     # Load data from the specified directory
        ...     features, labels = data_loader.load_data(sub_directories="path_to_audio_data")
        ...
        ...     # features: numpy array of spectrograms (processed audio data)
        ...     # labels: numpy array of labels corresponding to each spectrogram
        ...     print(f"Loaded {len(features)} feature arrays and {len(labels)} labels.")
        >>>
    """

    def __init__(self,
                 sample_rate: int,
                 window_size: int,
                 overlap: int,
                 window_size_factor: int,
                 file_extension: str):

        super().__init__(window_size, overlap)
        """
        Initializes the DataLoader instance.

        Args:
            sample_rate (int): The sample rate for loading audio files.
            window_size (int): The size of the sliding window for segmenting audio signals.
            overlap (int): The overlap between consecutive windows.
            window_size_factor (int): Factor for dividing segments during normalization.
            file_extension (str): File extension for the audio files to load (e.g., '.wav').
        """

        # Store the parameters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_size_factor = window_size_factor
        self.file_extension = file_extension

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:
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
        file_extension = file_extension or self.file_extension

        # Check if the provided directory exists
        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        # Get the paths to the class directories
        class_paths = self._get_class_paths(sub_directories)
        logging.info(f"Found {len(class_paths)} classes.")

        # Process each class directory to obtain feature and labels
        list_all_feature, list_all_labels = self._process_all_classes(class_paths, file_extension)

        # Convert the feature list into a numpy array
        array_features = numpy.array(list_all_feature, dtype=numpy.float32)
        array_features = numpy.expand_dims(array_features, axis=-1)  # Add channel dimension for compatibility with models

        logging.info(f"Loaded {len(array_features)} feature arrays.")
        logging.info("Data loading complete.")

        return array_features, numpy.array(list_all_labels, dtype=numpy.int32)


    @staticmethod
    def _get_class_paths(parent_directory: str) -> list:
        """
        Retrieves the class subdirectories within the given parent directory.

        Args:
            parent_directory (str): The directory containing class subdirectories.

        Returns:
            list: A list of paths to the class subdirectories.
        """
        class_paths = []

        # Loop through each directory in the parent directory
        for class_dir in os.listdir(parent_directory):
            class_path = os.path.join(parent_directory, class_dir)

            # If it is a directory, add it to the list
            if os.path.isdir(class_path):
                class_paths.append(class_path)

        return class_paths

    def _process_all_classes(self, class_paths: list, file_extension: str) -> tuple:
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


    def _process_file(self, file_name: str) -> tuple:
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
        raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)

        # Extract the label from the file path (based on directory structure)
        sound_label = self._extract_label_from_path(file_name)

        list_spectrogram, labels_feature = [], []

        # Generate windows and process each window of the raw audio signal
        for (start, end) in self.generate_windows(raw_signal):
            # Only process windows that match the desired window size
            if len(raw_signal[start:end]) == self.window_size:
                normalized_segment = self._segment_and_normalize(raw_signal[start:end])
                list_spectrogram.append(normalized_segment)
                labels_feature.append(sound_label)

        return list_spectrogram, labels_feature

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

    def _segment_and_normalize(self, segment: numpy.ndarray) -> numpy.ndarray:
        """
        Segments and normalizes an audio segment into smaller patches for training.

        Args:
            segment (numpy.ndarray): A segment of the audio signal.

        Returns:
            numpy.ndarray: The normalized patches.
        """
        # Calculate the size of each local window
        local_window = len(segment) // self.window_size_factor

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
