import os
import glob
import logging
import numpy as np
import librosa
from tqdm import tqdm

class DataLoader:

    def __init__(self,
                 sample_rate=16000,
                 window_size=1024,
                 overlap=2,
                 window_size_factor=4,
                 file_extension='*.wav'):

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_size_factor = window_size_factor
        self.file_extension = file_extension

    def load_data(self, sub_directories: str = None, file_extension: str = None) -> tuple:

        logging.info("Starting to load data...")

        file_extension = file_extension or self.file_extension

        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        class_paths = self._get_class_paths(sub_directories)
        logging.info(f"Found {len(class_paths)} classes.")

        all_spectrograms, all_labels = self._process_all_classes(class_paths, file_extension)

        features = np.array(all_spectrograms, dtype=np.float32)
        features = np.expand_dims(features, axis=-1)  # Add channel dimension

        logging.info(f"Loaded {len(features)} feature arrays.")
        logging.info("Data loading complete.")

        return features, np.array(all_labels, dtype=np.int32)

    def _get_class_paths(self, parent_directory: str) -> list:

        class_paths = []

        for class_dir in os.listdir(parent_directory):
            class_path = os.path.join(parent_directory, class_dir)

            if os.path.isdir(class_path):
                class_paths.append(class_path)

        return class_paths

    def _process_all_classes(self, class_paths: list, file_extension: str) -> tuple:

        list_spectrogram = []
        list_labels = []

        for class_path in class_paths:
            logging.info(f"Processing class directory: {class_path}...")
            spectrograms, labels = self._process_class_directory(class_path, file_extension)
            list_spectrogram.extend(spectrograms)
            list_labels.extend(labels)

        return list_spectrogram, list_labels

    def _process_class_directory(self, class_path: str, file_extension: str) -> tuple:

        spectrograms = []
        labels = []

        for file_name in tqdm(glob.glob(os.path.join(class_path, file_extension))):
            try:
                file_spectrograms, file_labels = self._process_file(file_name)
                spectrograms.extend(file_spectrograms)
                labels.extend(file_labels)
            except Exception as e:
                logging.error(f"Error processing file '{file_name}': {e}")

        return spectrograms, labels

    def _process_file(self, file_name: str) -> tuple:

        signal, _ = librosa.load(file_name, sr=self.sample_rate)

        label = self._extract_label_from_path(file_name)

        spectrograms = []
        labels = []

        for (start, end) in self.windows(signal, self.window_size, self.overlap):
            if len(signal[start:end]) == self.window_size:
                normalized_segment = self._segment_and_normalize(signal[start:end])
                spectrograms.append(normalized_segment)
                labels.append(label)

        return spectrograms, labels

    def _extract_label_from_path(self, file_path: str) -> int:

        label_str = file_path.split('/')[-2].split('_')[0]
        return int(label_str)

    def _segment_and_normalize(self, segment: np.ndarray) -> np.ndarray:

        local_window = len(segment) // self.window_size_factor

        # Split into patches
        patches = [segment[i:i + local_window] for i in range(0, len(segment), local_window)]
        patches = np.abs(np.array(patches))

        # Normalize
        signal_min = np.min(patches)
        signal_max = np.max(patches)

        if signal_max != signal_min:
            normalized_patches = (patches - signal_min) / (signal_max - signal_min)
        else:
            normalized_patches = np.zeros_like(patches)

        return normalized_patches

    @staticmethod
    def windows(data, window_size, overlap):

        start = 0

        while start < len(data):

            yield start, start + window_size
            start += (window_size // overlap)