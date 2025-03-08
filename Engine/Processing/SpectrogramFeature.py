import glob
import logging
import os
import numpy
import librosa
from tqdm import tqdm


class SpectrogramFeature:
    def __init__(self, sample_rate=22050, window_size=2048, overlap=2,
                 number_filters_spectrogram=128, window_size_fft=2048, hop_length=512, decibel_scale_factor=80):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.number_filters_spectrogram = number_filters_spectrogram
        self.window_size_fft = window_size_fft
        self.hop_length = hop_length
        self.decibel_scale_factor = decibel_scale_factor

    def load_data(self, sub_directories: str, file_extension: str, stack_segments=False):
        logging.info("Starting to load data...")
        file_extension = file_extension or "*.wav"

        if not os.path.exists(sub_directories):
            logging.error(f"Directory '{sub_directories}' does not exist.")
            return None, None

        class_paths = self._get_class_paths(sub_directories)
        list_all_spectrogram, list_all_labels = self._process_files(class_paths, file_extension)

        return self._prepare_output(list_all_spectrogram, list_all_labels, stack_segments)

    @staticmethod
    def _get_class_paths(sub_directories):
        logging.info(f"Reading subdirectories in '{sub_directories}'...")
        class_paths = [os.path.join(sub_directories, d) for d in os.listdir(sub_directories) if
                       os.path.isdir(os.path.join(sub_directories, d))]
        logging.info(f"Found {len(class_paths)} class directories.")
        return class_paths

    def _process_files(self, class_paths, file_extension):
        list_spectrogram, labels = [], []

        for class_path in class_paths:
            logging.info(f"Processing class directory: {class_path}...")
            for file_name in tqdm(glob.glob(os.path.join(class_path, file_extension))):
                try:
                    signal, label = self._load_audio_and_extract_label(file_name)
                    list_spectrogram.extend(self._extract_spectrogram(signal))
                    labels.extend([label] * len(list_spectrogram))
                except Exception as e:
                    logging.error(f"Error processing file '{file_name}': {e}")

        return list_spectrogram, labels

    def _load_audio_and_extract_label(self, file_name):
        raw_signal, _ = librosa.load(file_name, sr=self.sample_rate)
        label = os.path.basename(os.path.dirname(file_name)).split('_')[0]
        return raw_signal, label

    def _extract_spectrogram(self, signal):
        list_spectrogram = []
        for start, end in self._generate_windows(len(signal)):
            if len(signal[start:end]) == self.window_size:
                list_spectrogram.append(self._generate_mel_spectrogram(signal[start:end]))
        return list_spectrogram

    def _generate_mel_spectrogram(self, signal_window):
        spectrogram = librosa.feature.melspectrogram(
            y=signal_window,
            sr=self.sample_rate,
            n_mels=self.number_filters_spectrogram,
            n_fft=self.window_size_fft,
            hop_length=self.hop_length
        )
        return (librosa.power_to_db(spectrogram, ref=numpy.max) / self.decibel_scale_factor) + 1

    def _prepare_output(self, spectrograms, labels, stack_segments):
        array_features = numpy.array(spectrograms).reshape(
            len(spectrograms),
            self.number_filters_spectrogram,
            self.window_size // self.hop_length,
            1
        )
        array_labels = numpy.array(labels, dtype=numpy.int32)

        if stack_segments:
            new_shape = list(array_features.shape)
            new_shape[1] += 1
            stacked_features = numpy.zeros(new_shape)
            stacked_features[:, :self.number_filters_spectrogram, :, :] = array_features
            return stacked_features.astype(numpy.float32), array_labels

        return array_features.astype(numpy.float32), array_labels