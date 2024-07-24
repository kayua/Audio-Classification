import glob
import os
import librosa
import librosa.display
import numpy
from keras.utils import to_categorical
from tqdm import tqdm

# Default values
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_HOP_LENGTH = 256
DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_NUMBER_FILTERS = 512
DEFAULT_SIZE_WINDOW = 1024
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_SCALE_NORMALIZATION = 80
DEFAULT_OVERLAP_SIZE = 2


class AudioFeatureExtractor:
    """
    A class used to extract audio features from audio files for machine learning applications.

    Attributes
    ----------
    sample_rate : int
        Sample rate for loading audio files
    hop_length : int
        Hop length for the STFT
    window_size_factor : int
        Factor to determine window size for segmenting audio
    number_filters : int
        Number of Mel filters to use in the Mel spectrogram
    scale_normalization : int
        Scale for normalizing the log spectrogram
    size_window : int
        Size of the FFT window
    overlap_size : int
        Factor to determine overlap size for windowing
    file_extension : str
        File extension for audio files to be processed
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, hop_length=DEFAULT_HOP_LENGTH,
                 window_size_factor=DEFAULT_WINDOW_SIZE_FACTOR, number_filters=DEFAULT_NUMBER_FILTERS,
                 scale_normalization=DEFAULT_SCALE_NORMALIZATION, size_window=DEFAULT_SIZE_WINDOW,
                 overlap_size=DEFAULT_OVERLAP_SIZE, file_extension=DEFAULT_FILE_EXTENSION):
        """
        Initializes the AudioFeatureExtractor with given or default parameters.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_size_factor = window_size_factor
        self.window_size = hop_length * (self.window_size_factor - 1)
        self.number_filters = number_filters
        self.scale_normalization = scale_normalization
        self.size_window = size_window
        self.overlap_size = overlap_size
        self.file_extension = file_extension

    @staticmethod
    def windows(data, window_size, overlap):
        """
        Generates start and end indices for segmenting the audio data into overlapping windows.
        """
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // overlap)

    def extract_features_bidimensional(self, sub_dirs):
        """
        Extracts bidimensional Mel spectrogram features from audio files in the specified directories.

        Parameters
        ----------
        sub_dirs : list
            List of directories containing audio files

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        array_features, array_labels = [], []

        for _, sub_directories in tqdm(enumerate(sub_dirs)):
            for file_name in glob.glob(os.path.join(sub_directories, self.file_extension)):
                sound_clip, _ = librosa.load(file_name, sr=self.sample_rate)
                label = file_name.split('/')[-2].split('_')[0]

                for (start, end) in self.windows(sound_clip, self.window_size, self.overlap_size):
                    if len(sound_clip[start:end]) == self.window_size:
                        signal = sound_clip[start:end]
                        spectrogram = librosa.feature.melspectrogram(y=signal, n_mels=self.number_filters,
                                                                     sr=self.sample_rate,
                                                                     n_fft=self.size_window, hop_length=self.hop_length)
                        log_spectrogram = librosa.power_to_db(spectrogram, ref=numpy.max)
                        log_spectrogram = (log_spectrogram / self.scale_normalization) + 1
                        array_features.append(log_spectrogram)
                        array_labels.append(label)

        array_features = numpy.asarray(array_features).reshape(len(array_features), self.number_filters,
                                                               self.window_size_factor, 1)
        array_labels = to_categorical(numpy.array(array_labels, dtype=numpy.float32))

        return numpy.array(array_features, dtype=numpy.float32), array_labels

    def extract_features_unidimensional(self, sub_directories):
        """
        Extracts unidimensional audio features from audio files in the specified directories.

        Parameters
        ----------
        sub_directories : list
            List of directories containing audio files

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        list_features, list_labels = [], []

        for _, sub_directories in tqdm(enumerate(sub_directories)):
            for file_name in glob.glob(os.path.join(sub_directories, self.file_extension)):
                sound_clip, _ = librosa.load(file_name, sr=self.sample_rate)
                label = file_name.split('/')[-2].split('_')[0]

                for (start, end) in self.windows(sound_clip, self.window_size, self.overlap_size):
                    if len(sound_clip[start:end]) == self.window_size:
                        signal = sound_clip[start:end]
                        list_features.append(signal)
                        list_labels.append(label)

        array_features = numpy.asarray(list_features).reshape(len(list_features), self.number_filters,
                                                              self.window_size_factor, 1)
        array_labels = to_categorical(numpy.array(list_labels, dtype=numpy.float32))

        return numpy.array(array_features, dtype=numpy.float32), array_labels

    def extract_bidimensional_dataset(self, sub_dirs):
        """
        Extracts bidimensional dataset from audio files in the specified directories.

        Parameters
        ----------
        sub_dirs : list
            List of directories containing audio files

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        return self.extract_features_bidimensional(sub_dirs)

    def extract_unidimensional_dataset(self, sub_dirs):
        """
        Extracts unidimensional dataset from audio files in the specified directories.

        Parameters
        ----------
        sub_dirs : list
            List of directories containing audio files

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        return self.extract_features_unidimensional(sub_dirs)

    def extract_unidimensional_feature(self, audio_file):
        """
        Extracts unidimensional features from a single audio file.

        Parameters
        ----------
        audio_file : str
            Path to the audio file

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        sound_clip, _ = librosa.load(audio_file, sr=self.sample_rate)
        label = os.path.basename(os.path.dirname(audio_file))
        number_segments = len(sound_clip) // self.window_size
        list_features = []

        for i in range(number_segments):
            start = i * self.window_size
            end = start + self.window_size
            signal = sound_clip[start:end]
            list_features.append(signal)

        features = numpy.asarray(list_features)
        array_labels = numpy.array([label] * number_segments)
        array_feature = list(features.shape)
        array_feature[1] += 1
        padding_array_features = numpy.zeros(array_feature)
        padding_array_features[:, :self.number_filters, :, :] = features

        return padding_array_features.astype(numpy.float32), array_labels

    def extract_bidimensional_feature(self, audio_file):
        """
        Extracts bidimensional features from a single audio file.

        Parameters
        ----------
        audio_file : str
            Path to the audio file

        Returns
        -------
        tuple
            Numpy arrays of features and labels
        """
        sound_clip, _ = librosa.load(audio_file, sr=self.sample_rate)
        labels = os.path.basename(os.path.dirname(audio_file))
        number_segments = len(sound_clip) // self.window_size
        list_features = []

        for i in range(number_segments):
            start = i * self.window_size
            end = start + self.window_size
            signal = sound_clip[start:end]
            spec = librosa.feature.melspectrogram(y=signal, n_mels=self.number_filters, sr=self.sample_rate,
                                                  n_fft=self.size_window, hop_length=self.hop_length)
            spectrogram = librosa.power_to_db(spec, ref=numpy.max)
            spectrogram = (spectrogram / self.scale_normalization) + 1
            list_features.append(spectrogram)

        features = numpy.asarray(list_features).reshape(number_segments, self.number_filters, self.window_size_factor,
                                                        1)
        array_labels = numpy.array([labels] * number_segments)
        array_features = list(features.shape)
        array_features[1] += 1
        padding_array_features = numpy.zeros(array_features)
        padding_array_features[:, :self.number_filters, :, :] = features

        return padding_array_features.astype(numpy.float32), array_labels

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def hop_length(self):
        return self._hop_length

    @hop_length.setter
    def hop_length(self, value):
        self._hop_length = value
        self._window_size = self._hop_length * (self._window_size_factor - 1)

    @property
    def window_size_factor(self):
        return self._window_size_factor

    @window_size_factor.setter
    def window_size_factor(self, value):
        self._window_size_factor = value
        self._window_size = self._hop_length * (self._window_size_factor - 1)

    @property
    def window_size(self):
        return self._window_size

    @property
    def number_filters(self):
        return self._number_filters

    @number_filters.setter
    def number_filters(self, value):
        self._number_filters = value

    @property
    def scale_normalization(self):
        return self._scale_normalization

    @scale_normalization.setter
    def scale_normalization(self, value):
        self._scale_normalization = value

    @property
    def size_window(self):
        return self._size_window

    @size_window.setter
    def size_window(self, value):
        self._size_window = value

    @property
    def overlap_size(self):
        return self._overlap_size

    @overlap_size.setter
    def overlap_size(self, value):
        self._overlap_size = value

    @property
    def file_extension(self):
        return self._file_extension

    @file_extension.setter
    def file_extension(self, value):
        self._file_extension = value

    @window_size.setter
    def window_size(self, value):
        self._window_size = value
