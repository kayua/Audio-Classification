#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import sys
    import numpy

    import librosa

    import librosa.display

    import matplotlib.pyplot as plt

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_SAMPLE_RATE = 8000
DEFAULT_MAX_FREQUENCY = 4000
DEFAULT_SIZE_FIGURE = (10, 6)
DEFAULT_COLOR_MAP = 'viridis'
DEFAULT_SPECTROGRAM_TITLE = "Spectrogram"
DEFAULT_SAVE_PATH = 'spectrogram.png'


class AudioSpectrogramViewer:
    """
    A class to visualize and save the spectrogram of a WAV audio file using librosa.
    """

    def __init__(self, audio_file, sample_rate=DEFAULT_SAMPLE_RATE, max_frequency=DEFAULT_MAX_FREQUENCY,
                 figure_size=DEFAULT_SIZE_FIGURE, colormap=DEFAULT_COLOR_MAP, title=DEFAULT_SPECTROGRAM_TITLE,
                 save_path=DEFAULT_SAVE_PATH):
        """
        Initialize with the path to the WAV audio file and optional parameters.

        Parameters:
        -----------
            @audio_file (str): Path to the WAV audio file.
            @sr (int, optional): Sampling rate of the audio file (default is 8000).
            @fmax (int, optional): Maximum frequency to display in the spectrogram (default is 4000 Hz).
            @figsize (tuple, optional): Size of the figure (width, height) in inches (default is (10, 6)).
            @colormap (str, optional): Colormap to use for the spectrogram plot (default is 'viridis').
            @title (str or None, optional): Title of the spectrogram plot (default is None).
            @save_path (str, optional): Path to save the spectrogram image (default is 'spectrogram.png').
        """
        self.audio_file = audio_file
        self.sample_rate = sample_rate
        self.max_frequency = max_frequency
        self.figure_size = figure_size
        self.colormap = colormap
        self.title = title
        self.save_path = save_path

    def plot_spectrogram(self):
        """
        Plot and save the spectrogram of the audio file with optional parameters.
        """
        # Load audio file with librosa
        signal, sample_rate = librosa.load(self.audio_file, sr=self.sample_rate)

        # Compute spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = librosa.stft(signal)

        # Convert magnitude spectrogram to decibels (dB)
        spectrogram_decibel_scale = librosa.amplitude_to_db(numpy.abs(spectrogram), ref=numpy.max)

        # Create a figure and plot the spectrogram
        plt.figure(figsize=self.figure_size)
        librosa.display.specshow(spectrogram_decibel_scale, sr=sample_rate, x_axis='time', y_axis='linear',
                                 fmax=self.max_frequency, cmap=self.colormap)
        plt.colorbar(format='%+2.0f dB')
        if self.title:
            plt.title(self.title)
        else:
            plt.title('Spectrogram of {}'.format(self.audio_file))
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig(self.save_path)
        plt.close()

    # Setters
    def set_audio_file(self, audio_file):
        self.audio_file = audio_file

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def set_max_frequency(self, max_frequency):
        self.max_frequency = max_frequency

    def set_figure_size(self, figure_size):
        self.figure_size = figure_size

    def set_colormap(self, colormap):
        self.colormap = colormap

    def set_title(self, title):
        self.title = title

    def set_save_path(self, save_path):
        self.save_path = save_path

    # Getters
    def get_audio_file(self):
        return self.audio_file

    def get_sample_rate(self):
        return self.sample_rate

    def get_max_frequency(self):
        return self.max_frequency

    def get_figure_size(self):
        return self.figure_size

    def get_colormap(self):
        return self.colormap

    def get_title(self):
        return self.title

    def get_save_path(self):
        return self.save_path