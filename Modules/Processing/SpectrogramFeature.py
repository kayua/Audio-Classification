import glob
import logging
import os

import librosa
import numpy
from tqdm import tqdm


def load_data(self, sub_directories: str = None, file_extension: str = None, stack_segments = False) -> tuple:

    logging.info("Starting to load data...")
    list_spectrogram, list_labels, list_class_path = [], [], []
    file_extension = file_extension or self.file_extension

    # Check if directory exists
    if not os.path.exists(sub_directories):
        logging.error(f"Directory '{sub_directories}' does not exist.")
        return None, None

    # Collect all class directories
    logging.info(f"Reading subdirectories in '{sub_directories}'...")
    for class_dir in os.listdir(sub_directories):

        class_path = os.path.join(sub_directories, class_dir)

        if os.path.isdir(class_path):
            list_class_path.append(class_path)

    logging.info(f"Found {len(list_class_path)} class directories.")

    # Process each audio file in subdirectories
    for sub_directory in list_class_path:
        logging.info(f"Processing class directory: {sub_directory}...")

        for file_name in tqdm(glob.glob(os.path.join(sub_directory, file_extension))):

            try:

                signal, _ = librosa.load(file_name, sr=self.sample_rate)
                label = file_name.split('/')[-2].split('_')[0]

                for (start, end) in self.windows(signal, self.window_size, self.overlap):

                    if len(signal[start:end]) == self.window_size:
                        signal_window = signal[start:end]

                        # Generate mel spectrogram
                        spectrogram = librosa.feature.melspectrogram(
                            y=signal_window,
                            n_mels=self.number_filters_spectrogram,
                            sr=self.sample_rate,
                            n_fft=self.window_size_fft,
                            hop_length=self.hop_length
                        )

                        # Convert spectrogram to decibels
                        spectrogram_decibel_scale = librosa.power_to_db(spectrogram, ref=numpy.max)
                        spectrogram_decibel_scale = (spectrogram_decibel_scale / self.decibel_scale_factor) + 1

                        # Append spectrogram and label
                        list_spectrogram.append(spectrogram_decibel_scale)
                        list_labels.append(label)

            except Exception as e:
                logging.error(f"Error processing file '{file_name}': {e}")

    # For Residual Model
    if stack_segments:

        array_features = numpy.array(list_spectrogram).reshape(len(list_spectrogram),
                                                           self.number_filters_spectrogram,
                                                           self.window_size_factor, 1)
        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info("Reshaping feature array.")
        new_shape = list(array_features.shape)
        new_shape[1] += 1
        new_array = numpy.zeros(new_shape)
        new_array[:, :self.number_filters_spectrogram, :, :] = array_features

        logging.info("Data loading complete.")
        return numpy.array(new_array, dtype=numpy.float32), array_labels

    else:

        array_features = numpy.array(list_spectrogram).reshape(
            len(list_spectrogram),
            self.number_filters_spectrogram,
            self.window_size // self.hop_length,
            1
        )

        array_labels = numpy.array(list_labels, dtype=numpy.int32)

        logging.info(f"Loaded {len(array_features)} spectrogram features.")
        logging.info("Data loading complete.")

        return numpy.array(array_features, dtype=numpy.float32), array_labels

