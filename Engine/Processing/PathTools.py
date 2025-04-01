#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
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

    import logging

    from tqdm import tqdm

except ImportError as error:
    print(error)
    sys.exit(-1)


class PathTools:

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
