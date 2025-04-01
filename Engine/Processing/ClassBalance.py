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
    import sys
    import numpy

    from sklearn.utils import resample

except ImportError as error:
    print(error)
    sys.exit(-1)

class ClassBalancer:
    """
    A class responsible for balancing dataset classes by oversampling minority classes.
    This is useful for handling imbalanced classification problems.

    Methods
    -------
    balance(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        Balances the classes by oversampling all classes to match the size of the largest class.
    """

    @staticmethod
    def balance_class(features: numpy.ndarray, labels: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Balances dataset classes by oversampling minority classes so that all classes
        have the same number of samples as the majority class.

        Parameters
        ----------
        features : numpy.ndarray
            A 2D array where each row is a sample and each column is a feature.

        labels : numpy.ndarray
            A 1D array where each element is the class label corresponding to each sample in `features`.

        Returns
        -------
        balanced_features : numpy.ndarray
            A 2D array with balanced samples across all classes.

        balanced_labels : numpy.ndarray
            A 1D array containing the class labels after balancing.
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length.")

        unique_classes = numpy.unique(labels)
        max_samples = max([sum(labels == class_index) for class_index in unique_classes])

        balanced_features, balanced_labels = [], []

        for class_id in unique_classes:
            # Extract features and labels for the current class
            features_class, labels_class = features[labels == class_id], labels[labels == class_id]

            # Resample to match the largest class
            features_class_resampled, labels_class_resampled = resample(
                features_class, labels_class,
                replace=True,           # Oversampling with replacement
                n_samples=max_samples,  # Target size: same as majority class
                random_state=0          # For reproducibility
            )

            balanced_features.append(features_class_resampled)
            balanced_labels.append(labels_class_resampled)

        # Combine balanced data from all classes
        balanced_features, balanced_labels = numpy.vstack(balanced_features), numpy.hstack(balanced_labels)

        return balanced_features, balanced_labels