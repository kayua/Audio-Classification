import numpy
from sklearn.utils import resample

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
    def balance(features: numpy.ndarray, labels: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        max_samples = max([sum(labels == c) for c in unique_classes])

        balanced_features = []
        balanced_labels = []

        for c in unique_classes:
            # Extract features and labels for the current class
            features_class = features[labels == c]
            labels_class = labels[labels == c]

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
        balanced_features = numpy.vstack(balanced_features)
        balanced_labels = numpy.hstack(balanced_labels)

        return balanced_features, balanced_labels