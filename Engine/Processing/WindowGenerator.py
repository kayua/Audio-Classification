import numpy


class WindowGenerator:
    """
    A class for generating windowed segments from sequential data.
    This is particularly useful for splitting time-series data or audio signals into overlapping windows.

    Attributes
    ----------
    window_size : int
        The fixed size of each window.

    overlap : int
        The factor controlling overlap between consecutive windows.
        For example, overlap=2 results in 50% overlap, overlap=4 results in 75% overlap.

    Methods
    -------
    generate_windows(data: np.ndarray) -> Generator[Tuple[int, int], None, None]
        Generates start and end indices for each window in the provided data.
    """

    def __init__(self, window_size: int, overlap: int):
        """
        Initializes the WindowGenerator with a given window size and overlap factor.

        Parameters
        ----------
        window_size : int
            The size of each window.

        overlap : int
            The overlap factor, controlling how much consecutive windows overlap.
            For example, overlap=2 means 50% overlap, overlap=4 means 75% overlap.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if overlap <= 0:
            raise ValueError("overlap must be a positive integer.")

        self.window_size = window_size
        self.overlap = overlap

    def generate_windows(self, data):
        """
        Generates the start and end indices for each window over the input data.

        Parameters
        ----------
        data : numpy.ndarray
            The input data array, typically a 1D array like audio samples or time-series data.

        Yields
        ------
        tuple
            A tuple containing the (start, end) indices for each window.
        """
        if not isinstance(data, numpy.ndarray):
            raise ValueError("Input data must be a numpy.ndarray.")

        start = 0
        step_size = self.window_size // self.overlap

        while start < len(data):
            yield start, start + self.window_size
            start += step_size