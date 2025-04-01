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
# Copyright (c) 2025 unknown
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

except ImportError as error:
    print(error)
    sys.exit(-1)


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