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
    import time
    import json
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.callbacks import Callback

except ImportError as error:
    print(error)
    sys.exit(-1)


class ModelMonitorCallback(Callback):
    """
    A custom callback for monitoring and logging the training process of a model.

    This callback tracks the time taken for each epoch and each batch, logs metrics, and
    estimates the remaining time for training. It saves the collected information into a
    JSON file at the end of training.

    Parameters:
    - filename (str): Path to the directory where the training log file will be saved.
      The log file will be named with the format: `monitor_model_<k_fold>_fold.json`.
    - k_fold (int): The fold index used in k-fold cross-validation. This is used to
      differentiate the logs of different folds.
    """

    def __init__(self, filename, k_fold):
        """
        Initializes the ModelMonitorCallback with the file path for saving logs and
        the fold index for k-fold cross-validation.

        Parameters:
        - filename (str): Directory where the training logs will be saved.
        - k_fold (int): The fold index for cross-validation.
        """
        super(ModelMonitorCallback, self).__init__()

        # Create a log filename using the provided path and fold index.
        self.filename = '{}/monitor_model_{}_fold.json'.format(filename, k_fold)

        # Variables to track times of epochs and batches
        self.batch_times = None
        self.epoch_start_time = None
        self.batch_start_time = None

        # Data structure to store training metrics and times
        self.data = {
            "start_time": None,
            "elapsed_time": None,
            "expected_finish": None,
            "epochs": [],  # List to store details of each epoch
            "avg_epoch_time": [],  # List to store average time per epoch
            "avg_batch_time": [],  # List to store average time per batch
            "total_time": None,
            "status": "Normal"  # The status of the training (default is 'Normal')
        }

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of the training. Initializes the start time and resets
        the batch times.
        """
        self.data["start_time"] = time.time()  # Store the start time of training
        self.batch_times = []  # Initialize the batch time list
        logging.info("Training started...")
        print("Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch. Records the start time of the epoch.

        Parameters:
        - epoch (int): The index of the current epoch.
        """
        self.epoch_start_time = time.time()  # Record the start time of the current epoch
        logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} started...")
        print(f"Epoch {epoch + 1}/{self.params['epochs']} started...")

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each batch. Records the start time of the batch.

        Parameters:
        - batch (int): The index of the current batch.
        """
        self.batch_start_time = time.time()  # Record the start time of the current batch
        logging.debug(f"Batch {batch} started...")

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch. Calculates the batch time and appends it to the list.

        Parameters:
        - batch (int): The index of the current batch.
        """
        # Calculate the time taken for the batch and append it to the list
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        logging.debug(f"Batch {batch} ended. Time taken: {batch_time:.4f}s")

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Calculates the epoch time, average batch time,
        and other metrics, then updates the stored data.

        Parameters:
        - epoch (int): The index of the current epoch.
        - logs (dict): A dictionary containing the metrics computed for the epoch (e.g., loss, accuracy).
        """
        # If logs are not provided, initialize as empty dictionary
        if logs is None:
            logs = {}

        # If epoch start time is not set, use the current time
        if not hasattr(self, "epoch_start_time") or self.epoch_start_time is None:
            logging.warning("Epoch start time not set. Using current time.")
            self.epoch_start_time = time.time()

        # Calculate the time taken for the epoch
        epoch_time = time.time() - self.epoch_start_time

        # Calculate the average batch time for the epoch
        avg_batch_time = numpy.mean(self.batch_times) if self.batch_times else 0.0

        # Append the epoch data (time, avg_batch_time, metrics) to the data structure
        self.data["epochs"].append({
            "epoch": epoch + 1,
            "time": epoch_time,
            "avg_batch_time": avg_batch_time,
            "metrics": logs
        })
        self.batch_times = []  # Reset the batch times list

        # Calculate elapsed time from the start of training
        elapsed_time = time.time() - self.data.get("start_time", time.time())
        self.data["elapsed_time"] = elapsed_time

        # Calculate the average time per epoch
        avg_epoch_time = numpy.mean([e["time"] for e in self.data["epochs"]]) if self.data["epochs"] else epoch_time

        # Calculate the remaining epochs
        remaining_epochs = max(self.params["epochs"] - (epoch + 1), 0)

        # Estimate the time for the training to finish
        expected_finish = time.time() + remaining_epochs * avg_epoch_time
        self.data["expected_finish"] = expected_finish

        # Log the results for the current epoch
        logging.info(
            f"Epoch {epoch + 1} ended. Time: {epoch_time:.4f}s, "
            f"Avg batch time: {avg_batch_time:.4f}s, "
            f"Expected finish: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_finish))}"
        )

    def on_train_end(self, logs=None):
        """
        Called at the end of the training. Calculates the total training time and average times
        for epochs and batches. The final training log is saved to a JSON file.

        Parameters:
        - logs (dict): The final logs recorded during the training.
        """
        # Calculate the total training time
        total_time = time.time() - self.data["start_time"]

        # Calculate the average epoch time
        avg_epoch_time = numpy.mean([epoch["time"] for epoch in self.data["epochs"]])

        # Calculate the average batch time
        avg_batch_time = numpy.mean([epoch["avg_batch_time"] for epoch in self.data["epochs"]])

        # Update the data dictionary with final training times
        self.data["total_time"] = total_time
        self.data["avg_epoch_time"] = avg_epoch_time
        self.data["avg_batch_time"] = avg_batch_time

        # Log the total training time and average epoch time
        logging.info(f"Training ended. Total time: {total_time:.4f}s, Avg epoch time: {avg_epoch_time:.4f}s")

        # Save the data to a JSON file
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

        # Log that the training logs have been saved
        logging.info(f"Training logs saved to {self.filename}")