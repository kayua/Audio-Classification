#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
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
    import os
    import sys

    import logging

    from datetime import datetime

    from logging.handlers import RotatingFileHandler

except ImportError as error:
    print(error)
    sys.exit(-1)

class Logger:
    """
    A Logger class designed to manage and configure logging for an application.
    It supports logging to both console and rotating log files. The log file name is
    dynamically generated based on the current date and time, and it creates backups of
    the log file to prevent the log file from growing too large.

    Attributes:
        _logger (logging.Logger): The main logger object for logging messages.
        _logging_format (str): The format in which log messages are written.
        _rotatingFileHandler (logging.Handler): The handler that writes logs to a rotating file.
        _consoleHandler (logging.Handler): The handler that writes logs to the console.
    """

    def __init__(self, input_arguments):
        """
        Initializes the Logger instance by setting up the logging format and handlers
        based on the input arguments.

        Args:
            input_arguments (argparse.Namespace): Configuration arguments, such as verbosity.
                - input_arguments.verbosity (logging level): The level of logging (e.g., logging.DEBUG, logging.INFO).
        """
        self._logger = logging.getLogger()
        self._set_logging_format(input_arguments.verbosity)
        self._set_logging_handlers(input_arguments)

    def _set_logging_format(self, verbosity):
        """
        Configures the logging format based on the verbosity level.

        Args:
            verbosity (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        """
        logging_format = '%(asctime)s\t***\t%(message)s'

        if verbosity == logging.DEBUG:
            logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

        self._logging_format = logging_format

    def _set_logging_handlers(self, input_arguments):
        """
        Configures the log handlers (RotatingFileHandler and StreamHandler) based on the
        input arguments provided. The handlers control where the logs are output (console or file).

        Args:
            input_arguments (argparse.Namespace): Configuration arguments, such as verbosity.
        """
        LOGGING_FILE_NAME = self._get_log_filename()
        logging_filename = os.path.join(self.get_logs_path(), LOGGING_FILE_NAME)

        self._logger.setLevel(input_arguments.verbosity)

        self._setup_rotating_file_handler(logging_filename, input_arguments.verbosity)
        self._setup_console_handler(input_arguments.verbosity)

    @staticmethod
    def _get_log_filename():
        """
        Generates a log file name based on the current date and time.

        Returns:
            str: The generated log file name in the format 'YYYY-MM-DD_HH-MM-SS.log'.
        """
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'

    def _setup_rotating_file_handler(self, logging_filename, verbosity):
        """
        Configures the RotatingFileHandler for logging to a file with rotation.

        Args:
            logging_filename (str): The file path for the log file.
            verbosity (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self._rotatingFileHandler = RotatingFileHandler(
            filename=logging_filename,
            maxBytes=1000000,  # 1 MB maximum size for the log file
            backupCount=5      # Limits to 5 backup log files
        )
        self._rotatingFileHandler.setLevel(verbosity)
        self._rotatingFileHandler.setFormatter(logging.Formatter(self._logging_format))
        self._logger.addHandler(self._rotatingFileHandler)

    def _setup_console_handler(self, verbosity):
        """
        Configures the StreamHandler for logging to the console.

        Args:
            verbosity (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self._consoleHandler = logging.StreamHandler()
        self._consoleHandler.setLevel(verbosity)
        self._consoleHandler.setFormatter(logging.Formatter(self._logging_format))
        self._logger.addHandler(self._consoleHandler)

    def _clear_existing_handlers(self):
        """
        Clears any existing handlers from the logger before adding new ones.
        This ensures that no duplicate handlers are added.
        """
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

    @staticmethod
    def get_logs_path():
        """
        Returns the path to the logs directory, creating the directory if it doesn't exist.

        Returns:
            str: The path to the logs directory.
        """
        logs_dir = 'Logs'
        os.makedirs(logs_dir, exist_ok=True)  # Creates the logs directory if it doesn't exist
        return logs_dir

    def log_info(self, message):
        """
        Logs an informational message to both the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.info(message)

    def log_debug(self, message):
        """
        Logs a debug message to both the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.debug(message)

    def log_warning(self, message):
        """
        Logs a warning message to both the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.warning(message)

    def log_error(self, message):
        """
        Logs an error message to both the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.error(message)

    def log_critical(self, message):
        """
        Logs a critical message to both the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.critical(message)


def auto_logger(function):
    """
    Decorator to initialize an instance of the Logger class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Logger.
    """

    def wrapper(self, *args, **kwargs):
        self.logger = Logger(self.input_arguments)
        return function(self, *args, **kwargs)

    return wrapper