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

    import logging
    import subprocess

except ImportError as error:
    print(error)
    sys.exit(-1)

class RunScript:

    @staticmethod
    def run_python_script(*args) -> int:
        """
        Executes a Python script with the specified arguments and logs the output.

        Args:
            *args: Additional arguments to pass to the script.

        Returns:
            int: The return code of the executed script.
        """
        logging.info("Starting the execution of 'GeneratePDF.py' with arguments: %s", args)

        try:
            # Prepare the command to run the script
            command = ['python3', 'GeneratePDF.py'] + list(args)
            logging.info("Command to be executed: %s", command)

            # Execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Log standard output
            logging.info("Standard Output:\n%s", result.stdout)

            # Log standard error if it exists
            if result.stderr:
                logging.warning("Standard Error:\n%s", result.stderr)

            return result.returncode

        except subprocess.CalledProcessError as e:

            logging.error("An error occurred while executing the script: %s", e)
            logging.error("Standard Output:\n%s", e.stdout)
            logging.error("Standard Error:\n%s", e.stderr)
            return e.returncode

        finally:

            logging.info("Execution of 'GeneratePDF.py' completed.")