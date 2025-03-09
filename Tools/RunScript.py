#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/26'
__credits__ = ['unknown']


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