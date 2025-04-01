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
    import json

    import numpy
    import psutil
    import logging
    import platform

    import subprocess
    import tensorflow

    from tensorflow.keras.callbacks import Callback

except ImportError as error:
    print(error)
    sys.exit(-1)


class ResourceMonitorCallback(Callback):
    """
    This class is a custom callback for monitoring system resources during the training of a
    machine learning model. It collects and logs various system metrics (CPU, memory, disk
    usage, GPU, etc.) at regular intervals during training. The data is saved into a JSON file
    at the specified location for further analysis.

    Attributes:
        @json_file_path (str): Path to the JSON file where system resource data will be saved.
        @k_fold (int): The fold number for k-fold validation, used to create a unique file for each fold.
        @save_interval (int): Number of epochs after which resource data will be saved.
        @machine_info (dict): A dictionary containing the machine's hardware and software information.
        @epoch_counter (int): Counter for the number of completed epochs.

    Methods:
        __init__(json_file_path, k_fold, save_interval=1): Initializes the callback with the path to
         the JSON file, the fold number, and the save interval.
        _get_machine_info(): Gathers initial machine information like platform, TensorFlow version,
         CPU, memory, disk info, etc.
        _get_cpu_info(): Retrieves information about the CPU using the 'lscpu' command.
        _get_memory_info(): Retrieves memory usage information using the 'free -h' command.
        _get_disk_info(): Retrieves disk usage information using the 'df -h' command.
        on_train_begin(logs=None): Called at the start of training to save initial machine info to
         the JSON file.
        on_epoch_end(epoch, logs=None): Called at the end of each epoch to collect and save resource
         data if the save interval is met.
        _get_gpu_info(): Retrieves GPU utilization and memory usage using the 'nvidia-smi' command.
        _get_system_temperatures(): Retrieves CPU and GPU temperature information using the 'sensors'
         and 'nvidia-smi' commands.
        _get_network_info(): Retrieves network bandwidth and transfer rate using the psutil library.

    Example:
        >>> python
        ... # Initialize the callback
        ... resource_monitor = ResourceMonitorCallback(json_file_path='/path/to/save', k_fold=1, save_interval=5)
        ...# Use it in the training loop
        ...model.fit(X_train, y_train, epochs=50, callbacks=[resource_monitor])
        >>>
    """

    def __init__(self, json_file_path, k_fold, save_interval=1):
        super(ResourceMonitorCallback, self).__init__()
        """
        Initializes the ResourceMonitorCallback with the path to the JSON file where data will be saved,
        the k-fold number for the experiment, and the interval at which the resource data will be saved.

        Args:
            json_file_path (str): Path where the system resource data will be stored.
            k_fold (int): The fold number for k-fold validation.
            save_interval (int, optional): The number of epochs after which resource data will be saved. Defaults to 1.
        """
        self.json_file_path = '{}/monitor_resource_{}_fold.json'.format(json_file_path, k_fold)
        self.save_interval = save_interval
        self.machine_info = self._get_machine_info()
        self.epoch_counter = 0

    def _get_machine_info(self):
        """
        Gathers initial machine information such as platform, TensorFlow version, CPU, memory, disk usage, etc.

        Returns:
            dict: A dictionary containing system information including platform, versions, CPU, memory, disk info, etc.
        """
        try:
            machine_info = {
                "platform": platform.platform(),
                "tensorflow_version": tensorflow.__version__,
                "numpy_version": numpy.__version__,
                "python_version": platform.python_version(),
                "cpu_info": self._get_cpu_info(),
                "memory_info": self._get_memory_info(),
                "disk_info": self._get_disk_info(),
                "epoch": [],
                "cpu_percent": {"mean": [], "per_core": []},
                "ram_usage": [],
                "swap_usage": [],
                "storage_available": [],
                "cpu_temperature": {"mean": [], "per_core": []},
                "gpu_temperature": [],
                "network_bandwidth": [],
                "network_transfer_rate": []
            }

            return machine_info

        except Exception as e:

            logging.error(f'Failed to collect machine info: {e}')
            return {}

    @staticmethod
    def _get_cpu_info():
        """
        Retrieves CPU information using the 'lscpu' command.

        Returns:
            str: CPU information as a string.
        """
        try:

            cpu_info = subprocess.check_output(['lscpu']).decode()
            logging.info('CPU info collected.')
            return cpu_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:

            logging.error(f'Error collecting CPU info: {e}')
            return "Unknown CPU info"

    @staticmethod
    def _get_memory_info():
        """
        Retrieves memory usage information using the 'free -h' command.

        Returns:
            str: Memory usage information as a string.
        """
        try:

            memory_info = subprocess.check_output(['free', '-h']).decode()
            logging.debug('Memory info collected.')
            return memory_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:

            logging.error(f'Error collecting memory info: {e}')
            return "Unknown memory info"

    @staticmethod
    def _get_disk_info():
        """
        Retrieves disk usage information using the 'df -h' command.

        Returns:
            str: Disk usage information as a string.
        """
        try:
            disk_info = subprocess.check_output(['df', '-h']).decode()
            logging.debug('Disk info collected.')
            return disk_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:

            logging.error(f'Error collecting disk info: {e}')
            return "Unknown disk info"

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training to save initial machine information to a JSON file.

        Args:
            logs (dict, optional): Training logs, unused in this method.
        """
        try:

            with open(self.json_file_path, 'w') as json_file:
                json.dump(self.machine_info, json_file, indent=4)
                json_file.write('\n')

            logging.info('Training started and initial machine info saved.')

        except IOError as e:
            logging.error(f'Error writing to JSON file at training start: {e}')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to collect and save system resource data if the save interval is met.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Training logs, unused in this method.
        """
        self.epoch_counter += 1
        logging.info(f'Epoch {epoch} ended. Logs: {logs}')

        if self.epoch_counter % self.save_interval == 0:

            try:

                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                cpu_percent_mean = psutil.cpu_percent(interval=None)
                ram_usage = psutil.virtual_memory().percent
                swap_usage = psutil.swap_memory().percent
                storage_available = psutil.disk_usage('/').percent

                data = {
                    "epoch": epoch,
                    "cpu_percent": {"mean": cpu_percent_mean, "per_core": cpu_percent},
                    "ram_usage": ram_usage,
                    "swap_usage": swap_usage,
                    "storage_available": storage_available
                }

                with open(self.json_file_path, 'r+') as json_file:

                    try:
                        data_list = json.load(json_file)

                    except json.JSONDecodeError as e:
                        logging.error(f'Error loading JSON file: {e}')
                        data_list = self.machine_info  # Initialize with machine info

                    for key, value in data.items():

                        if isinstance(value, dict):

                            for subkey, subvalue in value.items():
                                data_list[key][subkey].append(subvalue)

                        else:
                            data_list[key].append(value)

                    json_file.seek(0)
                    json.dump(data_list, json_file, indent=4)
                    json_file.truncate()

                logging.info(f'Resource data for epoch {epoch} saved successfully.')

            except IOError as e:
                logging.error(f'Error writing resource data for epoch {epoch} to JSON: {e}')

    @staticmethod
    def _get_gpu_info():
        """
        Retrieves GPU information such as utilization and memory usage using the 'nvidia-smi' command.

        Returns:
            tuple: A tuple containing GPU utilization and memory usage.
        """
        try:

            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'])

            gpu_data = result.decode('utf-8').strip().split('\n')
            gpu_utilization = int(gpu_data[0].split(',')[0])
            gpu_memory_usage = int(gpu_data[0].split(',')[1])
            logging.debug('GPU info collected.')
            return gpu_utilization, gpu_memory_usage

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting GPU info: {e}')
            return None, None

    @staticmethod
    def _get_system_temperatures():
        """
        Retrieves system temperatures including CPU and GPU temperatures using the 'sensors'
        and 'nvidia-smi' commands.

        Returns:
            tuple: A tuple containing CPU temperature (mean and per-core) and GPU temperature.
        """
        try:

            cpu_temp_result = subprocess.check_output(['sensors'])
            cpu_temp_data = cpu_temp_result.decode('utf-8').split('\n')
            cpu_temps = [line.split(':')[-1].strip().split()[0] for line in cpu_temp_data if 'Core' in line]
            cpu_temperature = {
                "mean": numpy.mean([float(temp.replace('°C', '')) for temp in cpu_temps]),
                "per_core": [float(temp.replace('°C', '')) for temp in cpu_temps]
            }

            logging.debug('CPU temperature info collected.')

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting CPU temperature info: {e}')
            cpu_temperature = {"mean": None, "per_core": None}

        try:

            gpu_temp_result = subprocess.check_output(['nvidia-smi',
                                                       '--query-gpu=temperature.gpu',
                                                       '--format=csv,noheader,nounits'])

            gpu_temperature = int(gpu_temp_result.decode('utf-8').strip())
            logging.debug('GPU temperature info collected.')

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting GPU temperature info: {e}')
            gpu_temperature = None

        return cpu_temperature, gpu_temperature

    @staticmethod
    def _get_network_info():
        """
        Retrieves network bandwidth and transfer rate using the psutil library.

        Returns:
            tuple: A tuple containing the total bandwidth (bytes) and the transfer rate (in MB).
        """
        try:

            network_stats = psutil.net_io_counters()
            bandwidth = network_stats.bytes_sent + network_stats.bytes_recv
            transfer_rate = network_stats.bytes_sent / (1024 * 1024)  # In MB
            logging.debug('Network info collected.')
            return bandwidth, transfer_rate

        except Exception as e:

            logging.error(f'Error collecting network info: {e}')
            return None, None
