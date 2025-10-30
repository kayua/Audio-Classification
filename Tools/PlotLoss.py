#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{1}'
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
    import json
    import logging
    from pathlib import Path
    import matplotlib.pyplot as plt

except ImportError as error:
    print(error)
    sys.exit(-1)


class PlotLossCurve:
    """
    This class is responsible for plotting and saving training and validation loss curves
    for different machine learning models. It allows extensive customization of the plot
    through various parameters passed during initialization.

    Features:
        - Increased font sizes by 50% for better readability
        - Save plot data to JSON for reproducibility
        - Load and plot from JSON files
        - Comprehensive data persistence

    Args:
        @history_dict_list (list): A list of dictionaries where each dictionary contains the 'Name'
         of a model and the 'History' dictionary with loss values.
        @path_output (str): The directory path where the loss plot images should be saved.
        @figsize (tuple): The size of the plot figure (width, height). Default is (10, 6).
        @training_loss_color (str): The color of the training loss curve. Default is "blue".
        @validation_loss_color (str): The color of the validation loss curve. Default is "orange".
        @title_fontsize (int): Font size of the plot title. Automatically increased by 50%.
        @axis_fontsize (int): Font size for the axis labels. Automatically increased by 50%.
        @legend_fontsize (int): Font size of the legend. Automatically increased by 50%.
        @xlabel (str): Label for the x-axis (epochs). Default is "Epochs".
        @ylabel (str): Label for the y-axis (loss). Default is "Loss".
        @title (str): Title of the plot. Default is "Loss Graph".
        @grid (bool): If True, enables grid lines on the plot. Default is True.
        @line_style_training (str): Line style for the training loss curve. Default is "-".
        @line_style_validation (str): Line style for the validation loss curve. Default is "--".
        @line_width (int): Width of the lines for both loss curves. Default is 2.

    Example:
    -------
        >>> # python
        ...     history_dict_list = [
        ...     {"Name": "Model1", "History": {"loss": [0.5, 0.4, 0.3], "val_loss": [0.55, 0.45, 0.35]}},
        ...     {"Name": "Model2", "History": {"loss": [0.6, 0.5, 0.4], "val_loss": [0.65, 0.55, 0.45]}}
        ...     ]
        ...     path_output = "/path/to/save/plots/"
        ...
        ...     # Create the LossPlotter instance with custom options
        ...     plotter = PlotLossCurve(
        ...     loss_curve_figure_size=(12, 8),
        ...     loss_curve_training_loss_color="blue",
        ...     loss_curve_validation_loss_color="orange",
        ...     loss_curve_title_font_size=18,
        ...     loss_curve_line_style_training="-.",
        ...     loss_curve_line_style_validation=":",
        ...     loss_curve_line_width=3
        ...     )
        ...
        ...     # Generate and save the plots (also saves JSON)
        ...     plotter.plot_loss(history_dict_list, path_output)
        ...
        ...     # Later, reload and plot from JSON
        ...     plotter.plot_from_json("/path/to/plot_data.json", path_output)
        >>>
    """

    def __init__(self, loss_curve_figure_size: tuple, loss_curve_training_loss_color: str,
                 loss_curve_validation_loss_color: str, loss_curve_title_font_size: int,
                 loss_curve_axis_font_size: int, loss_curve_legend_font_size: int,
                 loss_curve_x_label: str, loss_curve_y_label: str, loss_curve_title: str,
                 loss_curve_grid: bool, loss_curve_line_style_training: str,
                 loss_curve_line_style_validation: str, loss_curve_line_width: int):
        """
        Initializes the LossPlotter instance with the list of model histories and the output path
        for saving the loss plots. Additionally, it allows for customization of the plot styles
        and parameters.

        NOTE: Font sizes are automatically increased by 50% for better readability.

        Args:
            @loss_curve_figure_size (tuple): The size of the plot figure (width, height).
            @loss_curve_training_loss_color (str): The color of the training loss curve.
            @loss_curve_validation_loss_color (str): The color of the validation loss curve.
            @loss_curve_title_font_size (int): Font size of the plot title (will be increased by 50%).
            @loss_curve_axis_font_size (int): Font size for the axis labels (will be increased by 50%).
            @loss_curve_legend_font_size (int): Font size of the legend (will be increased by 50%).
            @loss_curve_x_label (str): Label for the x-axis (epochs).
            @loss_curve_y_label (str): Label for the y-axis (loss).
            @loss_curve_title (str): Title of the plot.
            @loss_curve_grid (bool): If True, enables grid lines on the plot.
            @loss_curve_line_style_training (str): Line style for the training loss curve.
            @loss_curve_line_style_validation (str): Line style for the validation loss curve.
            @loss_curve_line_width (int): Width of the lines for both loss curves.
        """

        # Validate figure size
        if not isinstance(loss_curve_figure_size, tuple) or len(loss_curve_figure_size) != 2:
            raise ValueError("loss_curve_figure_size must be a tuple of two values (width, height).")

        if not all(isinstance(i, (int, float)) and i > 0 for i in loss_curve_figure_size):
            raise ValueError("Both width and height in loss_curve_figure_size must be positive numbers.")

        # Validate color values (training and validation)
        valid_colors = ['blue', 'green', 'red', 'orange', 'purple', 'black', 'yellow', 'cyan', 'magenta']

        for color in [loss_curve_training_loss_color, loss_curve_validation_loss_color]:
            if not isinstance(color, str) or color not in valid_colors:
                raise ValueError(
                    f"{color} is not a valid color. Please use one of the following: {', '.join(valid_colors)}.")

        # Validate font sizes
        if not isinstance(loss_curve_title_font_size, int) or loss_curve_title_font_size <= 0:
            raise ValueError("loss_curve_title_font_size must be a positive integer.")

        if not isinstance(loss_curve_axis_font_size, int) or loss_curve_axis_font_size <= 0:
            raise ValueError("loss_curve_axis_font_size must be a positive integer.")

        if not isinstance(loss_curve_legend_font_size, int) or loss_curve_legend_font_size <= 0:
            raise ValueError("loss_curve_legend_font_size must be a positive integer.")

        # Validate x and y labels
        if not isinstance(loss_curve_x_label, str) or not loss_curve_x_label:
            raise ValueError("loss_curve_x_label must be a non-empty string.")

        if not isinstance(loss_curve_y_label, str) or not loss_curve_y_label:
            raise ValueError("loss_curve_y_label must be a non-empty string.")

        # Validate title
        if not isinstance(loss_curve_title, str) or not loss_curve_title:
            raise ValueError("loss_curve_title must be a non-empty string.")

        # Validate grid option
        if not isinstance(loss_curve_grid, bool):
            raise ValueError("loss_curve_grid must be a boolean value.")

        # Validate line styles
        valid_line_styles = ['-', '--', '-.', ':']

        for line_style in [loss_curve_line_style_training, loss_curve_line_style_validation]:
            if not isinstance(line_style, str) or line_style not in valid_line_styles:
                raise ValueError(
                    f"{line_style} is not a valid line style. Please use one of the following: {', '.join(valid_line_styles)}.")

        # Validate line width
        if not isinstance(loss_curve_line_width, int) or loss_curve_line_width <= 0:
            raise ValueError("loss_curve_line_width must be a positive integer.")

        # Store original font sizes for JSON serialization
        self._original_title_font_size = loss_curve_title_font_size
        self._original_axis_font_size = loss_curve_axis_font_size
        self._original_legend_font_size = loss_curve_legend_font_size

        # Apply 50% increase to all font sizes
        self._loss_curve_history_figure_size = loss_curve_figure_size
        self._loss_curve_history_training_loss_color = loss_curve_training_loss_color
        self._loss_curve_history_validation_loss_color = loss_curve_validation_loss_color
        self._loss_curve_history_title_font_size = int(loss_curve_title_font_size * 1.65)
        self._loss_curve_history_axis_font_size = int(loss_curve_axis_font_size * 1.65)
        self._loss_curve_history_legend_font_size = int(loss_curve_legend_font_size * 1.65)
        self._loss_curve_history_x_label = loss_curve_x_label
        self._loss_curve_history_y_label = loss_curve_y_label
        self._loss_curve_history_title = loss_curve_title
        self._loss_curve_history_grid = loss_curve_grid
        self._loss_curve_history_line_style_training = loss_curve_line_style_training
        self._loss_curve_history_line_style_validation = loss_curve_line_style_validation
        self._loss_curve_history_line_width = loss_curve_line_width

    def plot_loss(self, loss_curve_history_dict_list, loss_curve_path_output, save_json=True):
        """
        Plots the training and validation loss curves for each model in the history list and saves
        the plots to the specified output directory.

        This method generates and saves the loss plots for each model in the history_dict_list.
        The plots are saved as PDF images in the directory specified by path_output.
        Additionally, all plot data is saved to JSON for reproducibility.

        Args:
            loss_curve_history_dict_list (list): List of dictionaries with model histories
            loss_curve_path_output (str): Output directory path
            save_json (bool): If True, saves plot data to JSON files. Default is True.
        """
        logging.info("Starting the process of plotting and saving loss graphs for the models.")

        # Ensure output path exists
        Path(loss_curve_path_output).mkdir(parents=True, exist_ok=True)

        # Iterate over each model's history dictionary
        for history_dict in loss_curve_history_dict_list:
            try:
                model_name = history_dict['Name']
                history = history_dict['History']

                logging.info(f"Processing model '{model_name}'.")

                # Check if the history contains 'loss' data
                if 'loss' not in history:
                    logging.warning(f"No 'loss' data found for model '{model_name}', skipping plot.")
                    continue

                # Save plot data to JSON if enabled
                if save_json:
                    self._save_plot_data_to_json(model_name, history, loss_curve_path_output)

                # Create the figure and apply styles
                self._loss_curve_create_figure()

                # Plot the loss curves (training and validation)
                self._loss_curve_plot_loss(history, model_name)

                # Save the plot
                self._loss_curve_save_plot(model_name, loss_curve_path_output)

            except KeyError as e:
                logging.error(f"KeyError in model '{model_name}': {e}")

            except Exception as e:
                logging.error(f"An error occurred while processing model '{model_name}': {e}")
                raise

        logging.info("Plotting and saving of loss graphs for all models completed.")

    def _save_plot_data_to_json(self, model_name, history, output_path):
        """
        Saves all plot data and configuration to a JSON file for reproducibility.

        Args:
            model_name (str): The name of the model
            history (dict): The history dictionary containing loss data
            output_path (str): The output directory path
        """
        plot_data = {
            "model_name": model_name,
            "history": {
                "loss": history.get('loss', []),
                "val_loss": history.get('val_loss', [])
            },
            "plot_configuration": {
                "figure_size": list(self._loss_curve_history_figure_size),
                "training_loss_color": self._loss_curve_history_training_loss_color,
                "validation_loss_color": self._loss_curve_history_validation_loss_color,
                "title_font_size_original": self._original_title_font_size,
                "axis_font_size_original": self._original_axis_font_size,
                "legend_font_size_original": self._original_legend_font_size,
                "title_font_size_actual": self._loss_curve_history_title_font_size,
                "axis_font_size_actual": self._loss_curve_history_axis_font_size,
                "legend_font_size_actual": self._loss_curve_history_legend_font_size,
                "x_label": self._loss_curve_history_x_label,
                "y_label": self._loss_curve_history_y_label,
                "title": self._loss_curve_history_title,
                "grid": self._loss_curve_history_grid,
                "line_style_training": self._loss_curve_history_line_style_training,
                "line_style_validation": self._loss_curve_history_line_style_validation,
                "line_width": self._loss_curve_history_line_width
            },
            "metadata": {
                "epochs": len(history.get('loss', [])),
                "has_validation_data": 'val_loss' in history,
                "font_increase_applied": "50%"
            }
        }

        json_file_path = f'{output_path}{model_name}_plot_data.json'

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(plot_data, json_file, indent=4, ensure_ascii=False)

        logging.info(f"Plot data saved to JSON: {json_file_path}")

    @classmethod
    def load_from_json(cls, json_file_path):
        """
        Creates a PlotLossCurve instance from a JSON file containing plot configuration.

        Args:
            json_file_path (str): Path to the JSON file with plot configuration

        Returns:
            PlotLossCurve: A new instance configured from the JSON file

        Example:
            >>> plotter = PlotLossCurve.load_from_json("model1_plot_data.json")
            >>> plotter.plot_loss([{"Name": "Model1", "History": {...}}], "/output/")
        """
        logging.info(f"Loading plot configuration from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            plot_data = json.load(json_file)

        config = plot_data['plot_configuration']

        # Create instance using original font sizes (will be increased by 50% automatically)
        instance = cls(
            loss_curve_figure_size=tuple(config['figure_size']),
            loss_curve_training_loss_color=config['training_loss_color'],
            loss_curve_validation_loss_color=config['validation_loss_color'],
            loss_curve_title_font_size=config['title_font_size_original'],
            loss_curve_axis_font_size=config['axis_font_size_original'],
            loss_curve_legend_font_size=config['legend_font_size_original'],
            loss_curve_x_label=config['x_label'],
            loss_curve_y_label=config['y_label'],
            loss_curve_title=config['title'],
            loss_curve_grid=config['grid'],
            loss_curve_line_style_training=config['line_style_training'],
            loss_curve_line_style_validation=config['line_style_validation'],
            loss_curve_line_width=config['line_width']
        )

        logging.info("PlotLossCurve instance created from JSON configuration")
        return instance

    def plot_from_json(self, json_file_path, output_path):
        """
        Loads plot data from a JSON file and generates the plot.

        This is a convenience method that loads both the configuration and data from JSON,
        then generates the plot. Useful for reproducing plots later.

        Args:
            json_file_path (str): Path to the JSON file containing plot data
            output_path (str): Directory where the plot should be saved

        Example:
            >>> plotter = PlotLossCurve(...configuration...)
            >>> plotter.plot_from_json("model1_plot_data.json", "/output/")
        """
        logging.info(f"Loading and plotting from JSON: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            plot_data = json.load(json_file)

        model_name = plot_data['model_name']
        history = plot_data['history']

        # Ensure output path exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Create the history dict in the expected format
        history_dict = {
            'Name': model_name,
            'History': history
        }

        # Plot using the loaded data
        self.plot_loss([history_dict], output_path, save_json=False)

        logging.info(f"Plot recreated from JSON for model '{model_name}'")

    @staticmethod
    def plot_from_json_static(json_file_path, output_path):
        """
        Static method to load configuration and data from JSON and plot in one step.

        This method creates a PlotLossCurve instance from the JSON configuration,
        then immediately plots the data. Most convenient for one-off plotting from saved data.

        Args:
            json_file_path (str): Path to the JSON file containing plot data and configuration
            output_path (str): Directory where the plot should be saved

        Example:
            >>> PlotLossCurve.plot_from_json_static("model1_plot_data.json", "/output/")
        """
        logging.info(f"Static plotting from JSON: {json_file_path}")

        # Load the instance from JSON
        plotter = PlotLossCurve.load_from_json(json_file_path)

        # Plot using the loaded instance
        plotter.plot_from_json(json_file_path, output_path)

        logging.info("Static plot from JSON completed")

    def _loss_curve_create_figure(self):
        """
        Creates the plot figure with the specified customization options (e.g., figure size and grid).
        """
        plt.figure(figsize=self._loss_curve_history_figure_size)

        # Add grid if enabled
        if self._loss_curve_history_grid:
            plt.grid(True)

    def _loss_curve_plot_loss(self, history, model_name):
        """
        Plots the training and validation loss curves for a single model.

        Args:
            history (dict): The history dictionary for a model containing 'loss' and optionally 'val_loss'.
            model_name (str): The name of the model.
        """
        # Plot the training loss
        plt.plot(history['loss'], label='Training Loss', color=self._loss_curve_history_training_loss_color,
                 linestyle=self._loss_curve_history_line_style_training, linewidth=self._loss_curve_history_line_width)

        # Plot the validation loss, if available
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', color=self._loss_curve_history_validation_loss_color,
                     linestyle=self._loss_curve_history_line_style_validation,
                     linewidth=self._loss_curve_history_line_width)

        # Add title, labels, and legend with increased font sizes
        plt.title(f'{self._loss_curve_history_title} - {model_name}', fontsize=self._loss_curve_history_title_font_size)
        plt.xlabel(self._loss_curve_history_x_label, fontsize=self._loss_curve_history_axis_font_size)
        plt.ylabel(self._loss_curve_history_y_label, fontsize=self._loss_curve_history_axis_font_size)
        plt.legend(fontsize=self._loss_curve_history_legend_font_size)

    @staticmethod
    def _loss_curve_save_plot(model_name, loss_curve_path_output):
        """
        Saves the generated plot to the specified output directory.

        Args:
            model_name (str): The name of the model to generate the file name for the plot.
            loss_curve_path_output (str): Output directory path
        """
        file_path = f'{loss_curve_path_output}{model_name}_loss.pdf'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        logging.info(f"Loss plot saved for model '{model_name}' at {file_path}")
