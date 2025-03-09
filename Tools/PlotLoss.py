#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2024/07/17'
__last_update__ = '2024/07/17'
__credits__ = ['unknown']


try:
    import sys
    import logging

    import matplotlib.pyplot as plt

except ImportError as error:
    print(error)
    sys.exit(-1)



class PlotLossCurve:
    """
    This class is responsible for plotting and saving training and validation loss curves
    for different machine learning models. It allows extensive customization of the plot
    through various parameters passed during initialization.


    Args:

        @history_dict_list (list): A list of dictionaries where each dictionary contains the 'Name'
         of a model and the 'History' dictionary with loss values.
        @path_output (str): The directory path where the loss plot images should be saved.
        @figsize (tuple): The size of the plot figure (width, height). Default is (10, 6).
        @training_loss_color (str): The color of the training loss curve. Default is "blue".
        @validation_loss_color (str): The color of the validation loss curve. Default is "orange".
        @title_fontsize (int): Font size of the plot title. Default is 16.
        @axis_fontsize (int): Font size for the axis labels (x and y). Default is 12.
        @legend_fontsize (int): Font size of the legend. Default is 10.
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
        ...     history_dict_list,
        ...     path_output,
        ...     loss_curve_figure_size=(12, 8),
        ...     loss_curve_training_loss_color="blue",
        ...     loss_curve_validation_loss_color="orange",
        ...     loss_curve_title_font_size=18,
        ...     loss_curve_line_style_training="-.",
        ...     loss_curve_line_style_validation=":",
        ...     loss_curve_line_width=3
        ...     )
        ...
        ...     # Generate and save the plots
        ...     plotter.plot_loss()
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

        Args:

            @loss_curve_history_path_output (str): The directory path where the loss plot images should be saved.
            @loss_curve_history_figure_size (tuple): The size of the plot figure (width, height). Default is (10, 6).
            @loss_curve_history_training_loss_color (str): The color of the training loss curve. Default is "blue".
            @loss_curve_history_validation_loss_color (str): The color of the validation loss curve. Default is "orange".
            @loss_curve_history_title_font_size (int): Font size of the plot title. Default is 16.
            @loss_curve_history_axis_font_size (int): Font size for the axis labels (x and y). Default is 12.
            @loss_curve_history_legend_font_size (int): Font size of the legend. Default is 10.
            @loss_curve_history_x_label (str): Label for the x-axis (epochs). Default is "Epochs".
            @loss_curve_history_y_label (str): Label for the y-axis (loss). Default is "Loss".
            @loss_curve_history_title (str): Title of the plot. Default is "Loss Graph".
            @loss_curve_history_grid (bool): If True, enables grid lines on the plot. Default is True.
            @loss_curve_history_line_style_training (str): Line style for the training loss curve. Default is "-".
            @loss_curve_history_line_style_validation (str): Line style for the validation loss curve. Default is "--".
            @loss_curve_history_line_width (int): Width of the lines for both loss curves. Default is 2.
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

        self._loss_curve_history_figure_size = loss_curve_figure_size
        self._loss_curve_history_training_loss_color = loss_curve_training_loss_color
        self._loss_curve_history_validation_loss_color = loss_curve_validation_loss_color
        self._loss_curve_history_title_font_size = loss_curve_title_font_size
        self._loss_curve_history_axis_font_size = loss_curve_axis_font_size
        self._loss_curve_history_legend_font_size = loss_curve_legend_font_size
        self._loss_curve_history_x_label = loss_curve_x_label
        self._loss_curve_history_y_label = loss_curve_y_label
        self._loss_curve_history_title = loss_curve_title
        self._loss_curve_history_grid = loss_curve_grid
        self._loss_curve_history_line_style_training = loss_curve_line_style_training
        self._loss_curve_history_line_style_validation = loss_curve_line_style_validation
        self._loss_curve_history_line_width = loss_curve_line_width

    def plot_loss(self, loss_curve_history_dict_list, loss_curve_path_output):
        """
        Plots the training and validation loss curves for each model in the history list and saves
        the plots to the specified output directory.

        This method generates and saves the loss plots for each model in the history_dict_list.
        The plots are saved as PNG images in the directory specified by path_output.
        """
        logging.info("Starting the process of plotting and saving loss graphs for the models.")

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
                     linestyle=self._loss_curve_history_line_style_validation, linewidth=self._loss_curve_history_line_width)

        # Add title, labels, and legend
        plt.title(f'{self._loss_curve_history_title} - {model_name}', fontsize=self._loss_curve_history_title_font_size)
        plt.xlabel(self._loss_curve_history_x_label, fontsize=self._loss_curve_history_axis_font_size)
        plt.ylabel(self._loss_curve_history_y_label, fontsize=self._loss_curve_history_axis_font_size)
        plt.legend(fontsize=self._loss_curve_history_legend_font_size)

    def _loss_curve_save_plot(self, model_name, loss_curve_path_output):
        """
        Saves the generated plot to the specified output directory.

        Args:
            model_name (str): The name of the model to generate the file name for the plot.
        """
        file_path = f'{loss_curve_path_output}{model_name}_loss.png'
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory
        logging.info(f"Loss plot saved for model '{model_name}' at {file_path}")