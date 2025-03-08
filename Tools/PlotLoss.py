import logging
import matplotlib.pyplot as plt


class LossPlotter:
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
        ...     plotter = LossPlotter(
        ...     history_dict_list,
        ...     path_output,
        ...     figure_size=(12, 8),
        ...     training_loss_color="blue",
        ...     validation_loss_color="orange",
        ...     title_font_size=18,
        ...     line_style_training="-.",
        ...     line_style_validation=":",
        ...     line_width=3
        ...     )
        ...
        ...     # Generate and save the plots
        ...     plotter.plot_and_save_loss()
        >>>
    """

    def __init__(self, history_dict_list, path_output, figure_size=(10, 6), training_loss_color="blue",
                 validation_loss_color="orange", title_font_size=16, axis_font_size=12,
                 legend_font_size=10, x_label="Epochs", y_label="Loss", title="Loss Graph",
                 grid=True, line_style_training='-', line_style_validation='--', line_width=2):
        """
        Initializes the LossPlotter instance with the list of model histories and the output path
        for saving the loss plots. Additionally, it allows for customization of the plot styles
        and parameters.

        Args:
            @history_dict_list (list): A list of dictionaries where each dictionary contains the 'Name'
                                      of a model and the 'History' dictionary with loss values.
            @path_output (str): The directory path where the loss plot images should be saved.
            @figure_size (tuple): The size of the plot figure (width, height). Default is (10, 6).
            @training_loss_color (str): The color of the training loss curve. Default is "blue".
            @validation_loss_color (str): The color of the validation loss curve. Default is "orange".
            @title_font_size (int): Font size of the plot title. Default is 16.
            @axis_font_size (int): Font size for the axis labels (x and y). Default is 12.
            @legend_font_size (int): Font size of the legend. Default is 10.
            @x_label (str): Label for the x-axis (epochs). Default is "Epochs".
            @y_label (str): Label for the y-axis (loss). Default is "Loss".
            @title (str): Title of the plot. Default is "Loss Graph".
            @grid (bool): If True, enables grid lines on the plot. Default is True.
            @line_style_training (str): Line style for the training loss curve. Default is "-".
            @line_style_validation (str): Line style for the validation loss curve. Default is "--".
            @line_width (int): Width of the lines for both loss curves. Default is 2.
        """
        self.history_dict_list = history_dict_list
        self.path_output = path_output
        self.figure_size = figure_size
        self.training_loss_color = training_loss_color
        self.validation_loss_color = validation_loss_color
        self.title_font_size = title_font_size
        self.axis_font_size = axis_font_size
        self.legend_font_size = legend_font_size
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.grid = grid
        self.line_style_training = line_style_training
        self.line_style_validation = line_style_validation
        self.line_width = line_width

    def plot_and_save_loss(self):
        """
        Plots the training and validation loss curves for each model in the history list and saves
        the plots to the specified output directory.

        This method generates and saves the loss plots for each model in the history_dict_list.
        The plots are saved as PNG images in the directory specified by path_output.
        """
        logging.info("Starting the process of plotting and saving loss graphs for the models.")

        # Iterate over each model's history dictionary
        for history_dict in self.history_dict_list:

            try:
                model_name = history_dict['Name']
                history = history_dict['History']

                logging.info(f"Processing model '{model_name}'.")

                # Check if the history contains 'loss' data
                if 'loss' not in history:
                    logging.warning(f"No 'loss' data found for model '{model_name}', skipping plot.")
                    continue

                # Create the figure and apply styles
                self._create_figure()

                # Plot the loss curves (training and validation)
                self._plot_loss(history, model_name)

                # Save the plot
                self._save_plot(model_name)

            except KeyError as e:
                logging.error(f"KeyError in model '{model_name}': {e}")

            except Exception as e:
                logging.error(f"An error occurred while processing model '{model_name}': {e}")
                raise

        logging.info("Plotting and saving of loss graphs for all models completed.")

    def _create_figure(self):
        """
        Creates the plot figure with the specified customization options (e.g., figure size and grid).
        """
        plt.figure(figsize=self.figure_size)

        # Add grid if enabled
        if self.grid:
            plt.grid(True)

    def _plot_loss(self, history, model_name):
        """
        Plots the training and validation loss curves for a single model.

        Args:
            history (dict): The history dictionary for a model containing 'loss' and optionally 'val_loss'.
            model_name (str): The name of the model.
        """
        # Plot the training loss
        plt.plot(history['loss'], label='Training Loss', color=self.training_loss_color,
                 linestyle=self.line_style_training, linewidth=self.line_width)

        # Plot the validation loss, if available
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', color=self.validation_loss_color,
                     linestyle=self.line_style_validation, linewidth=self.line_width)

        # Add title, labels, and legend
        plt.title(f'{self.title} - {model_name}', fontsize=self.title_font_size)
        plt.xlabel(self.x_label, fontsize=self.axis_font_size)
        plt.ylabel(self.y_label, fontsize=self.axis_font_size)
        plt.legend(fontsize=self.legend_font_size)

    def _save_plot(self, model_name):
        """
        Saves the generated plot to the specified output directory.

        Args:
            model_name (str): The name of the model to generate the file name for the plot.
        """
        file_path = f'{self.path_output}{model_name}_loss.png'
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory
        logging.info(f"Loss plot saved for model '{model_name}' at {file_path}")