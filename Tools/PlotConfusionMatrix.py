import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrixPlotter:
    """
    A class to plot and save confusion matrices with highly customizable visualizations.

    This class provides a method to plot confusion matrices from a list of dictionaries,
    each containing a confusion matrix and its associated class names. The class supports
    multiple customization options like figure size, colormap, font sizes, axis tick label rotation,
    color bar options, annotation format, and the option to display or save the plot.


    Example:
    -------
        >>> # python
        ...     cm_plotter = ConfusionMatrixPlotter()
        ...     cm_list = [
        ...     {
        ...     'confusion_matrix': [[50, 10], [5, 35]],
        ...     'class_names': ['Class A', 'Class B'],
        ...     'title': 'Model 1'
        ...     },
        ...     {
        ...     'confusion_matrix': [[30, 20], [10, 40]],
        ...     'class_names': ['Class A', 'Class B'],
        ...     'title': 'Model 2'
        ...     }
        ...     ]
        >>>     cm_plotter.plot_confusion_matrices(cm_list, file_name_path='confusion_matrices/', show_plot=True)
    """

    def __init__(self, figure_size=(5, 5), cmap='Blues', annot_font_size=10,
                 label_font_size=12, title_font_size=14, show_plot=False,
                 colorbar=True, annot_kws=None, fmt='d', rotation=45):
        """
        Initialize the ConfusionMatrixPlotter with customizable plotting options.

        Parameters
        ----------
        @fig_size : tuple, optional
            Size of the figure (default is (5, 5)).
        @cmap : str, optional
            Colormap for the confusion matrix (default is 'Blues').
        @annot_font_size : int, optional
            Font size for the annotations (default is 10).
        @label_font_size : int, optional
            Font size for the axis labels (default is 12).
        @title_font_size : int, optional
            Font size for the plot title (default is 14).
        @show_plot : bool, optional
            Whether to display the plot interactively (default is False).
        @colorbar : bool, optional
            Whether to display the color bar (default is True).
        @annot_kws : dict, optional
            Additional keyword arguments for annotations (default is None).
        @fmt : str, optional
            Format for annotations (default is 'd', for integers).
        @rotation : int, optional
            Rotation angle for x and y axis labels (default is 45).
        """
        # Initialize instance variables with provided or default values
        self.figure_size = figure_size  # Size of the figure to be plotted
        self.cmap = cmap  # Color map for the heatmap
        self.annot_font_size = annot_font_size  # Font size for annotations
        self.label_font_size = label_font_size  # Font size for axis labels
        self.title_font_size = title_font_size  # Font size for plot title
        self.show_plot = show_plot  # Whether to show the plot interactively
        self.colorbar = colorbar  # Whether to display the color bar
        self.annot_kws = annot_kws if annot_kws is not None else {
            "size": self.annot_font_size}  # Annotation keyword arguments
        self.fmt = fmt  # Format string for annotations (e.g., integers or floating point)
        self.rotation = rotation  # Rotation for axis labels (default is 45 degrees)

    @staticmethod
    def _process_confusion_matrix(confusion_matrix_dict):
        """
        Process the confusion matrix dictionary and return necessary data.

        Parameters
        ----------
        confusion_matrix_dict : dict
            A dictionary containing the confusion matrix, class names, and optional title.

        Returns
        -------
        numpy.ndarray
            The confusion matrix.
        list
            The class names.
        str
            The title for the confusion matrix plot.
        """
        # Convert confusion matrix to a NumPy array for easy manipulation
        confusion_matrix = np.array(confusion_matrix_dict["confusion_matrix"])
        # Get class names
        class_names = confusion_matrix_dict["class_names"]
        # Get plot title (if not provided, default to "Confusion Matrix")
        title = confusion_matrix_dict.get("title", "Confusion Matrix")
        return confusion_matrix, class_names, title

    def _plot_single_confusion_matrix(self, confusion_matrix, class_names, title, index, file_name_path):
        """
        Plot and save or display a single confusion matrix.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The confusion matrix to be plotted.
        class_names : list
            The class labels for the confusion matrix.
        title : str
            The title of the confusion matrix plot.
        index : int
            The index of the current confusion matrix in the list (for naming purposes).
        file_name_path : str
            The path where the confusion matrix image will be saved.
        """
        # Create a new figure with the specified size
        plt.figure(figsize=self.figure_size)

        # Create the heatmap using seaborn's heatmap function
        ax = sns.heatmap(
            confusion_matrix,
            annot=True,  # Annotate each cell with its value
            fmt=self.fmt,  # Format of the annotation (integer or float)
            cmap=self.cmap,  # Color map for the heatmap
            xticklabels=class_names,  # Class names for x-axis
            yticklabels=class_names,  # Class names for y-axis
            annot_kws=self.annot_kws,  # Additional annotation options (like font size)
            cbar=self.colorbar  # Show or hide the color bar
        )

        # Customize the axis labels and title
        ax.set_xlabel('Predicted Labels', fontsize=self.label_font_size)
        ax.set_ylabel('True Labels', fontsize=self.label_font_size)
        ax.set_title(title, fontsize=self.title_font_size)

        # Rotate the axis labels to avoid overlap
        plt.xticks(rotation=self.rotation)
        plt.yticks(rotation=self.rotation)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Define the file path to save the plot
        file_path = f"{file_name_path}matrix_{index + 1}.png"

        # If show_plot is True, display the plot; otherwise, save the plot as a file
        if self.show_plot:
            plt.show()  # Display the plot interactively
            logging.debug(f"Confusion matrix {index + 1} displayed on screen.")
        else:
            plt.savefig(file_path)  # Save the plot to a file
            logging.debug(f"Confusion matrix {index + 1} saved to {file_path}.")

        # Close the plot to free up memory
        plt.close()

    def plot_confusion_matrices(self, confusion_matrix_list, file_name_path):
        """
        Plot confusion matrices for a list of models and save them as images.

        Parameters
        ----------
        confusion_matrix_list : list of dict
            Each dictionary contains:
            - 'confusion_matrix': The confusion matrix as a 2D array.
            - 'class_names': The class labels for the matrix.
            - 'title' (optional): Title for the matrix plot.
        file_name_path : str
            Path to save the confusion matrix images.
        """
        logging.info("Starting confusion matrix plotting process.")

        # Iterate over each confusion matrix in the list
        for index, confusion_matrix_dict in enumerate(confusion_matrix_list):
            try:
                # Process the confusion matrix and extract data (matrix, class names, title)
                confusion_matrix, class_names, title = self._process_confusion_matrix(confusion_matrix_dict)

                logging.info(f"Plotting confusion matrix {index + 1}: {title}")

                # Call the function to plot and save/display the confusion matrix
                self._plot_single_confusion_matrix(confusion_matrix, class_names, title, index, file_name_path)

            except KeyError as e:
                # Handle missing keys in the input dictionary
                logging.error(f"Missing key in confusion matrix dictionary: {e}")
            except Exception as e:
                # Handle any other errors that occur during the plotting process
                logging.error(f"Error occurred while plotting confusion matrix {index + 1}: {e}")
                raise

        logging.info("Confusion matrix plotting process completed.")