import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

DEFAULT_CONF_MATRIX_CMAP = 'Blues'
DEFAULT_ROC_CURVE_COLOR = 'darkorange'
DEFAULT_ROC_CURVE_LABEL = 'ROC curve (area = %0.2f)'
DEFAULT_ROC_CURVE_LS = '--'
DEFAULT_ROC_CURVE_LW = 2
DEFAULT_ROC_CURVE_FONT_SIZE = 12
DEFAULT_CONF_MATRIX_FONT_SIZE = 10


class ClassificationResults:
    """
    A class to calculate and visualize classification metrics including confusion matrix and ROC curve.

    Attributes:
        matriz_cmap (str): Colormap for confusion matrix heatmap.
        roc_curve_color (str): Color for ROC curve plot.
        true_predictions (array): True positive rates from ROC curve.
        false_predictions (array): False positive rates from ROC curve.
        area_under_curve (float): Area under ROC curve (AUC).
        class_report (dict): Classification report with precision, recall, F1-score, etc.
        confusion_matrix (array): Confusion matrix.
        y_score (array): Predicted probabilities or scores for positive class.
        roc_curve_label (str): Label for ROC curve plot.
        roc_curve_ls (str): Line style for ROC curve plot.
        roc_curve_lw (int): Line width for ROC curve plot.
        roc_curve_font_size (int): Font size for ROC curve plot.
        confusion_matrix_font_size (int): Font size for confusion matrix heatmap.

    Methods:
        calculate_metrics(y_real, y_predicted):
            Calculates confusion matrix and classification report.
            Optionally computes ROC curve metrics if y_score is provided.

        plot_metrics(conf_matrix_cmap=None, roc_curve_color=None, roc_curve_label=None,
                     roc_curve_ls=None, roc_curve_lw=None, roc_curve_font_size=None,
                     conf_matrix_font_size=None):
            Plots confusion matrix and ROC curve.

        save_metrics_plot(file_name, conf_matrix_cmap=None, roc_curve_color=None,
                          roc_curve_label=None, roc_curve_ls=None, roc_curve_lw=None,
                          roc_curve_font_size=None, conf_matrix_font_size=None):
            Saves plotted metrics to a file.

        save_metrics_text(y_real, y_predicted, file_name):
            Saves classification report, confusion matrix, and ROC metrics to a text file.
    """

    def __init__(self, matriz_cmap=DEFAULT_CONF_MATRIX_CMAP, roc_curve_color=DEFAULT_ROC_CURVE_COLOR,
                 roc_curve_label=DEFAULT_ROC_CURVE_LABEL, roc_curve_ls=DEFAULT_ROC_CURVE_LS,
                 roc_curve_lw=DEFAULT_ROC_CURVE_LW, roc_curve_font_size=DEFAULT_ROC_CURVE_FONT_SIZE,
                 confusion_matrix_font_size=DEFAULT_CONF_MATRIX_FONT_SIZE, y_score=None):
        """
        Initializes ClassificationResults with default or provided parameters.

        Args:
            matriz_cmap (str): Colormap for confusion matrix heatmap.
            roc_curve_color (str): Color for ROC curve plot.
            roc_curve_label (str): Label for ROC curve plot.
            roc_curve_ls (str): Line style for ROC curve plot.
            roc_curve_lw (int): Line width for ROC curve plot.
            roc_curve_font_size (int): Font size for ROC curve plot.
            confusion_matrix_font_size (int): Font size for confusion matrix heatmap.
            y_score (array, optional): Predicted probabilities or scores for positive class.
        """
        self.matriz_cmap = matriz_cmap
        self.roc_curve_color = roc_curve_color
        self.true_predictions = None
        self.false_predictions = None
        self.area_under_curve = None
        self.class_report = None
        self.confusion_matrix = None
        self.y_score = y_score
        self.roc_curve_label = roc_curve_label
        self.roc_curve_ls = roc_curve_ls
        self.roc_curve_lw = roc_curve_lw
        self.roc_curve_font_size = roc_curve_font_size
        self.confusion_matrix_font_size = confusion_matrix_font_size

    def calculate_metrics(self, y_real, y_predicted):
        """
        Calculates confusion matrix and classification report.

        Optionally computes ROC curve metrics if y_score is provided.

        Args:
            y_real (array): True labels.
            y_predicted (array): Predicted labels.
        """
        self.confusion_matrix = confusion_matrix(y_real, y_predicted)
        self.class_report = classification_report(y_real, y_predicted, output_dict=True)

        if self.y_score is not None:
            false_predictions, true_predictions, _ = roc_curve(y_real, self.y_score)
            self.area_under_curve = auc(false_predictions, true_predictions)
            self.false_predictions = false_predictions
            self.true_predictions = true_predictions

    def plot_metrics(self, conf_matrix_cmap=None, roc_curve_color=None,
                     roc_curve_label=None, roc_curve_ls=None, roc_curve_lw=None,
                     roc_curve_font_size=None, conf_matrix_font_size=None):
        """
        Plots confusion matrix and ROC curve.

        Args:
            conf_matrix_cmap (str, optional): Colormap for confusion matrix heatmap.
            roc_curve_color (str, optional): Color for ROC curve plot.
            roc_curve_label (str, optional): Label for ROC curve plot.
            roc_curve_ls (str, optional): Line style for ROC curve plot.
            roc_curve_lw (int, optional): Line width for ROC curve plot.
            roc_curve_font_size (int, optional): Font size for ROC curve plot.
            conf_matrix_font_size (int, optional): Font size for confusion matrix heatmap.
        """
        conf_matrix_cmap = conf_matrix_cmap or self.matriz_cmap
        roc_curve_color = roc_curve_color or self.roc_curve_color
        roc_curve_label = roc_curve_label or (self.roc_curve_label % self.area_under_curve
                                              if hasattr(self, 'roc_auc') else 'ROC curve')
        roc_curve_ls = roc_curve_ls or self.roc_curve_ls
        roc_curve_lw = roc_curve_lw or self.roc_curve_lw
        roc_curve_font_size = roc_curve_font_size or self.roc_curve_font_size
        conf_matrix_font_size = conf_matrix_font_size or self.confusion_matrix_font_size

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.heatmap(self.confusion_matrix, annot=True, cmap=conf_matrix_cmap,
                    fmt='g', annot_kws={"size": conf_matrix_font_size})
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        # Plot ROC curve
        if hasattr(self, 'fpr') and hasattr(self, 'tpr'):
            plt.subplot(1, 2, 2)
            plt.plot(self.false_predictions, self.true_predictions, color=roc_curve_color,
                     lw=roc_curve_lw,
                     label=roc_curve_label)
            plt.plot([0, 1], [0, 1], color='navy', lw=roc_curve_lw,
                     linestyle=roc_curve_ls)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right', fontsize=roc_curve_font_size)

        plt.tight_layout()

    def save_metrics_plot(self, file_name, conf_matrix_cmap=None, roc_curve_color=None,
                          roc_curve_label=None, roc_curve_ls=None, roc_curve_lw=None,
                          roc_curve_font_size=None, conf_matrix_font_size=None):
        """
        Saves plotted metrics to a file.

        Args:
            file_name (str): File path to save the plot.
            conf_matrix_cmap (str, optional): Colormap for confusion matrix heatmap.
            roc_curve_color (str, optional): Color for ROC curve plot.
            roc_curve_label (str, optional): Label for ROC curve plot.
            roc_curve_ls (str, optional): Line style for ROC curve plot.
            roc_curve_lw (int, optional): Line width for ROC curve plot.
            roc_curve_font_size (int, optional): Font size for ROC curve plot.
            conf_matrix_font_size (int, optional): Font size for confusion matrix heatmap.
        """
        self.plot_metrics(conf_matrix_cmap, roc_curve_color, roc_curve_label,
                          roc_curve_ls, roc_curve_lw, roc_curve_font_size, conf_matrix_font_size)
        plt.savefig(file_name)
        plt.close()

    def save_metrics_text(self, y_real, y_predicted, file_name):
        """
        Saves classification report, confusion matrix, and ROC metrics to a text file.

        Args:
            y_real (array): True labels.
            y_predicted (array): Predicted labels.
            file_name (str): File path to save the text file.
        """
        with open(file_name, 'w') as f:
            f.write(f'Classification Report:\n\n{classification_report(y_real, y_predicted)}\n\n')
            f.write(f'Confusion Matrix:\n\n{self.confusion_matrix}\n\n')

            if hasattr(self, 'roc_auc'):
                f.write(f'ROC AUC: {self.area_under_curve:.4f}\n')
                f.write(f'FPR: {self.false_predictions}\n')
                f.write(f'TPR: {self.true_predictions}\n')

    @property
    def matriz_cmap(self):
        return self._matriz_cmap

    @matriz_cmap.setter
    def matriz_cmap(self, value):
        self._matriz_cmap = value

    @property
    def roc_curve_color(self):
        return self._roc_curve_color

    @roc_curve_color.setter
    def roc_curve_color(self, value):
        self._roc_curve_color = value

    @property
    def y_score(self):
        return self._y_score

    @y_score.setter
    def y_score(self, value):
        self._y_score = value

    @property
    def roc_curve_label(self):
        return self._roc_curve_label

    @roc_curve_label.setter
    def roc_curve_label(self, value):
        self._roc_curve_label = value

    @property
    def roc_curve_ls(self):
        return self._roc_curve_ls

    @roc_curve_ls.setter
    def roc_curve_ls(self, value):
        self._roc_curve_ls = value

    @property
    def roc_curve_lw(self):
        return self._roc_curve_lw

    @roc_curve_lw.setter
    def roc_curve_lw(self, value):
        self._roc_curve_lw = value

    @property
    def roc_curve_font_size(self):
        return self._roc_curve_font_size

    @roc_curve_font_size.setter
    def roc_curve_font_size(self, value):
        self._roc_curve_font_size = value

    @property
    def confusion_matrix_font_size(self):
        return self._confusion_matrix_font_size

    @confusion_matrix_font_size.setter
    def confusion_matrix_font_size(self, value):
        self._confusion_matrix_font_size = value
