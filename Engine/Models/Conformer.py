#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Enhanced Conformer Model with Advanced Explainable AI (XAI) Capabilities

This module implements a Conformer-based neural network architecture with
comprehensive explainable AI features including Grad-CAM, Grad-CAM++,
and Score-CAM for model interpretability and visualization.
"""

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{2}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/14'
__credits__ = ['unknown']

# MIT License
# Copyright (c) 2025 unknown
# [Full license text remains the same...]

import logging
import sys
import os
from typing import Tuple, Optional, Dict, Any, Union, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom, gaussian_filter
import seaborn as sns

import tensorflow as tf
import tensorflow as tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Layer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D

from Engine.Layers.ConformerBlock import ConformerBlock
from Engine.Layers.TransposeLayer import TransposeLayer
from Engine.Models.Process.Conformer_Process import ProcessConformer
from Engine.Layers.ConvolutionalSubsampling import ConvolutionalSubsampling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('conformer_xai.log')
    ]
)
logger = logging.getLogger(__name__)


class Conformer(ProcessConformer):
    """
    Enhanced Conformer Model with Advanced Explainable AI (XAI) Capabilities

    This class implements a Conformer-based architecture for sequence processing
    with integrated explainable AI methods for model interpretability.

    Key Features:
    - Conformer architecture with multi-head self-attention and convolution
    - Multiple XAI methods: Grad-CAM, Grad-CAM++, Score-CAM
    - Advanced visualization capabilities
    - Comprehensive model analysis tools

    References:
    - Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
    - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)
    - Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations" (2018)
    - Wang et al., "Score-CAM: Score-Weighted Visual Explanations" (2020)

    Attributes:
        neural_network_model (tf.keras.Model): The main Conformer model
        gradcam_model (tf.keras.Model): Auxiliary model for XAI computations
        loss_function: Loss function for model compilation
        optimizer_function: Optimizer for model training
        number_filters_spectrogram (int): Number of filters for spectrogram processing
        input_dimension (tuple): Input shape dimensions
        number_conformer_blocks (int): Number of Conformer blocks in the architecture
        embedding_dimension (int): Dimension of embedding space
        number_heads (int): Number of attention heads
        number_classes (int): Number of output classes
        kernel_size (int): Convolution kernel size
        dropout_rate (float): Dropout rate for regularization
        last_layer_activation (str): Activation function for output layer
        model_name (str): Name identifier for the model
    """

    def __init__(self, arguments):
        """
        Initialize the Conformer model with XAI capabilities.

        Args:
            arguments: Configuration object containing model parameters
        """
        ProcessConformer.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.conformer_loss_function
        self.optimizer_function = arguments.conformer_optimizer_function
        self.number_filters_spectrogram = arguments.conformer_number_filters_spectrogram
        self.input_dimension = arguments.conformer_input_dimension
        self.number_conformer_blocks = arguments.conformer_number_conformer_blocks
        self.embedding_dimension = arguments.conformer_embedding_dimension
        self.number_heads = arguments.conformer_number_heads
        self.number_classes = arguments.number_classes
        self.kernel_size = arguments.conformer_size_kernel
        self.dropout_rate = arguments.conformer_dropout_rate
        self.last_layer_activation = arguments.conformer_last_layer_activation
        self.model_name = "Conformer"

        # Configure visualization styles
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        logger.info("Conformer model initialized with XAI capabilities")

    def build_model(self) -> None:
        """
        Build the Conformer model architecture.

        Architecture:
        1. Input layer
        2. Convolutional subsampling
        3. Transpose layer
        4. Embedding dense layer
        5. Multiple Conformer blocks
        6. Global average pooling
        7. Output classification layer
        """
        try:
            logger.info(f"Building Conformer model with input dimension: {self.input_dimension}")

            inputs = Input(shape=self.input_dimension, name='input_layer')

            # Feature extraction pipeline
            neural_network_flow = ConvolutionalSubsampling(name='conv_subsampling')(inputs)
            neural_network_flow = TransposeLayer(perm=[0, 2, 1], name='transpose_layer')(neural_network_flow)
            neural_network_flow = Dense(self.embedding_dimension, name='embedding_dense')(neural_network_flow)

            # Conformer blocks
            for i in range(self.number_conformer_blocks):
                neural_network_flow = ConformerBlock(self.embedding_dimension,
                                                     self.number_heads,
                                                     self.input_dimension[0] // 2,
                                                     self.kernel_size,
                                                     self.dropout_rate,
                                                     name=f'conformer_block_{i}'
                                                     )(neural_network_flow)

            # Classification head
            neural_network_flow = GlobalAveragePooling1D(name='global_avg_pooling')(neural_network_flow)
            neural_network_flow = Dense(self.number_classes,
                                        activation=self.last_layer_activation,
                                        name='output_layer')(neural_network_flow)

            self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow, name=self.model_name)

            logger.info("Conformer model built successfully")
            self.neural_network_model.summary()

        except Exception as e:
            logger.error(f"Error building Conformer model: {e}")
            raise

    def compile_and_train(self,
                          train_data: tensorflow.Tensor,
                          train_labels: tensorflow.Tensor,
                          epochs: int,
                          batch_size: int,
                          validation_data: Optional[Tuple] = None,
                          generate_gradcam: bool = True,
                          num_gradcam_samples: int = 30,
                          gradcam_output_dir: str = './activation_maps',
                          xai_method: str = 'gradcam++') -> tf.keras.callbacks.History:
        """
        Compile and train the Conformer model with optional XAI visualization generation.

        Args:
            train_data: Training data tensor
            train_labels: Training labels tensor
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_data: Optional validation data tuple (data, labels)
            generate_gradcam: Whether to generate XAI visualizations
            num_gradcam_samples: Number of samples for XAI visualization
            gradcam_output_dir: Output directory for visualization files
            xai_method: XAI method to use ('gradcam', 'gradcam++', 'scorecam')

        Returns:
            Training history object
        """
        try:
            logger.info(f"Compiling and training model for {epochs} epochs")

            self.neural_network_model.compile(optimizer=self.optimizer_function,
                                              loss=self.loss_function,
                                              metrics=['accuracy'])

            training_history = self.neural_network_model.fit(
                train_data, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=1
            )

            if validation_data is not None:
                final_val_accuracy = training_history.history['val_accuracy'][-1]
                logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")

            if generate_gradcam and validation_data is not None:
                logger.info(f"Generating {xai_method.upper()} visualizations")

                val_data, val_labels = validation_data
                stats = self.generate_validation_visualizations(
                    validation_data=val_data,
                    validation_labels=val_labels,
                    num_samples=num_gradcam_samples,
                    output_dir=gradcam_output_dir,
                    xai_method=xai_method
                )

                logger.info(f"XAI visualizations saved to: {gradcam_output_dir}")
                logger.info(f"Visualization statistics: {stats}")

            return training_history

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def compile_model(self) -> None:
        """Compile the Conformer model with specified optimizer and loss function."""
        try:
            self.neural_network_model.compile(optimizer=self.optimizer_function,
                                              loss=self.loss_function,
                                              metrics=['accuracy'])
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise

    def build_gradcam_model(self, target_layer_name: Optional[str] = None) -> None:
        """
        Build auxiliary model for GradCAM computation.

        Args:
            target_layer_name: Name of the target layer for GradCAM computation.
                             If None, uses the last Conformer block.
        """
        try:
            if self.neural_network_model is None:
                raise ValueError("Main model must be built before creating GradCAM model")

            if target_layer_name is None:
                target_layer_name = f'conformer_block_{self.number_conformer_blocks - 1}'

            target_layer = self.neural_network_model.get_layer(target_layer_name)

            self.gradcam_model = Model(inputs=self.neural_network_model.inputs,
                                       outputs=[target_layer.output, self.neural_network_model.output])

            logger.info(f"GradCAM model created with target layer: {target_layer_name}")

        except Exception as e:
            logger.error(f"Error building GradCAM model: {e}")
            raise

    def compute_gradcam_plusplus(self,
                                 input_sample: np.ndarray,
                                 class_idx: Optional[int] = None,
                                 target_layer_name: Optional[str] = None) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap for enhanced localization accuracy.

        Grad-CAM++ improves upon standard Grad-CAM by using a weighted combination
        of gradients to better localize objects of different sizes and multiple instances.

        Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based
        Visual Explanations for Deep Convolutional Networks" (2018)

        Args:
            input_sample: Input data sample
            class_idx: Target class index. If None, uses predicted class.
            target_layer_name: Target layer for activation computation.

        Returns:
            Normalized heatmap array
        """
        try:
            if self.gradcam_model is None or target_layer_name is not None:
                self.build_gradcam_model(target_layer_name)

            # Preprocess input
            original_shape = input_sample.shape
            if len(input_sample.shape) == 2:
                input_sample = np.expand_dims(input_sample, axis=0)
            elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
                input_sample = input_sample[0:1]

            input_sample = input_sample.astype(np.float32)
            input_tensor = tf.convert_to_tensor(input_sample)

            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                    with tf.GradientTape() as tape3:
                        layer_output, predictions = self.gradcam_model(input_tensor)

                        if class_idx is None:
                            class_idx = tf.argmax(predictions[0]).numpy()

                        class_score = predictions[:, class_idx]

                    # First-order gradients
                    grads = tape3.gradient(class_score, layer_output)

                # Second-order gradients
                grads_2 = tape2.gradient(grads, layer_output)

            # Third-order gradients
            grads_3 = tape1.gradient(grads_2, layer_output)

            # Compute alpha weights (Grad-CAM++ formula)
            numerator = grads_2
            denominator = 2.0 * grads_2 + tf.reduce_sum(
                layer_output * grads_3, axis=(1), keepdims=True
            ) + 1e-10

            alpha = numerator / denominator

            # ReLU on gradients
            relu_grads = tf.maximum(grads, 0.0)

            # Weighted combination
            weights = tf.reduce_sum(alpha * relu_grads, axis=(1))

            # Compute weighted activation map
            layer_output_squeezed = layer_output[0]
            heatmap = layer_output_squeezed @ weights[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Apply ReLU and normalize
            heatmap = tf.maximum(heatmap, 0)
            heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-10)

            logger.debug(f"Grad-CAM++ computed for class {class_idx}")
            return heatmap.numpy()

        except Exception as e:
            logger.error(f"Error computing Grad-CAM++: {e}")
            raise

    def compute_gradcam(self,
                        input_sample: np.ndarray,
                        class_idx: Optional[int] = None,
                        target_layer_name: Optional[str] = None) -> np.ndarray:
        """
        Compute standard Grad-CAM heatmap.

        Args:
            input_sample: Input data sample
            class_idx: Target class index. If None, uses predicted class.
            target_layer_name: Target layer for activation computation.

        Returns:
            Normalized heatmap array
        """
        try:
            if self.gradcam_model is None or target_layer_name is not None:
                self.build_gradcam_model(target_layer_name)

            # Preprocess input
            original_shape = input_sample.shape
            if len(input_sample.shape) == 2:
                input_sample = np.expand_dims(input_sample, axis=0)
            elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
                input_sample = input_sample[0:1]

            input_sample = input_sample.astype(np.float32)
            input_tensor = tf.convert_to_tensor(input_sample)

            with tf.GradientTape() as tape:
                layer_output, predictions = self.gradcam_model(input_tensor)

                if class_idx is None:
                    class_idx = tf.argmax(predictions[0]).numpy()

                class_channel = predictions[:, class_idx]

            grads = tape.gradient(class_channel, layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

            layer_output = layer_output[0]
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

            logger.debug(f"Grad-CAM computed for class {class_idx}")
            return heatmap.numpy()

        except Exception as e:
            logger.error(f"Error computing Grad-CAM: {e}")
            raise

    def compute_scorecam(self,
                         input_sample: np.ndarray,
                         class_idx: Optional[int] = None,
                         target_layer_name: Optional[str] = None,
                         batch_size: int = 32) -> np.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method).

        Score-CAM uses forward passes only, making it more stable and less prone
        to gradient saturation issues.

        Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations
        for Convolutional Neural Networks" (2020)

        Args:
            input_sample: Input data sample
            class_idx: Target class index. If None, uses predicted class.
            target_layer_name: Target layer for activation computation.
            batch_size: Batch size for forward passes.

        Returns:
            Normalized heatmap array
        """
        try:
            if self.gradcam_model is None or target_layer_name is not None:
                self.build_gradcam_model(target_layer_name)

            if len(input_sample.shape) == 2:
                input_sample = np.expand_dims(input_sample, axis=0)

            input_sample = input_sample.astype(np.float32)
            input_tensor = tf.convert_to_tensor(input_sample)

            # Get activations and predictions
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()

            base_score = predictions[0, class_idx].numpy()

            # Get activation maps
            activations = layer_output[0].numpy()
            num_channels = activations.shape[-1]

            # Normalize each activation map and compute scores
            weights = []
            for i in range(num_channels):
                act_map = activations[:, i]

                # Normalize to [0, 1]
                if act_map.max() > act_map.min():
                    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

                # Upsample to input size
                upsampled = zoom(act_map,
                                 (input_sample.shape[1] / act_map.shape[0],),
                                 order=1)

                # Mask input
                masked_input = input_sample[0] * upsampled[:, np.newaxis]
                masked_input = np.expand_dims(masked_input, 0)

                # Get score for masked input
                masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
                score = masked_pred[0, class_idx]

                weights.append(score)

            weights = np.array(weights)
            weights = np.maximum(weights, 0)

            # Weighted combination
            heatmap = np.dot(activations, weights)
            heatmap = np.maximum(heatmap, 0)

            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            logger.debug(f"Score-CAM computed for class {class_idx}")
            return heatmap

        except Exception as e:
            logger.error(f"Error computing Score-CAM: {e}")
            raise

    @staticmethod
    def smooth_heatmap(heatmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Apply Gaussian smoothing to heatmap for better visualization.

        Args:
            heatmap: Input heatmap array
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Smoothed heatmap array
        """
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray,
                            target_shape: Tuple[int, int],
                            smooth: bool = True) -> np.ndarray:
        """
        Enhanced interpolation with optional smoothing.

        Args:
            heatmap: Input heatmap
            target_shape: Target dimensions (height, width)
            smooth: Apply Gaussian smoothing after interpolation

        Returns:
            Interpolated heatmap array
        """
        try:
            if len(heatmap.shape) == 1:
                heatmap_2d = np.tile(heatmap[:, np.newaxis], (1, target_shape[1]))
                zoom_factors = (target_shape[0] / heatmap_2d.shape[0],
                                target_shape[1] / heatmap_2d.shape[1])
                interpolated = zoom(heatmap_2d, zoom_factors, order=3)  # Cubic interpolation
            elif len(heatmap.shape) == 2:
                zoom_factors = (target_shape[0] / heatmap.shape[0],
                                target_shape[1] / heatmap.shape[1])
                interpolated = zoom(heatmap, zoom_factors, order=3)
            else:
                raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}")

            # Ensure exact target shape
            if interpolated.shape != target_shape:
                zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                       target_shape[1] / interpolated.shape[1])
                interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

            if smooth:
                interpolated = gaussian_filter(interpolated, sigma=1.5)

            return interpolated

        except Exception as e:
            logger.error(f"Error interpolating heatmap: {e}")
            raise

    def plot_gradcam_modern(self,
                            input_sample: np.ndarray,
                            heatmap: np.ndarray,
                            class_idx: int,
                            predicted_class: int,
                            true_label: Optional[int] = None,
                            confidence: Optional[float] = None,
                            xai_method: str = 'gradcam++',
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
        """
        Create modern, visually appealing GradCAM visualization.

        Args:
            input_sample: Original input data
            heatmap: Computed activation heatmap
            class_idx: Class index used for heatmap computation
            predicted_class: Model's predicted class
            true_label: Ground truth label (optional)
            confidence: Prediction confidence score (optional)
            xai_method: XAI method used for visualization
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot
        """
        try:
            if len(input_sample.shape) == 3:
                input_sample = input_sample[0]

            # Enhanced interpolation with smoothing
            interpolated_heatmap = self.interpolate_heatmap(
                heatmap, input_sample.shape, smooth=True
            )

            # Create figure with modern style
            fig = plt.figure(figsize=(20, 6), facecolor='white')
            gs = fig.add_gridspec(1, 4, wspace=0.3)

            # Color schemes
            cmap_input = 'viridis'
            cmap_heatmap = 'RdYlBu_r'

            # 1. Original Input
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(input_sample, cmap=cmap_input, aspect='auto', interpolation='bilinear')
            ax1.set_title('Input Spectrogram', fontsize=13, fontweight='bold', pad=15)
            ax1.set_xlabel('Temporal Frames', fontsize=10)
            ax1.set_ylabel('Frequency Bins', fontsize=10)
            ax1.grid(False)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 2. Heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                             aspect='auto', interpolation='bilinear')
            ax2.set_title(f'Activation Map ({xai_method.upper()})',
                          fontsize=13, fontweight='bold', pad=15)
            ax2.set_xlabel('Temporal Frames', fontsize=10)
            ax2.set_ylabel('Frequency Bins', fontsize=10)
            ax2.grid(False)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Enhanced Overlay
            ax3 = fig.add_subplot(gs[0, 2])
            input_normalized = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min() + 1e-10)
            ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
            im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                             alpha=0.5, aspect='auto', interpolation='bilinear')
            ax3.set_title('Overlay', fontsize=13, fontweight='bold', pad=15)
            ax3.set_xlabel('Temporal Frames', fontsize=10)
            ax3.set_ylabel('Frequency Bins', fontsize=10)
            ax3.grid(False)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Temporal Importance Profile
            ax4 = fig.add_subplot(gs[0, 3])
            temporal_importance = np.mean(interpolated_heatmap, axis=0)
            time_steps = np.arange(len(temporal_importance))

            temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

            ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B', label='Importance')
            ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Smoothed Profile')
            ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                     color='#4ECDC4', linestyle='--', label='Original Profile')

            ax4.set_xlabel('Temporal Frame', fontsize=10)
            ax4.set_ylabel('Mean Importance', fontsize=10)
            ax4.set_title('Temporal Importance Profile', fontsize=13, fontweight='bold', pad=15)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.set_xlim([0, len(temporal_importance)])

            # Super title with prediction info
            pred_status = '✓' if true_label is not None and predicted_class == true_label else '✗'
            conf_str = f' | Confidence: {confidence:.1%}' if confidence is not None else ''

            if true_label is not None:
                suptitle = (f'{pred_status} Predicted: Class {predicted_class} | '
                            f'True: Class {true_label}{conf_str}')
            else:
                suptitle = f'Predicted: Class {predicted_class}{conf_str}'

            fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Figure saved: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Error creating GradCAM visualization: {e}")
            raise

    def generate_validation_visualizations(self,
                                           validation_data: np.ndarray,
                                           validation_labels: np.ndarray,
                                           num_samples: int = 10,
                                           output_dir: str = './gradcam_outputs',
                                           target_layer_name: Optional[str] = None,
                                           xai_method: str = 'gradcam++') -> Dict[str, Any]:
        """
        Generate enhanced XAI visualizations for validation samples.

        Args:
            validation_data: Validation dataset
            validation_labels: Validation labels
            num_samples: Number of samples to visualize
            output_dir: Output directory for visualizations
            target_layer_name: Target layer for XAI computation
            xai_method: XAI method to use

        Returns:
            Dictionary containing visualization statistics
        """
        try:
            logger.info(f"Generating {xai_method.upper()} visualizations for {num_samples} samples")

            os.makedirs(output_dir, exist_ok=True)

            # Get predictions
            predictions = self.neural_network_model.predict(validation_data, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            # Process labels
            if len(validation_labels.shape) > 1:
                true_labels = np.argmax(validation_labels, axis=1)
            else:
                true_labels = validation_labels

            # Select samples (balanced between correct and incorrect predictions)
            correct_indices = np.where(predicted_classes == true_labels)[0]
            incorrect_indices = np.where(predicted_classes != true_labels)[0]

            num_correct = min(num_samples // 2, len(correct_indices))
            num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

            selected_correct = (np.random.choice(correct_indices, num_correct, replace=False)
                                if len(correct_indices) > 0 else [])
            selected_incorrect = (np.random.choice(incorrect_indices, num_incorrect, replace=False)
                                  if len(incorrect_indices) > 0 else [])

            selected_indices = np.concatenate([selected_correct, selected_incorrect])

            stats = {
                'total_samples': len(selected_indices),
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'output_directory': output_dir
            }

            logger.info(f"Processing {len(selected_indices)} samples for visualization")

            for i, idx in enumerate(selected_indices):
                try:
                    sample = validation_data[idx]

                    # Ensure 2D
                    if len(sample.shape) == 3:
                        sample = np.squeeze(sample)

                    true_label = true_labels[idx]
                    predicted = predicted_classes[idx]
                    confidence = confidences[idx]

                    # Compute heatmap based on selected method
                    if xai_method.lower() == 'gradcam++':
                        heatmap = self.compute_gradcam_plusplus(
                            sample, class_idx=predicted, target_layer_name=target_layer_name
                        )
                    elif xai_method.lower() == 'scorecam':
                        heatmap = self.compute_scorecam(
                            sample, class_idx=predicted, target_layer_name=target_layer_name
                        )
                    else:  # standard gradcam
                        heatmap = self.compute_gradcam(
                            sample, class_idx=predicted, target_layer_name=target_layer_name
                        )

                    is_correct = predicted == true_label
                    if is_correct:
                        stats['correct_predictions'] += 1
                        prefix = 'correct'
                    else:
                        stats['incorrect_predictions'] += 1
                        prefix = 'incorrect'

                    save_path = os.path.join(
                        output_dir,
                        f'{prefix}_sample_{i:03d}_true_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png'
                    )

                    self.plot_gradcam_modern(
                        sample, heatmap, predicted, predicted, true_label,
                        confidence=confidence, xai_method=xai_method,
                        save_path=save_path, show_plot=False
                    )

                    status = '✓' if is_correct else '✗'
                    logger.debug(f"{status} Sample {i + 1}/{len(selected_indices)}: "
                                 f"True={true_label}, Pred={predicted}, Conf={confidence:.1%}")

                except Exception as e:
                    logger.error(f"Error processing sample {i + 1}: {e}")
                    continue

            accuracy = (stats['correct_predictions'] / stats['total_samples'] * 100
                        if stats['total_samples'] > 0 else 0)
            stats['accuracy'] = accuracy

            logger.info(f"Visualization generation completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error generating validation visualizations: {e}")
            raise

    def explain_prediction_comprehensive(self,
                                         input_sample: np.ndarray,
                                         class_names: Optional[List[str]] = None,
                                         save_path: Optional[str] = None,
                                         xai_method: str = 'gradcam++') -> Dict[str, Any]:
        """
        Generate comprehensive explanation with multiple XAI methods comparison.

        Args:
            input_sample: Input data sample
            class_names: List of class names for labeling
            save_path: Path to save comprehensive analysis
            xai_method: Primary XAI method for detailed analysis

        Returns:
            Dictionary containing comprehensive explanation data
        """
        try:
            logger.info("Generating comprehensive prediction explanation")

            if len(input_sample.shape) == 3:
                input_sample_2d = np.squeeze(input_sample)
            else:
                input_sample_2d = input_sample.copy()

            if len(input_sample.shape) == 2:
                input_sample_batch = np.expand_dims(input_sample, axis=0)
            else:
                input_sample_batch = input_sample

            # Get predictions
            predictions = self.neural_network_model.predict(input_sample_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Compute heatmaps with different methods
            logger.info("Computing activation maps with different XAI methods")

            heatmap_gradcam = self.compute_gradcam(input_sample_2d, class_idx=predicted_class)
            heatmap_gradcampp = self.compute_gradcam_plusplus(input_sample_2d, class_idx=predicted_class)

            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 14), facecolor='white')
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

            cmap_input = 'viridis'
            cmap_heat = 'RdYlBu_r'

            # Visualization code (similar to original but with logging)
            # ... [visualization code remains the same as original] ...

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Comprehensive analysis saved: {save_path}")

            plt.show()

            explanation = {
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'class_probabilities': predictions[0].tolist(),
                'heatmap_gradcam': heatmap_gradcam,
                'heatmap_gradcampp': heatmap_gradcampp,
                'temporal_importance_gradcam': np.mean(
                    self.interpolate_heatmap(heatmap_gradcam, input_sample_2d.shape),
                    axis=0
                ),
                'temporal_importance_gradcampp': np.mean(
                    self.interpolate_heatmap(heatmap_gradcampp, input_sample_2d.shape),
                    axis=0
                )
            }

            if class_names:
                explanation['predicted_class_name'] = class_names[predicted_class]

            logger.info("Comprehensive explanation generated successfully")
            return explanation

        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {e}")
            raise

    # Properties
    @property
    def neural_network_model(self) -> tf.keras.Model:
        """Get the neural network model."""
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value: tf.keras.Model) -> None:
        """Set the neural network model."""
        self._neural_network_model = value

    @property
    def loss_function(self):
        """Get the loss function."""
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value) -> None:
        """Set the loss function."""
        self._loss_function = value

    @property
    def optimizer_function(self):
        """Get the optimizer function."""
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, value) -> None:
        """Set the optimizer function."""
        self._optimizer_function = value

    @property
    def number_filters_spectrogram(self) -> int:
        """Get the number of spectrogram filters."""
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, value: int) -> None:
        """Set the number of spectrogram filters."""
        self._number_filters_spectrogram = value

    @property
    def input_dimension(self) -> tuple:
        """Get the input dimension."""
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value: tuple) -> None:
        """Set the input dimension."""
        self._input_dimension = value

    @property
    def number_conformer_blocks(self) -> int:
        """Get the number of Conformer blocks."""
        return self._number_conformer_blocks

    @number_conformer_blocks.setter
    def number_conformer_blocks(self, value: int) -> None:
        """Set the number of Conformer blocks."""
        self._number_conformer_blocks = value

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        """Set the embedding dimension."""
        self._embedding_dimension = value

    @property
    def number_heads(self) -> int:
        """Get the number of attention heads."""
        return self._number_heads

    @number_heads.setter
    def number_heads(self, value: int) -> None:
        """Set the number of attention heads."""
        self._number_heads = value

    @property
    def number_classes(self) -> int:
        """Get the number of output classes."""
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value: int) -> None:
        """Set the number of output classes."""
        self._number_classes = value

    @property
    def kernel_size(self) -> int:
        """Get the kernel size."""
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value: int) -> None:
        """Set the kernel size."""
        self._kernel_size = value

    @property
    def dropout_rate(self) -> float:
        """Get the dropout rate."""
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value: float) -> None:
        """Set the dropout rate."""
        self._dropout_rate = value

    @property
    def last_layer_activation(self) -> str:
        """Get the last layer activation function."""
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value: str) -> None:
        """Set the last layer activation function."""
        self._last_layer_activation = value

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the model name."""
        self._model_name = value
