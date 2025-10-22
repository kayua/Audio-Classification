import numpy
import tensorflow
from typing import Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCamPlusPlus:
    """
    Grad-CAM++ (Gradient-weighted Class Activation Mapping++) implementation.

    This class implements the Grad-CAM++ algorithm which produces visual explanations
    for decisions from convolutional neural networks. It is an extension of Grad-CAM
    that provides better visual explanations with more detailed localization.

    Reference Paper:
        Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018).
        "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks."
        IEEE Winter Conference on Applications of Computer Vision (WACV).
        arXiv:1710.11063

    Mathematical Formulation:
        For a given class c, the Grad-CAM++ heatmap is computed as:

        L_{Grad-CAM++}^c = ReLU(∑_k α_k^c * A^k)

        where:
        - A^k is the k-th feature map of the target convolutional layer
        - α_k^c are weighting coefficients computed as:

          α_k^c = ∑_i∑_j [ (∂²Y^c/∂(A_ij^k)²) / (2*(∂²Y^c/∂(A_ij^k)²) + ∑_a∑_b A_ab^k * (∂³Y^c/∂(A_ij^k)³)) ] * ReLU(∂Y^c/∂A_ij^k)

        - Y^c is the score for class c before softmax
        - The higher-order derivatives provide better localization properties

    Attributes:
        gradcam_model (tensorflow.keras.Model): Model for computing Grad-CAM++
        model (tensorflow.keras.Model): Original model (if provided)
    """

    def __init__(self, model: Optional[tensorflow.keras.Model] = None):
        """
        Initialize GradCam++ instance.

        Args:
            model (tensorflow.keras.Model, optional): Pre-trained Keras model for
                which Grad-CAM++ explanations will be generated. If None, model
                must be provided when building the Grad-CAM++ model.

        Raises:
            TypeError: If model is provided but not a tensorflow.keras.Model instance
        """
        self.gradcam_model = None

        if model is not None:
            if not isinstance(model, tensorflow.keras.Model):
                raise TypeError("Model must be a tensorflow.keras.Model instance")
            self.model = model
        else:
            self.model = None

        logger.info("GradCam++ initialized")

    def build_gradcam_plus_plus_model(self,
                                      target_layer_name: str,
                                      model: Optional[tensorflow.keras.Model] = None) -> None:
        """
        Build Grad-CAM++ model that outputs both target layer activations and predictions.

        Args:
            target_layer_name (str): Name of the target convolutional layer for Grad-CAM++
            model (tensorflow.keras.Model, optional): Model to use if not provided during initialization

        Raises:
            ValueError: If no model is available or target layer is not found
            TypeError: If target_layer_name is not a string or model is invalid
        """
        try:
            if model is not None:
                if not isinstance(model, tensorflow.keras.Model):
                    raise TypeError("Model must be a tensorflow.keras.Model instance")
                self.model = model

            if self.model is None:
                raise ValueError("No model provided. Either provide during initialization or in this method.")

            if not isinstance(target_layer_name, str):
                raise TypeError("target_layer_name must be a string")

            # Get target layer
            try:
                target_layer = self.model.get_layer(target_layer_name)
            except ValueError as e:
                raise ValueError(
                    f"Layer '{target_layer_name}' not found in model. Available layers: {[layer.name for layer in self.model.layers]}") from e

            # Verify it's a convolutional layer
            if not any(layer_type in str(type(target_layer)) for layer_type in ['Conv2D', 'Convolution', 'Conv']):
                logger.warning(
                    f"Target layer '{target_layer_name}' may not be a convolutional layer. Results may be suboptimal.")

            # Create model that outputs both target layer and final predictions
            grad_model = tensorflow.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[target_layer.output, self.model.output]
            )

            self.gradcam_model = grad_model
            logger.info(f"Grad-CAM++ model built successfully with target layer: {target_layer_name}")

        except Exception as e:
            logger.error(f"Error building Grad-CAM++ model: {str(e)}")
            raise

    def compute_gradcam_plus_plus(self,
                                  input_sample: numpy.ndarray,
                                  class_idx: Optional[int] = None,
                                  target_layer_name: Optional[str] = None,
                                  model: Optional[tensorflow.keras.Model] = None) -> numpy.ndarray:
        """
        Compute Grad-CAM++ heatmap for given input sample and class.

        Args:
            input_sample (numpy.ndarray): Input data sample. Shape should be compatible
                with model input. For 2D data, will be expanded to (1, H, W) or (1, H, W, C).
            class_idx (int, optional): Class index for which to generate heatmap.
                If None, uses predicted class.
            target_layer_name (str, optional): Name of target convolutional layer.
                Required if Grad-CAM++ model not built.
            model (tensorflow.keras.Model, optional): Model to use if not provided earlier.

        Returns:
            numpy.ndarray: Grad-CAM++ heatmap with values in range [0, 1]

        Raises:
            ValueError: For invalid input shapes, missing model, or computation errors
            TypeError: For invalid input types
            RuntimeError: For tensorflow computation errors

        Example:
            >>> import tensorflow as tf
            >>> import numpy as np
            >>> from gradcam_plusplus import GradCam
            >>>
            >>> # Load a pre-trained model
            >>> model = tf.keras.applications.ResNet50(weights='imagenet')
            >>>
            >>> # Initialize Grad-CAM++
            >>> gradcam = GradCam(model=model)
            >>>
            >>> # Load and preprocess image
            >>> img = tf.keras.applications.resnet50.preprocess_input(
            ...     np.expand_dims(your_image, axis=0)
            >>> )
            >>>
            >>> # Generate heatmap for target class (e.g., class 285 for 'Egyptian cat')
            >>> heatmap = gradcam.compute_gradcam_plus_plus(
            ...     input_sample=img,
            ...     class_idx=285,
            ...     target_layer_name='conv5_block3_out'
            >>> )
            >>>
            >>> # Heatmap can be overlayed on original image for visualization
            >>> print(f"Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        """
        try:
            # Input validation
            if not isinstance(input_sample, numpy.ndarray):
                raise TypeError("input_sample must be a numpy.ndarray")

            if input_sample.size == 0:
                raise ValueError("input_sample cannot be empty")

            # Build model if needed
            if self.gradcam_model is None or target_layer_name is not None:
                if target_layer_name is None:
                    raise ValueError(
                        "target_layer_name must be provided when building Grad-CAM++ model for the first time")
                self.build_gradcam_plus_plus_model(target_layer_name, model)

            if self.gradcam_model is None:
                raise RuntimeError("Grad-CAM++ model failed to build")

            # Input shape processing
            original_shape = input_sample.shape
            processed_sample = self._preprocess_input_gradcam_plus_plus(input_sample)

            # Convert to tensor
            input_tensor = tensorflow.convert_to_tensor(processed_sample)

            # Compute gradients using higher-order derivatives
            with tensorflow.GradientTape(persistent=True) as tape3:
                with tensorflow.GradientTape(persistent=True) as tape2:
                    with tensorflow.GradientTape(persistent=True) as tape1:
                        tape1.watch(input_tensor)
                        tape2.watch(input_tensor)
                        tape3.watch(input_tensor)

                        # Forward pass
                        layer_output, predictions = self.gradcam_model(input_tensor)

                        # Determine class index
                        if class_idx is None:
                            class_idx = tensorflow.argmax(predictions[0]).numpy()
                            logger.info(f"Using predicted class index: {class_idx}")

                        if class_idx < 0 or class_idx >= predictions.shape[-1]:
                            raise ValueError(f"class_idx {class_idx} out of range [0, {predictions.shape[-1] - 1}]")

                        class_score = predictions[:, class_idx]

                    # First order gradients
                    grads = tape1.gradient(class_score, layer_output)
                    if grads is None:
                        raise RuntimeError(
                            "Could not compute first-order gradients. Check if model and layer are compatible.")

                # Second order gradients
                grads_2 = tape2.gradient(grads, layer_output)
                if grads_2 is None:
                    raise RuntimeError("Could not compute second-order gradients.")

            # Third order gradients
            grads_3 = tape3.gradient(grads_2, layer_output)
            if grads_3 is None:
                raise RuntimeError("Could not compute third-order gradients.")

            # Clean up persistent tapes
            del tape1, tape2, tape3

            # Compute Grad-CAM++ weights using the mathematical formulation
            weights = self._compute_gradcam_plusplus_weights(layer_output, grads, grads_2, grads_3)

            # Generate heatmap
            heatmap = self._generate_heatmap(layer_output, weights)

            logger.info(f"Grad-CAM++ computed successfully for class {class_idx}. "
                        f"Input shape: {original_shape}, Heatmap shape: {heatmap.shape}")

            return heatmap

        except tensorflow.errors.InvalidArgumentError as e:
            logger.error(f"TensorFlow computation error: {str(e)}")
            raise RuntimeError(f"TensorFlow computation failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error computing Grad-CAM++: {str(e)}")
            raise

    @staticmethod
    def _preprocess_input_gradcam_plus_plus(input_sample: numpy.ndarray) -> numpy.ndarray:
        """
        Preprocess input sample to ensure correct shape and type.

        Args:
            input_sample (numpy.ndarray): Input sample to preprocess

        Returns:
            numpy.ndarray: Preprocessed input sample

        Raises:
            ValueError: If input shape is invalid
        """
        input_sample = input_sample.astype(numpy.float32)

        # Handle different input dimensions
        if len(input_sample.shape) == 1:
            raise ValueError("1D input not supported. Expected 2D, 3D, or 4D input.")

        elif len(input_sample.shape) == 2:
            # Assume (H, W) -> (1, H, W, 1) or (1, H, W) depending on model
            input_sample = numpy.expand_dims(input_sample, axis=0)
            logger.warning("2D input expanded to 3D. Consider providing input in model-compatible shape.")

        elif len(input_sample.shape) == 3:
            if input_sample.shape[0] != 1:
                # Assume (H, W, C) -> (1, H, W, C) or (H, W, C) with batch size 1
                input_sample = input_sample[numpy.newaxis, ...]
                logger.info("3D input expanded with batch dimension")

        elif len(input_sample.shape) == 4:
            # Already batched, use first sample if batch size > 1
            if input_sample.shape[0] > 1:
                logger.warning(f"Using first sample from batch of {input_sample.shape[0]}")
                input_sample = input_sample[0:1]

        else:
            raise ValueError(f"Unsupported input dimension: {len(input_sample.shape)}D")

        return input_sample

    @staticmethod
    def _compute_gradcam_plusplus_weights(layer_output: tensorflow.Tensor,
                                          grads: tensorflow.Tensor,
                                          grads_2: tensorflow.Tensor,
                                          grads_3: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Compute Grad-CAM++ weighting coefficients using higher-order derivatives.

        Implements the mathematical formulation from the Grad-CAM++ paper:
        α_k^c = ∑_i∑_j [ (∂²Y^c/∂(A_ij^k)²) / (2*(∂²Y^c/∂(A_ij^k)²) + ∑_a∑_b A_ab^k * (∂³Y^c/∂(A_ij^k)³)) ] * ReLU(∂Y^c/∂A_ij^k)

        Args:
            layer_output: Target layer activations
            grads: First-order gradients (∂Y^c/∂A_ij^k)
            grads_2: Second-order gradients (∂²Y^c/∂(A_ij^k)²)
            grads_3: Third-order gradients (∂³Y^c/∂(A_ij^k)³)

        Returns:
            Weights for Grad-CAM++ computation
        """
        # Determine reduction axis based on layer output shape
        if len(layer_output.shape) == 3:
            reduce_axis = 2  # For 1D convolutional layers
        elif len(layer_output.shape) == 4:
            reduce_axis = 3  # For 2D convolutional layers
        else:
            reduce_axis = -1  # Fallback

        try:
            # Compute numerator and denominator for alpha coefficients
            numerator = grads_2
            denominator = (2.0 * grads_2 +
                           tensorflow.reduce_sum(layer_output * grads_3,
                                                 axis=reduce_axis, keepdims=True) +
                           1e-10)  # Small epsilon for numerical stability

            alpha = numerator / denominator

            # Apply ReLU to gradients and compute weights
            relu_grads = tensorflow.maximum(grads, 0.0)
            weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))

            return weights

        except tensorflow.errors.InvalidArgumentError as e:
            logger.error("Error in weight computation. Check layer compatibility.")
            raise

    @staticmethod
    def _generate_heatmap(layer_output: tensorflow.Tensor,
                          weights: tensorflow.Tensor) -> numpy.ndarray:
        """
        Generate final heatmap from layer outputs and computed weights.

        Args:
            layer_output: Target layer activations
            weights: Computed Grad-CAM++ weights

        Returns:
            Normalized heatmap as numpy array
        """
        try:
            # Remove batch dimension and compute weighted combination
            layer_output_squeezed = layer_output[0]

            # Compute heatmap: L = ReLU(∑_k α_k^c * A^k)
            heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
            heatmap = tensorflow.squeeze(heatmap)

            # Apply ReLU and normalize
            heatmap = tensorflow.maximum(heatmap, 0)  # ReLU
            heatmap_max = tensorflow.math.reduce_max(heatmap)

            if heatmap_max > 1e-10:
                heatmap = heatmap / heatmap_max
            else:
                logger.warning("Heatmap maximum is near zero - results may not be meaningful")

            return heatmap.numpy()

        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            raise