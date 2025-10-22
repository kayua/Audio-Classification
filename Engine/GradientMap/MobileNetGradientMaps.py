try:
    import sys
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Activation

    from Engine.Models.Process.MobileNet_Process import MobileNetProcess

except ImportError as error:
    print(error)
    sys.exit(-1)



class MobileNetGradientMaps:

    def __init__(self):

        pass


    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Build an auxiliary model for GradCAM/GradCAM++ computation.

        Args:
            target_layer_name: Name of target layer. If None, uses last pointwise conv layer
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating GradCAM model")

        # Find last pointwise convolution layer if no target specified
        if target_layer_name is None:
            # Get last block's pointwise conv
            last_block_idx = len(self.filters_per_block) - 1
            target_layer_name = f'pointwise_conv_block_{last_block_idx}'

        target_layer = self.neural_network_model.get_layer(target_layer_name)

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

    def compute_gradcam_plusplus(self, input_sample: numpy.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> numpy.ndarray:
        """
        Compute Grad-CAM++ heatmap (improved version with better localization).

        Args:
            input_sample: Input image (2D or 3D array)
            class_idx: Target class index (if None, uses predicted class)
            target_layer_name: Target layer name (if None, uses default)

        Returns:
            Normalized heatmap as numpy array
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct input shape
        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=-1)

        if len(input_sample.shape) == 3:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape1:

            with tensorflow.GradientTape() as tape2:

                with tensorflow.GradientTape() as tape3:
                    layer_output, predictions = self.gradcam_model(input_tensor)

                    if class_idx is None:
                        class_idx = tensorflow.argmax(predictions[0]).numpy()

                    class_score = predictions[:, class_idx]

                grads = tape3.gradient(class_score, layer_output)

            grads_2 = tape2.gradient(grads, layer_output)

        grads_3 = tape1.gradient(grads_2, layer_output)

        reduce_axis = 3 if len(layer_output.shape) == 4 else -1

        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(layer_output * grads_3,
                                                            axis=reduce_axis, keepdims=True) + 1e-10

        alpha = numerator / denominator

        relu_grads = tensorflow.maximum(grads, 0.0)

        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))

        layer_output_squeezed = layer_output[0]
        heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)

        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_gradcam(self, input_sample: numpy.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> numpy.ndarray:
        """
        Standard Grad-CAM computation.

        Args:
            input_sample: Input image
            class_idx: Target class index
            target_layer_name: Target layer name

        Returns:
            Normalized heatmap
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=-1)
        if len(input_sample.shape) == 3:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        # Pool gradients
        pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # Normalize
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_scorecam(self, input_sample: numpy.ndarray, class_idx: int = None,
                         target_layer_name: str = None, batch_size: int = 32) -> numpy.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method).

        Args:
            input_sample: Input image
            class_idx: Target class index
            target_layer_name: Target layer name
            batch_size: Batch size for processing activation maps

        Returns:
            Normalized heatmap
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=-1)
        if len(input_sample.shape) == 3:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        # Get activations
        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        # Get activation maps
        activations = layer_output[0].numpy()
        num_channels = activations.shape[-1]

        weights = []
        for i in range(num_channels):
            act_map = activations[:, :, i]

            # Normalize to [0, 1]
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            zoom_factors = (input_sample.shape[1] / act_map.shape[0],
                            input_sample.shape[2] / act_map.shape[1])
            upsampled = zoom(act_map, zoom_factors, order=1)

            # Handle channel dimension
            if input_sample.shape[-1] == 1:
                upsampled = upsampled[:, :, numpy.newaxis]
            elif input_sample.shape[-1] == 3:
                upsampled = numpy.repeat(upsampled[:, :, numpy.newaxis], 3, axis=-1)

            # Mask input
            masked_input = input_sample[0] * upsampled
            masked_input = numpy.expand_dims(masked_input, 0)

            # Get score for masked input
            masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
            score = masked_pred[0, class_idx]

            weights.append(score)

        weights = numpy.array(weights)
        weights = numpy.maximum(weights, 0)

        # Weighted combination
        heatmap = numpy.tensordot(activations, weights, axes=([2], [0]))

        heatmap = numpy.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap
