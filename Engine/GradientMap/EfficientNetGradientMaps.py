try:
    import sys
    import numpy
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import GlobalAveragePooling2D

    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications import EfficientNetB1
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications import EfficientNetB5
    from tensorflow.keras.applications import EfficientNetB6
    from tensorflow.keras.applications import EfficientNetB7

    from Engine.Models.Process.EfficientNet_Process import ProcessEfficientNet

except ImportError as error:
    print(error)
    sys.exit(-1)


class EfficientNetGradientMaps:

    def __init__(self):
        pass

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Build an auxiliary model for GradCAM/GradCAM++ computation.

        Para EfficientNet, usa a última camada convolucional por padrão,
        que mantém estrutura espacial 2D antes do pooling global.

        Args:
            target_layer_name: Nome da camada alvo. Se None, tenta encontrar
                              automaticamente a última camada convolucional.
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating GradCAM model")

        if target_layer_name is None:
            # Encontrar a última camada convolucional do EfficientNet
            # Geralmente são camadas com nome começando com 'block' ou 'top_conv'
            for layer in reversed(self.neural_network_model.layers):
                if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                    if len(layer.output_shape) == 4:  # Garantir que tem dimensão espacial
                        target_layer_name = layer.name
                        break

            if target_layer_name is None:
                # Fallback: usar uma camada específica conhecida
                target_layer_name = 'top_conv'

        target_layer = self.neural_network_model.get_layer(target_layer_name)

        self.gradcam_model = Model(inputs=self.neural_network_model.inputs,
                                   outputs=[target_layer.output, self.neural_network_model.output])

    def compute_gradcam_plusplus(self, input_sample: numpy.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> numpy.ndarray:
        """
        Compute Grad-CAM++ heatmap for EfficientNet (VERSÃO CORRIGIDA).

        CORREÇÕES:
        - Axis corrigido no reduce_sum para arquiteturas CNN 2D
        - Tratamento adequado para saída 4D (batch, height, width, channels)
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct input shape (batch, height, width, channels)
        if len(input_sample.shape) == 3:
            input_sample = numpy.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 4 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape1:
            with tensorflow.GradientTape() as tape2:
                with tensorflow.GradientTape() as tape3:
                    layer_output, predictions = self.gradcam_model(input_tensor)

                    if class_idx is None:
                        class_idx = tensorflow.argmax(predictions[0]).numpy()

                    class_score = predictions[:, class_idx]

                # First-order gradients
                grads = tape3.gradient(class_score, layer_output)

            # Second-order gradients
            grads_2 = tape2.gradient(grads, layer_output)

        # Third-order gradients
        grads_3 = tape1.gradient(grads_2, layer_output)

        # ✅ CORREÇÃO: Para EfficientNet com saída 4D (batch, height, width, channels)
        # usar axis=(1, 2) para dimensões espaciais
        if len(layer_output.shape) == 4:
            # Dimensões espaciais (height, width)
            spatial_axes = (1, 2)
        else:
            # Fallback para outras dimensionalidades
            spatial_axes = tuple(range(1, len(layer_output.shape) - 1))

        # Compute alpha weights (Grad-CAM++ formula) - CORRIGIDO
        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(layer_output * grads_3,
                                                            axis=spatial_axes, keepdims=True) + 1e-10

        alpha = numerator / denominator

        # ReLU on gradients
        relu_grads = tensorflow.maximum(grads, 0.0)

        # Weighted combination - calcular média ponderada ao longo das dimensões espaciais
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=spatial_axes)

        # Compute weighted activation map
        layer_output_squeezed = layer_output[0]  # Remove batch dimension

        # Para 4D: (height, width, channels) @ (channels,) -> (height, width)
        if len(layer_output_squeezed.shape) == 3:
            heatmap = tensorflow.reduce_sum(
                layer_output_squeezed * weights[tensorflow.newaxis, tensorflow.newaxis, :],
                axis=-1
            )
        else:
            heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
            heatmap = tensorflow.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tensorflow.maximum(heatmap, 0)

        # Normalização robusta
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_gradcam(self, input_sample: numpy.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> numpy.ndarray:
        """Standard Grad-CAM computation for EfficientNet."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 3:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        elif len(input_sample.shape) == 4 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        # Para EfficientNet com saída 4D (batch, height, width, channels)
        if len(layer_output.shape) == 4:
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

        elif len(layer_output.shape) == 3:  # Fallback
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1))

        else:
            pooled_grads = tensorflow.reduce_mean(grads, axis=0)

        layer_output = layer_output[0]

        # Compute weighted combination
        if len(layer_output.shape) == 3:  # (height, width, channels)
            heatmap = tensorflow.reduce_sum(
                layer_output * pooled_grads[tensorflow.newaxis, tensorflow.newaxis, :],
                axis=-1
            )
        else:
            heatmap = layer_output @ pooled_grads[..., tensorflow.newaxis]
            heatmap = tensorflow.squeeze(heatmap)

        # Normalização robusta
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)

        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_scorecam(self, input_sample: numpy.ndarray, class_idx: int = None,
                         target_layer_name: str = None, batch_size: int = 32) -> numpy.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method) for EfficientNet.
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

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

        # Para EfficientNet: (height, width, channels)
        if len(activations.shape) == 3:
            num_channels = activations.shape[-1]
        else:
            num_channels = activations.shape[-1]

        weights = []

        for i in range(num_channels):
            # Extract channel activation map
            if len(activations.shape) == 3:  # (height, width, channels)
                act_map = activations[:, :, i]
            else:
                act_map = activations[:, i] if len(activations.shape) == 2 else activations[i]

            # Normalize to [0, 1]
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            if len(act_map.shape) == 2:  # 2D activation map
                zoom_factors = (input_sample.shape[1] / act_map.shape[0],
                                input_sample.shape[2] / act_map.shape[1])
                upsampled = zoom(act_map, zoom_factors, order=1)

                # Replicate across channels
                upsampled = numpy.repeat(upsampled[:, :, numpy.newaxis],
                                         input_sample.shape[3], axis=2)
            else:  # 1D activation map (fallback)
                upsampled = zoom(act_map, (input_sample.shape[1] / act_map.shape[0],), order=1)
                upsampled = numpy.tile(upsampled[:, numpy.newaxis, numpy.newaxis],
                                       (1, input_sample.shape[2], input_sample.shape[3]))

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
        if len(activations.shape) == 3:  # (height, width, channels)
            heatmap = numpy.tensordot(activations, weights, axes=([2], [0]))
        elif len(activations.shape) == 2:  # (sequence, features) - fallback
            heatmap = numpy.dot(activations, weights)
        else:
            heatmap = numpy.tensordot(activations, weights, axes=([-1], [0]))

        heatmap = numpy.squeeze(heatmap)
        heatmap = numpy.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap