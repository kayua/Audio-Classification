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

    def compute_gradcam_plusplus(self, input_sample: numpy.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> numpy.ndarray:
        """
        Compute Grad-CAM++ heatmap with proper shape handling.
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

        # ✅ Determine spatial axes based on layer output shape
        layer_shape = layer_output.shape
        print(f"🔍 Layer output shape: {layer_shape}")

        if len(layer_shape) == 4:  # (batch, height, width, channels)
            spatial_axes = (1, 2)
        elif len(layer_shape) == 3:  # (batch, time, features) or (batch, height, width)
            spatial_axes = (1,)
        else:
            spatial_axes = tuple(range(1, len(layer_shape) - 1))

        # Compute alpha weights (Grad-CAM++ formula)
        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
            layer_output * grads_3, axis=spatial_axes, keepdims=True
        ) + 1e-10

        alpha = numerator / denominator

        # ReLU on gradients
        relu_grads = tensorflow.maximum(grads, 0.0)

        # Weighted combination
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=spatial_axes)

        # Remove batch dimension from layer output
        layer_output_squeezed = layer_output[0]

        print(f"🔍 Layer output (no batch) shape: {layer_output_squeezed.shape}")
        print(f"🔍 Weights shape: {weights.shape}")

        # Compute heatmap based on dimensionality
        if len(layer_output_squeezed.shape) == 3:  # (height, width, channels)
            # Standard 2D convolutional case
            heatmap = tensorflow.reduce_sum(
                layer_output_squeezed * weights[tensorflow.newaxis, tensorflow.newaxis, :],
                axis=-1
            )
        elif len(layer_output_squeezed.shape) == 2:  # (sequence, features) or (height, features)
            # 1D or flattened case
            heatmap = tensorflow.reduce_sum(
                layer_output_squeezed * weights[tensorflow.newaxis, :],
                axis=-1
            )
        else:
            # Fallback for other cases
            heatmap = tensorflow.tensordot(layer_output_squeezed, weights, axes=([-1], [0]))
            heatmap = tensorflow.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tensorflow.maximum(heatmap, 0)

        # Convert to numpy and ensure 2D
        heatmap_np = heatmap.numpy()

        # ✅ CRITICAL: Squeeze all extra dimensions
        while len(heatmap_np.shape) > 2:
            heatmap_np = numpy.squeeze(heatmap_np, axis=0)

        print(f"🔍 Final heatmap shape: {heatmap_np.shape}")

        # Normalização robusta
        heatmap_max = numpy.max(heatmap_np)
        if heatmap_max > 1e-10:
            heatmap_np = heatmap_np / heatmap_max

        return heatmap_np

    def compute_gradcam(self, input_sample: numpy.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> numpy.ndarray:
        """
        Standard Grad-CAM computation with proper shape handling.
        """
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

        # Determine pooling axes
        if len(layer_output.shape) == 4:  # (batch, height, width, channels)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
        elif len(layer_output.shape) == 3:  # (batch, sequence, features)
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
        elif len(layer_output.shape) == 2:  # (sequence, features)
            heatmap = tensorflow.reduce_sum(
                layer_output * pooled_grads[tensorflow.newaxis, :],
                axis=-1
            )
        else:
            heatmap = tensorflow.tensordot(layer_output, pooled_grads, axes=([-1], [0]))
            heatmap = tensorflow.squeeze(heatmap)

        # Normalização robusta
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_np = heatmap.numpy()

        # ✅ Ensure 2D output
        while len(heatmap_np.shape) > 2:
            heatmap_np = numpy.squeeze(heatmap_np, axis=0)

        heatmap_max = numpy.max(heatmap_np)
        if heatmap_max > 1e-10:
            heatmap_np = heatmap_np / heatmap_max

        return heatmap_np

    @staticmethod
    def interpolate_heatmap(heatmap: numpy.ndarray, target_shape: tuple,
                            smooth: bool = True) -> numpy.ndarray:
        """
        Interpola heatmap para o tamanho do espectrograma com tratamento robusto de shapes.
        """
        if not isinstance(heatmap, numpy.ndarray):
            heatmap = numpy.array(heatmap)

        print(f"🔍 Interpolating heatmap from {heatmap.shape} to {target_shape}")

        # ✅ Remove extra dimensions if present
        while len(heatmap.shape) > 2:
            heatmap = numpy.squeeze(heatmap, axis=0)

        # ✅ Handle different heatmap shapes
        if len(heatmap.shape) == 2:
            # Standard 2D heatmap
            zoom_factors = (
                target_shape[0] / heatmap.shape[0],
                target_shape[1] / heatmap.shape[1]
            )
            interpolated = zoom(heatmap, zoom_factors, order=3)

        elif len(heatmap.shape) == 1:
            # 1D heatmap - expand to 2D
            # Assume it's temporal (time axis)
            temporal_interp = zoom(heatmap, (target_shape[1] / heatmap.shape[0],), order=3)

            # Create frequency profile (stronger at lower frequencies)
            freq_profile = numpy.linspace(1.0, 0.6, target_shape[0])

            # Expand to 2D
            interpolated = freq_profile[:, numpy.newaxis] * temporal_interp[numpy.newaxis, :]

        else:
            raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}. Expected 1D or 2D.")

        # Ensure exact target shape
        if interpolated.shape != target_shape:
            print(f"⚠️  Adjusting shape from {interpolated.shape} to {target_shape}")
            zoom_factors_adjust = (
                target_shape[0] / interpolated.shape[0],
                target_shape[1] / interpolated.shape[1]
            )
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        # Apply smoothing if requested
        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=2.0)

        print(f"✅ Final interpolated shape: {interpolated.shape}")

        return interpolated

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Constrói o modelo Grad-CAM para extrair ativações intermediárias.
        Corrigido para funcionar com TensorFlow/Keras moderno.
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built before creating GradCAM model")

        if target_layer_name is None:
            # Método 1: Procurar por camadas convolucionais 4D
            from tensorflow.keras.layers import Conv2D, DepthwiseConv2D

            for layer in reversed(self.neural_network_model.layers):
                # Verificar se é uma camada convolucional
                if isinstance(layer, (Conv2D, DepthwiseConv2D)):
                    try:
                        # Tentar acessar o output shape através do layer.output
                        if hasattr(layer, 'output'):
                            output_shape = layer.output.shape

                            # Verificar se é 4D (batch, height, width, channels)
                            if len(output_shape) == 4:
                                target_layer_name = layer.name
                                print(f"🎯 Target layer automatically selected: {target_layer_name}")
                                print(f"   Output shape: {output_shape}")
                                break

                    except (AttributeError, ValueError, TypeError):
                        continue

            # Método 2: Procurar por nome de camada conhecida
            if target_layer_name is None:
                possible_layers = [
                    'head_conv',  # EfficientNet from scratch
                    'top_conv',  # Alternativa comum
                    'block7a_project_conv',  # Último bloco
                    'block6d_project_conv',  # Penúltimo bloco
                    'conv_head'  # Outro padrão comum
                ]

                for possible_name in possible_layers:
                    try:
                        layer = self.neural_network_model.get_layer(possible_name)
                        target_layer_name = possible_name
                        print(f"🎯 Using known layer name: {target_layer_name}")
                        break
                    except ValueError:
                        continue

            # Método 3: Procurar por padrão de nome
            if target_layer_name is None:
                for layer in reversed(self.neural_network_model.layers):
                    if any(pattern in layer.name.lower() for pattern in ['conv', 'project', 'expand']):
                        try:
                            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                                target_layer_name = layer.name
                                print(f"🎯 Using pattern-matched layer: {target_layer_name}")
                                break
                        except:
                            continue

            # Se ainda não encontrou, erro informativo
            if target_layer_name is None:
                layer_info = []
                for layer in self.neural_network_model.layers:
                    try:
                        if hasattr(layer, 'output'):
                            layer_info.append(f"{layer.name}: {layer.output.shape}")
                    except:
                        layer_info.append(f"{layer.name}: <shape unavailable>")

                raise ValueError(
                    f"Could not find suitable convolutional layer for Grad-CAM.\n"
                    f"Available layers:\n" + "\n".join(layer_info[:10]) + "\n..."
                )

        # Obter a camada alvo
        try:
            target_layer = self.neural_network_model.get_layer(target_layer_name)
            print(f"✅ Successfully found target layer: {target_layer_name}")
        except ValueError:
            available = [l.name for l in self.neural_network_model.layers if 'conv' in l.name.lower()]
            raise ValueError(
                f"Layer '{target_layer_name}' not found in model.\n"
                f"Available conv layers: {available[:5]}..."
            )

        # Criar modelo Grad-CAM
        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

        print(f"✅ Grad-CAM model built successfully!")
        print(f"   Input shape: {self.gradcam_model.input_shape}")
        print(f"   Target layer output shape: {target_layer.output.shape}")
        print(f"   Model output shape: {self.neural_network_model.output.shape}")


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