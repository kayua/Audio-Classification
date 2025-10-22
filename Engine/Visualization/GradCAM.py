import numpy
import tensorflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:

    def __init__(self):
        pass

    def compute_gradcam(self, input_sample: numpy.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> numpy.ndarray:
        """Standard Grad-CAM computation."""
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        if len(layer_output.shape) == 3:  # (batch, sequence, features)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1))

        elif len(layer_output.shape) == 4:  # (batch, height, width, channels)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

        else:
            pooled_grads = tensorflow.reduce_mean(grads, axis=0)

        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # Normalização robusta
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)

        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()
