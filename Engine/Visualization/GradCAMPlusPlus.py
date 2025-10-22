import numpy
import tensorflow
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAMPlusPlus:
    def __init__(self):
        pass

    def compute_gradcam_plusplus(self, input_sample: numpy.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> numpy.ndarray:

        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
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

                grads = tape3.gradient(class_score, layer_output)

            grads_2 = tape2.gradient(grads, layer_output)

        grads_3 = tape1.gradient(grads_2, layer_output)

        if len(layer_output.shape) == 3:
            reduce_axis = 2

        elif len(layer_output.shape) == 4:
            reduce_axis = 3

        else:
            reduce_axis = -1

        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(layer_output * grads_3,
                                                            axis=reduce_axis, keepdims=True) + 1e-10

        alpha = numerator / denominator
        relu_grads = tensorflow.maximum(grads, 0.0)
        weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1,))
        layer_output_squeezed = layer_output[0]
        heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)

        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()