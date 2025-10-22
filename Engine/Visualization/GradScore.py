import numpy
import tensorflow
import logging
from scipy.ndimage import zoom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradScore:
    def __init__(self):
        pass
    def compute_scorecam(self, input_sample: numpy.ndarray, class_idx: int = None,
                         target_layer_name: str = None, batch_size: int = 32) -> numpy.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method).
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        if len(input_sample.shape) == 2:
            input_sample = numpy.expand_dims(input_sample, axis=0)

        input_sample = input_sample.astype(numpy.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        # Get activations
        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        # Get activation maps
        activations = layer_output[0].numpy()

        if len(activations.shape) == 2:  # (sequence, features)
            num_channels = activations.shape[-1]

        else:
            num_channels = activations.shape[-1]

        weights = []

        for i in range(num_channels):
            if len(activations.shape) == 2:
                act_map = activations[:, i]
            else:
                act_map = activations[:, :, i] if len(activations.shape) == 3 else activations[:, i]

            # Normalize to [0, 1]
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            if len(act_map.shape) == 1:
                upsampled = zoom(act_map, (input_sample.shape[1] / act_map.shape[0],), order=1)
                upsampled = numpy.tile(upsampled[:, numpy.newaxis], (1, input_sample.shape[2]))
            else:
                zoom_factors = (input_sample.shape[1] / act_map.shape[0],
                                input_sample.shape[2] / act_map.shape[1])
                upsampled = zoom(act_map, zoom_factors, order=1)

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
        if len(activations.shape) == 2:
            heatmap = numpy.dot(activations, weights)
        else:
            heatmap = numpy.tensordot(activations, weights, axes=([[-1], [0]]))
            heatmap = numpy.squeeze(heatmap)

        heatmap = numpy.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap
