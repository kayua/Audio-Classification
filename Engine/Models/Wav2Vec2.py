#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import sys
    import logging
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow
    from tensorflow.keras import Model
    from Engine.Layers.GELU import GELU

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape

    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Activation
    from Engine.Layers.MaskLayer import MaskCreator

    from tensorflow.keras.layers import TimeDistributed

    from Engine.Layers.MaskTimeLayer import TimeMasking

    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from Engine.Loss.ContrastiveLoss import ContrastiveLoss

    from tensorflow.keras.layers import GlobalAveragePooling1D
    from Engine.Layers.QuantizerLayerMLP import QuantizationLayer

    from Engine.Models.Process.Wav2Vec2_Process import Wav2Vec2Process
    from Engine.Modules.GumbelVectorQuantizer import GumbelVectorQuantizer

except ImportError as error:
    print(error)
    sys.exit(-1)


class AudioWav2Vec2(MaskCreator, Wav2Vec2Process):
    """
    Enhanced AudioWav2Vec2 with COMPLETE Explainable AI (XAI) capabilities

    FUNCIONALIDADES XAI IMPLEMENTADAS:
    ==================================
    1. ✅ Grad-CAM: Visualização padrão de mapas de ativação
    2. ✅ Grad-CAM++: Versão melhorada com ponderação mais precisa
    3. ✅ Score-CAM: Método sem gradientes (gradient-free)
    4. ✅ Visualizações modernas e interativas
    5. ✅ Geração automática para validação
    6. ✅ Análise comparativa de múltiplos métodos XAI
    7. ✅ Suporte para arquitetura Wav2Vec2 com TimeDistributed

    Adaptações específicas para Wav2Vec2:
    - Suporte para camadas TimeDistributed
    - Visualização após fine-tuning (modelo com única saída)
    - Tratamento especial para arquitetura com duas saídas
    """

    def __init__(self, arguments):
        """
        Initialize the AudioWav2Vec2 model with XAI capabilities.

        Args:
            @number_classes (int): The number of output classes for classification tasks.
            @last_layer_activation (str): The activation function for the output layer (e.g., 'softmax').
            @loss_function (str): The loss function used for training the model (e.g., 'categorical_crossentropy').
            @optimizer_function (str): The optimizer used for training the model (e.g., 'adam').
            @quantization_units (int): The number of quantization units.
            @key_dimension (int): The key dimension for the transformer attention.
            @dropout_rate (float): The dropout rate for regularization.
            @intermediary_layer_activation (str): The activation function for intermediary layers (e.g., 'relu').
            @input_dimension (tuple): The input dimension for the model (e.g., (128, 80) for Mel spectrograms).
            @number_heads (int): The number of attention heads in the transformer block.
            @kernel_size (int): The kernel size for the convolutional layers.
            @list_filters_encoder (list[int], optional): A list of the number of filters for each convolutional encoder block.
        """
        Wav2Vec2Process.__init__(self, arguments)

        self.neural_network_model = None
        self.gradcam_model = None
        self.list_filters_encoder = arguments.wav_to_vec_list_filters_encoder
        self.loss_function = arguments.wav_to_vec_loss_function
        self.optimizer_function = arguments.wav_to_vec_optimizer_function
        self.kernel_size = arguments.wav_to_vec_kernel_size
        self.quantization_units = arguments.wav_to_vec_quantization_bits
        self.key_dimension = arguments.wav_to_vec_key_dimension
        self.intermediary_layer_activation = arguments.wav_to_vec_intermediary_layer_activation
        self.number_heads = arguments.wav_to_vec_number_heads
        self.input_dimension = arguments.wav_to_vec_input_dimension
        self.number_classes = arguments.number_classes
        self.dropout_rate = arguments.wav_to_vec_dropout_rate
        self.last_layer_activation = arguments.wav_to_vec_last_layer_activation
        self.model_name = "Wav2Vec2"

        # Set modern style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """
        Build the Wav2Vec2 model architecture with proper layer naming for XAI.

        The model consists of the following components with named layers:
            - Convolutional encoder layers with the specified filters.
            - Transformer block with multi-head attention for capturing long-range dependencies.
            - Quantization layer for feature compression.
            - Dense layers for classification.
        """
        # Input layer
        inputs = Input(shape=self.input_dimension, name='input_layer')
        neural_network_flow = Reshape((128, 80, 1), name='reshape_input')(inputs)

        # Apply convolutional layers with proper naming
        for idx, number_filters in enumerate(self.list_filters_encoder):
            neural_network_flow = TimeDistributed(
                Conv1D(number_filters, self.kernel_size, strides=(2,), use_bias=True,
                       kernel_regularizer=None, bias_regularizer=None,
                       kernel_constraint=None),
                name=f'time_distributed_conv_{idx}'
            )(neural_network_flow)

            neural_network_flow = TimeDistributed(GELU(), name=f'time_distributed_gelu_{idx}')(neural_network_flow)
            neural_network_flow = TimeDistributed(LayerNormalization(),
                                                  name=f'time_distributed_layernorm_{idx}')(neural_network_flow)

        # Flatten the convolutional output
        flatten_flow = TimeDistributed(Flatten(), name='time_distributed_flatten')(neural_network_flow)

        # Dense layer
        dense_layer = TimeDistributed(
            Dense(self.number_classes, activation=self.intermediary_layer_activation),
            name='time_distributed_dense'
        )(flatten_flow)

        # Time masking
        time_masking = TimeMasking(mask_time_prob=0.2, number_mask_time_steps=5,
                                   name='time_masking')(dense_layer)

        # Transformer block with multi-head attention
        transformer_attention = MultiHeadAttention(
            num_heads=self.number_heads,
            key_dim=4,
            name='multi_head_attention'
        )(time_masking, time_masking)

        # Residual connection and normalization
        transformer_attention = Add(name='add_residual_1')([time_masking, transformer_attention])
        transformer_attention = LayerNormalization(name='layer_norm_1')(transformer_attention)

        # Feed-forward network
        feedforward_network = Dense(self.number_classes, name='ffn_dense_1')(transformer_attention)
        feedforward_network = GELU(name='ffn_gelu_1')(feedforward_network)

        feedforward_network = Dense(self.number_classes, name='ffn_dense_2')(feedforward_network)
        feedforward_network = GELU(name='ffn_gelu_2')(feedforward_network)

        # Add and normalize
        transformer_output = Add(name='add_residual_2')([transformer_attention, feedforward_network])
        transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)

        print(transformer_output)

        # Gumbel Vector Quantization
        quantized_output, perplexity = GumbelVectorQuantizer()(dense_layer, 128)

        # Create the Keras model with two outputs
        self.neural_network_model = Model(
            inputs=inputs,
            outputs=[transformer_output, quantized_output],
            name=self.model_name
        )

        # Compile with ContrastiveLoss
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=ContrastiveLoss(margin=0.5),
            metrics=['accuracy']
        )

        self.neural_network_model.summary()

        exit()

    def compile_and_train(self, train_data: tensorflow.Tensor, train_labels: tensorflow.Tensor,
                          epochs: int, batch_size: int, validation_data: tuple = None,
                          generate_gradcam: bool = True, num_gradcam_samples: int = 30,
                          gradcam_output_dir: str = './mapas_de_ativacao',
                          xai_method: str = 'gradcam++') -> tensorflow.keras.callbacks.History:
        """
        Compiles and trains the model with XAI visualization support.

        This method performs two-stage training:
        1. Pre-training with ContrastiveLoss
        2. Fine-tuning with supervised learning + XAI visualization

        Args:
            train_data: The input training data
            train_labels: The corresponding labels for the training data
            epochs: Number of training epochs
            batch_size: Size of the batches for each training step
            validation_data: Optional validation data tuple (X_val, y_val)
            generate_gradcam: Whether to generate activation maps after training
            num_gradcam_samples: Number of samples to visualize
            gradcam_output_dir: Output directory for visualizations
            xai_method: XAI method to use ('gradcam', 'gradcam++', or 'scorecam')

        Returns:
            Training history object
        """
        logging.info("Starting the initial compilation and training phase.")

        # Step 1: Pre-training with ContrastiveLoss
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=ContrastiveLoss(margin=0.75)
        )
        logging.info("Model compiled with ContrastiveLoss.")

        logging.info(f"Pre-training for {epochs} epochs with batch size {batch_size}.")
        self.neural_network_model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        logging.info("Pre-training completed. Setting the model as non-trainable.")

        # Step 2: Fine-tuning setup
        self.neural_network_model.trainable = False
        neural_network_flow = Flatten(name='flatten_final')(self.neural_network_model.output[0])

        # Add classification head
        neural_network_flow = Dense(self.number_classes, name='dense_classifier')(neural_network_flow)
        neural_network_flow = GELU(name='gelu_classifier')(neural_network_flow)
        logging.info(f"Added Dense layer with {self.number_classes} classes.")

        # Recreate model with single output for classification
        self.neural_network_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=neural_network_flow,
            name=f"{self.model_name}_finetuned"
        )

        # Step 3: Compile for supervised learning
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )
        logging.info("Recompiled the model for supervised learning.")

        # Step 4: Fine-tuning
        logging.info(f"Fine-tuning for {epochs} epochs with batch size {batch_size}.")
        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
        logging.info("Fine-tuning completed successfully.")

        if validation_data is not None:
            print(f"Acurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")

        # Step 5: Generate XAI visualizations
        if generate_gradcam and validation_data is not None:
            print("\n" + "=" * 80)
            print(f"GERANDO MAPAS DE ATIVAÇÃO ({xai_method.upper()}) - WAV2VEC2 MODEL")
            print("=" * 80)

            val_data, val_labels = validation_data

            stats = self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=num_gradcam_samples,
                output_dir=gradcam_output_dir,
                xai_method=xai_method
            )

            print(f"\n✓ Mapas de ativação salvos em: {gradcam_output_dir}")
            print("=" * 80 + "\n")

        return training_history

    def build_gradcam_model(self, target_layer_name: str = None) -> None:
        """
        Build an auxiliary model for GradCAM/GradCAM++ computation.

        For Wav2Vec2, we target layers before the flatten operation, specifically
        the TimeDistributed convolutional layers which preserve spatial structure.

        Args:
            target_layer_name: Name of target layer. If None, uses last TimeDistributed conv layer
        """
        if self.neural_network_model is None:
            raise ValueError("Model must be built and fine-tuned before creating GradCAM model")

        # Find appropriate target layer if not specified
        if target_layer_name is None:
            # Use the last TimeDistributed conv layer
            last_conv_idx = len(self.list_filters_encoder) - 1
            target_layer_name = f'time_distributed_conv_{last_conv_idx}'
            print(f"⚠️  AVISO: Usando '{target_layer_name}' como camada alvo")
            print("   Esta é a última camada convolucional TimeDistributed")
        else:
            print(f"⚠️  AVISO: Usando camada customizada '{target_layer_name}'")

        try:
            target_layer = self.neural_network_model.get_layer(target_layer_name)
        except ValueError:
            # Fallback: try to find any conv layer
            print(f"⚠️  Camada '{target_layer_name}' não encontrada.")
            print("   Tentando encontrar camada alternativa...")

            # Try to find the flatten layer and use the layer before it
            for i, layer in enumerate(self.neural_network_model.layers):
                if 'flatten' in layer.name.lower():
                    if i > 0:
                        target_layer = self.neural_network_model.layers[i - 1]
                        target_layer_name = target_layer.name
                        print(f"   Usando camada antes do flatten: '{target_layer_name}'")
                        break
            else:
                raise ValueError("Não foi possível encontrar uma camada adequada para GradCAM")

        self.gradcam_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=[target_layer.output, self.neural_network_model.output]
        )

        print(f"✓ GradCAM model criado com camada: {target_layer_name}")
        print(f"  Forma de saída da camada: {target_layer.output.shape}")

    def compute_gradcam_plusplus(self, input_sample: np.ndarray, class_idx: int = None,
                                 target_layer_name: str = None) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap for Wav2Vec2 architecture.

        Args:
            input_sample: Input spectrogram
            class_idx: Target class index (if None, uses predicted class)
            target_layer_name: Target layer name (if None, uses default)

        Returns:
            Normalized heatmap as numpy array
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct input shape
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(np.float32)
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

        # Determine reduce axis based on output shape
        if len(layer_output.shape) == 4:  # (batch, time, freq, channels)
            reduce_axis = 3
        elif len(layer_output.shape) == 3:  # (batch, sequence, features)
            reduce_axis = 2
        else:
            reduce_axis = -1

        # Compute alpha weights (Grad-CAM++ formula)
        numerator = grads_2
        denominator = 2.0 * grads_2 + tensorflow.reduce_sum(
            layer_output * grads_3, axis=reduce_axis, keepdims=True
        ) + 1e-10

        alpha = numerator / denominator

        # ReLU on gradients
        relu_grads = tensorflow.maximum(grads, 0.0)

        # Weighted combination - adapt to shape
        if len(layer_output.shape) == 4:
            weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1, 2))
        elif len(layer_output.shape) == 3:
            weights = tensorflow.reduce_sum(alpha * relu_grads, axis=(1,))
        else:
            weights = tensorflow.reduce_sum(alpha * relu_grads, axis=0)

        # Compute weighted activation map
        layer_output_squeezed = layer_output[0]
        heatmap = layer_output_squeezed @ weights[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max
        else:
            print("⚠️  Aviso: Heatmap com valores muito baixos")

        return heatmap.numpy()

    def compute_gradcam(self, input_sample: np.ndarray, class_idx: int = None,
                        target_layer_name: str = None) -> np.ndarray:
        """
        Standard Grad-CAM computation for Wav2Vec2.

        Args:
            input_sample: Input spectrogram
            class_idx: Target class index
            target_layer_name: Target layer name

        Returns:
            Normalized heatmap
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        with tensorflow.GradientTape() as tape:
            layer_output, predictions = self.gradcam_model(input_tensor)

            if class_idx is None:
                class_idx = tensorflow.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, layer_output)

        # Adapt pooling based on output shape
        if len(layer_output.shape) == 4:  # (batch, time, freq, channels)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
        elif len(layer_output.shape) == 3:  # (batch, sequence, features)
            pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1))
        else:
            pooled_grads = tensorflow.reduce_mean(grads, axis=0)

        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # Normalize
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap_max = tensorflow.math.reduce_max(heatmap)
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def compute_scorecam(self, input_sample: np.ndarray, class_idx: int = None,
                         target_layer_name: str = None, batch_size: int = 32) -> np.ndarray:
        """
        Compute Score-CAM heatmap (gradient-free method) for Wav2Vec2.

        Args:
            input_sample: Input spectrogram
            class_idx: Target class index
            target_layer_name: Target layer name
            batch_size: Batch size for processing

        Returns:
            Normalized heatmap
        """
        if self.gradcam_model is None or target_layer_name is not None:
            self.build_gradcam_model(target_layer_name)

        # Ensure correct shape
        if len(input_sample.shape) == 2:
            input_sample = np.expand_dims(input_sample, axis=0)
        elif len(input_sample.shape) == 3 and input_sample.shape[0] != 1:
            input_sample = input_sample[0:1]

        input_sample = input_sample.astype(np.float32)
        input_tensor = tensorflow.convert_to_tensor(input_sample)

        # Get activations
        layer_output, predictions = self.gradcam_model(input_tensor)

        if class_idx is None:
            class_idx = tensorflow.argmax(predictions[0]).numpy()

        # Get activation maps
        activations = layer_output[0].numpy()

        # Handle different shapes
        if len(activations.shape) == 3:  # (time, freq, channels)
            num_channels = activations.shape[-1]
        elif len(activations.shape) == 2:  # (sequence, features)
            num_channels = activations.shape[-1]
        else:
            num_channels = activations.shape[-1]

        weights = []
        for i in range(num_channels):
            # Extract activation map
            if len(activations.shape) == 3:
                act_map = activations[:, :, i]
            else:
                act_map = activations[:, i]

            # Normalize to [0, 1]
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

            # Upsample to input size
            if len(act_map.shape) == 1:
                upsampled = zoom(act_map, (input_sample.shape[1] / act_map.shape[0],), order=1)
                upsampled = np.tile(upsampled[:, np.newaxis], (1, input_sample.shape[2]))
            else:
                zoom_factors = (input_sample.shape[1] / act_map.shape[0],
                                input_sample.shape[2] / act_map.shape[1])
                upsampled = zoom(act_map, zoom_factors, order=1)

            # Mask input
            masked_input = input_sample[0] * upsampled
            masked_input = np.expand_dims(masked_input, 0)

            # Get score for masked input
            masked_pred = self.neural_network_model.predict(masked_input, verbose=0)
            score = masked_pred[0, class_idx]

            weights.append(score)

        weights = np.array(weights)
        weights = np.maximum(weights, 0)

        # Weighted combination
        if len(activations.shape) == 3:
            heatmap = np.tensordot(activations, weights, axes=([2], [0]))
        else:
            heatmap = np.dot(activations, weights)

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def smooth_heatmap(heatmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: np.ndarray, target_shape: tuple,
                            smooth: bool = True) -> np.ndarray:
        """
        Interpolate heatmap to target shape with optional smoothing.

        Args:
            heatmap: Input heatmap (1D or 2D array)
            target_shape: Target dimensions (height, width)
            smooth: Apply Gaussian smoothing after interpolation

        Returns:
            Interpolated heatmap
        """
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)

        # Case 1: 1D heatmap (sequence output)
        if len(heatmap.shape) == 1:
            print(f"  ℹ️  Heatmap 1D detectado (tamanho: {heatmap.shape[0]})")
            print(f"     Expandindo para 2D: {target_shape}")

            # Interpolate temporally
            temporal_interp = zoom(heatmap, (target_shape[0] / heatmap.shape[0],), order=3)

            # Expand to frequencies with smooth gradient
            freq_profile = np.linspace(1.0, 0.6, target_shape[1])
            heatmap_2d = temporal_interp[:, np.newaxis] * freq_profile[np.newaxis, :]

            interpolated = heatmap_2d

        # Case 2: 2D heatmap
        elif len(heatmap.shape) == 2:
            print(f"  ℹ️  Heatmap 2D detectado (forma: {heatmap.shape})")
            zoom_factors = (target_shape[0] / heatmap.shape[0],
                            target_shape[1] / heatmap.shape[1])
            interpolated = zoom(heatmap, zoom_factors, order=3)
        else:
            raise ValueError(f"Heatmap shape inesperado: {heatmap.shape}")

        # Fine adjustment
        if interpolated.shape != target_shape:
            zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                   target_shape[1] / interpolated.shape[1])
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        # Smoothing
        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=1.5)

        return interpolated

    def plot_gradcam_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'gradcam++',
                            save_path: str = None, show_plot: bool = True) -> None:
        """
        Modern, visually appealing GradCAM visualization for Wav2Vec2.

        Args:
            input_sample: Input spectrogram
            heatmap: Activation heatmap
            class_idx: Class index used for computation
            predicted_class: Predicted class
            true_label: True label (optional)
            confidence: Prediction confidence
            xai_method: XAI method name
            save_path: Path to save figure
            show_plot: Whether to display the plot
        """
        # Handle input dimensions
        if len(input_sample.shape) == 3:
            input_sample = input_sample[0]

        # Enhanced interpolation with smoothing
        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_sample.shape, smooth=True)

        # Create figure with modern style
        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        # Color schemes
        cmap_input = 'viridis'
        cmap_heatmap = 'jet'

        # 1. Original Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('📊 Espectrograma de Entrada',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Frames Temporais', fontsize=10)
        ax1.set_ylabel('Bins de Frequência', fontsize=10)
        ax1.grid(False)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'🔥 Mapa de Ativação ({xai_method.upper()})',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Frames Temporais', fontsize=10)
        ax2.set_ylabel('Bins de Frequência', fontsize=10)
        ax2.grid(False)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.6, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)

        ax3.set_title('🎯 Sobreposição', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Frames Temporais', fontsize=10)
        ax3.set_ylabel('Bins de Frequência', fontsize=10)
        ax3.grid(False)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=9)

        # 4. Temporal Importance Profile
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_importance = np.mean(interpolated_heatmap, axis=0)
        time_steps = np.arange(len(temporal_importance))

        temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

        ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Perfil Suavizado')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Perfil Original')

        ax4.set_xlabel('Frame Temporal', fontsize=10)
        ax4.set_ylabel('Importância Média', fontsize=10)
        ax4.set_title('📈 Perfil de Importância Temporal',
                      fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_xlim([0, len(temporal_importance)])
        ax4.set_ylim([0, 1])

        # Super title
        pred_status = '✅' if predicted_class == true_label else '❌'
        conf_str = f' | Confiança: {confidence:.1%}' if confidence is not None else ''

        if true_label is not None:
            suptitle = f'{pred_status} Predito: Classe {predicted_class} | Verdadeiro: Classe {true_label}{conf_str}'
        else:
            suptitle = f'Predito: Classe {predicted_class}{conf_str}'

        fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Figura salva: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_validation_visualizations(self, validation_data: np.ndarray,
                                           validation_labels: np.ndarray,
                                           num_samples: int = 10,
                                           output_dir: str = './gradcam_outputs',
                                           target_layer_name: str = None,
                                           xai_method: str = 'gradcam++') -> dict:
        """
        Generate enhanced XAI visualizations for validation samples.

        Args:
            validation_data: Validation input data
            validation_labels: Validation labels
            num_samples: Number of samples to visualize
            output_dir: Output directory for saving visualizations
            target_layer_name: Target layer for XAI (if None, uses default)
            xai_method: XAI method ('gradcam', 'gradcam++', or 'scorecam')

        Returns:
            Dictionary with statistics about generated visualizations
        """
        import os

        print(f"\n🔍 Dados de validação: {validation_data.shape}")
        print(f"🎯 Labels de validação: {validation_labels.shape}")
        print(f"📐 Dimensão de entrada esperada: {self.input_dimension}")
        print(f"🧠 Método XAI: {xai_method.upper()}")

        if target_layer_name is None:
            last_conv_idx = len(self.list_filters_encoder) - 1
            print(f"🎯 Camada alvo: 'time_distributed_conv_{last_conv_idx}' (última camada convolucional)")
        else:
            print(f"🎯 Camada alvo: '{target_layer_name}'")
        print()

        os.makedirs(output_dir, exist_ok=True)

        predictions = self.neural_network_model.predict(validation_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        if len(validation_labels.shape) > 1:
            true_labels = np.argmax(validation_labels, axis=1)
        else:
            true_labels = validation_labels

        correct_indices = np.where(predicted_classes == true_labels)[0]
        incorrect_indices = np.where(predicted_classes != true_labels)[0]

        num_correct = min(num_samples // 2, len(correct_indices))
        num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

        selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if len(
            correct_indices) > 0 else []
        selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if len(
            incorrect_indices) > 0 else []

        selected_indices = np.concatenate([selected_correct, selected_incorrect])

        stats = {
            'total_samples': len(selected_indices),
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

        print(f"🎨 Gerando visualizações para {len(selected_indices)} amostras...")
        print(f"📁 Diretório de saída: {output_dir}\n")

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
                    heatmap = self.compute_gradcam_plusplus(sample, class_idx=predicted,
                                                            target_layer_name=target_layer_name)
                elif xai_method.lower() == 'scorecam':
                    heatmap = self.compute_scorecam(sample, class_idx=predicted,
                                                    target_layer_name=target_layer_name)
                else:
                    heatmap = self.compute_gradcam(sample, class_idx=predicted,
                                                   target_layer_name=target_layer_name)

                is_correct = predicted == true_label
                if is_correct:
                    stats['correct_predictions'] += 1
                    prefix = 'correto'
                else:
                    stats['incorrect_predictions'] += 1
                    prefix = 'incorreto'

                save_path = os.path.join(output_dir,
                                         f'{prefix}_amostra_{i:03d}_real_{true_label}_pred_{predicted}_conf_{confidence:.2f}.png')

                self.plot_gradcam_modern(
                    sample, heatmap, predicted, predicted, true_label,
                    confidence=confidence, xai_method=xai_method,
                    save_path=save_path, show_plot=False
                )

                status = '✅' if is_correct else '❌'
                print(f"{status} Amostra {i + 1}/{len(selected_indices)}: "
                      f"Real={true_label}, Pred={predicted}, Conf={confidence:.1%}")

            except Exception as e:
                print(f"❌ Erro processando amostra {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'=' * 80}")
        print(f"✨ GERAÇÃO DE VISUALIZAÇÕES CONCLUÍDA!")
        print(f"{'=' * 80}")
        print(f"📊 Total de amostras: {stats['total_samples']}")
        print(f"✅ Predições corretas: {stats['correct_predictions']}")
        print(f"❌ Predições incorretas: {stats['incorrect_predictions']}")
        accuracy = stats['correct_predictions'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        print(f"🎯 Acurácia: {accuracy:.2f}%")
        print(f"💾 Arquivos salvos em: {output_dir}")
        print(f"{'=' * 80}\n")

        return stats

    def explain_prediction_comprehensive(self, input_sample: np.ndarray,
                                         class_names: list = None,
                                         save_path: str = None,
                                         show_plot: bool = True) -> dict:
        """
        Generate comprehensive explanation with multiple XAI methods comparison.

        Args:
            input_sample: Input spectrogram to explain
            class_names: List of class names (optional)
            save_path: Path to save comprehensive analysis figure
            show_plot: Whether to display the plot

        Returns:
            Dictionary with explanation data and heatmaps
        """
        # Prepare sample
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

        # Compute heatmaps
        print("🔄 Computando mapas de ativação com diferentes métodos...")

        heatmap_gradcam = self.compute_gradcam(input_sample_2d, class_idx=predicted_class)
        heatmap_gradcampp = self.compute_gradcam_plusplus(input_sample_2d, class_idx=predicted_class)

        print("✓ Mapas computados com sucesso!")

        # Interpolate
        heatmap_gc_interp = self.interpolate_heatmap(heatmap_gradcam, input_sample_2d.shape, smooth=True)
        heatmap_pp_interp = self.interpolate_heatmap(heatmap_gradcampp, input_sample_2d.shape, smooth=True)

        # Create figure
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heat = 'jet'

        # Row 1: Grad-CAM
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample_2d, cmap=cmap_input, aspect='auto')
        ax1.set_title('📊 Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(heatmap_gc_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax2.set_title('🔥 Grad-CAM', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(input_sample_2d, cmap='gray', aspect='auto')
        ax3.imshow(heatmap_gc_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')

        # Row 2: Grad-CAM++
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(input_sample_2d, cmap=cmap_input, aspect='auto')
        ax4.set_title('📊 Espectrograma Original', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(heatmap_pp_interp, cmap=cmap_heat, aspect='auto', vmin=0, vmax=1)
        ax5.set_title('🔥 Grad-CAM++ (Melhorado)', fontsize=12, fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(input_sample_2d, cmap='gray', aspect='auto')
        ax6.imshow(heatmap_pp_interp, cmap=cmap_heat, alpha=0.5, aspect='auto', vmin=0, vmax=1)
        ax6.set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')

        # Row 3: Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(len(predictions[0]))]

        colors = ['#2ecc71' if i == predicted_class else '#95a5a6' for i in range(len(predictions[0]))]
        bars = ax7.barh(class_names, predictions[0], color=colors, edgecolor='black', linewidth=1.5)
        ax7.set_xlabel('Probabilidade', fontsize=11, fontweight='bold')
        ax7.set_title(f'🎯 Predição: {class_names[predicted_class]} (Confiança: {confidence:.1%})',
                      fontsize=13, fontweight='bold')
        ax7.set_xlim([0, 1])
        ax7.grid(axis='x', alpha=0.3)

        for bar, prob in zip(bars, predictions[0]):
            ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.3f}', ha='left', va='center', fontsize=10)

        # Temporal comparison
        ax8 = fig.add_subplot(gs[2, 2])
        temporal_gc = np.mean(heatmap_gc_interp, axis=0)
        temporal_gcpp = np.mean(heatmap_pp_interp, axis=0)

        temporal_gc_smooth = gaussian_filter(temporal_gc, sigma=2)
        temporal_gcpp_smooth = gaussian_filter(temporal_gcpp, sigma=2)

        time_steps = np.arange(len(temporal_gc))
        ax8.plot(time_steps, temporal_gc_smooth, linewidth=2, label='Grad-CAM', color='#3498db')
        ax8.plot(time_steps, temporal_gcpp_smooth, linewidth=2, label='Grad-CAM++', color='#e74c3c')
        ax8.fill_between(time_steps, temporal_gc_smooth, alpha=0.2, color='#3498db')
        ax8.fill_between(time_steps, temporal_gcpp_smooth, alpha=0.2, color='#e74c3c')

        ax8.set_xlabel('Frame Temporal', fontsize=10)
        ax8.set_ylabel('Importância', fontsize=10)
        ax8.set_title('📈 Comparação Temporal', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

        fig.suptitle('🧠 Análise Explicativa Abrangente - Wav2Vec2 XAI',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Análise completa salva: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        explanation = {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'class_probabilities': predictions[0].tolist(),
            'heatmap_gradcam': heatmap_gradcam,
            'heatmap_gradcampp': heatmap_gradcampp,
            'heatmap_gradcam_interpolated': heatmap_gc_interp,
            'heatmap_gradcampp_interpolated': heatmap_pp_interp,
            'temporal_importance_gradcam': temporal_gc,
            'temporal_importance_gradcampp': temporal_gcpp
        }

        if class_names:
            explanation['predicted_class_name'] = class_names[predicted_class]

        print("✅ Explicação abrangente gerada com sucesso!")
        return explanation

    # Properties
    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_filters_encoder(self):
        return self._list_filters_encoder

    @list_filters_encoder.setter
    def list_filters_encoder(self, value):
        self._list_filters_encoder = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, value):
        self._optimizer_function = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

    @property
    def quantization_units(self):
        return self._quantization_units

    @quantization_units.setter
    def quantization_units(self, value):
        self._quantization_units = value

    @property
    def key_dimension(self):
        return self._key_dimension

    @key_dimension.setter
    def key_dimension(self, value):
        self._key_dimension = value

    @property
    def intermediary_layer_activation(self):
        return self._intermediary_layer_activation

    @intermediary_layer_activation.setter
    def intermediary_layer_activation(self, value):
        self._intermediary_layer_activation = value

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, value):
        self._number_heads = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value):
        self._number_classes = value

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value

    @property
    def last_layer_activation(self):
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value):
        self._last_layer_activation = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value