#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['Kayuã Oleques Paim']

# MIT License
#
# Copyright (c) 2025 Kayuã Oleques Paim
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
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    from pathlib import Path
    import tensorflow

    from tensorflow.keras import models
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention


    from Engine.Layers.PositionalEmbeddingsLayer import PositionalEmbeddingsLayer
    from Engine.Layers.DistillationCLSTokenLayer import DistillationCLSTokenLayer

    from Engine.Models.Process.AST_Process import ProcessAST

except ImportError as error:
    print(error)
    import sys

    sys.exit(-1)



class AudioSpectrogramTransformer(ProcessAST, VisualizationAST):

    def __init__(self, arguments):
        """
        Initialize the Audio Spectrogram Transformer.

        Args:
            arguments (object): Configuration object containing model parameters with attributes:
                - ast_head_size: Size of attention heads
                - ast_number_heads: Number of attention heads
                - ast_number_blocks: Number of transformer blocks
                - number_classes: Number of output classes
                - ast_patch_size: Tuple of (height, width) for patches
                - ast_dropout: Dropout rate
                - ast_optimizer_function: Optimizer name
                - ast_loss_function: Loss function name
                - ast_normalization_epsilon: Epsilon for layer normalization
                - ast_intermediary_activation: Activation for intermediate layers
                - ast_projection_dimension: Dimension for token projection
                - ast_number_filters_spectrogram: Number of spectrogram filters
        """
        super().__init__(arguments)
        self.neural_network_model = None
        self.attention_model = None
        self.attention_layers = []
        self.head_size = arguments.ast_head_size
        self.number_heads = arguments.ast_head_size
        self.number_blocks = arguments.ast_number_blocks
        self.number_classes = arguments.number_classes
        self.patch_size = arguments.ast_patch_size
        self.dropout = arguments.ast_dropout
        self.optimizer_function = arguments.ast_optimizer_function
        self.loss_function = arguments.ast_loss_function
        self.normalization_epsilon = arguments.ast_normalization_epsilon
        self.last_activation_layer = arguments.ast_intermediary_activation
        self.projection_dimension = arguments.ast_projection_dimension
        self.intermediary_activation = arguments.ast_intermediary_activation
        self.number_filters_spectrogram = arguments.ast_number_filters_spectrogram
        self.model_name = "AST_Distillation"
        self.use_distillation = True

    def transformer_encoder(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Build the transformer encoder with multiple blocks.

        Each transformer block consists of:
        1. Layer normalization
        2. Multi-head self-attention with residual connection
        3. Layer normalization
        4. Feed-forward network with residual connection

        Args:
            inputs (tensorflow.Tensor): Input tensor of shape (batch_size, sequence_length, projection_dim)

        Returns:
            tensorflow.Tensor: Output tensor of same shape as inputs after transformer processing
        """
        neural_network_flow = inputs

        for block_idx in range(self.number_blocks):
            neural_network_flow_normalized = LayerNormalization(epsilon=self.normalization_epsilon,
                                                                name=f'norm1_block_{block_idx}')(neural_network_flow)

            attention_layer = MultiHeadAttention(key_dim=self.projection_dimension,
                                                 num_heads=self.number_heads,
                                                 dropout=self.dropout,
                                                 name=f'attention_block_{block_idx}')

            attention_output, attention_scores = attention_layer(neural_network_flow_normalized,
                                                                 neural_network_flow_normalized,
                                                                 return_attention_scores=True)

            self.attention_layers.append((attention_layer, attention_scores))
            attention_output = Dropout(self.dropout, name=f'dropout1_block_{block_idx}')(attention_output)
            neural_network_flow = Add(name=f'add1_block_{block_idx}')([attention_output, neural_network_flow])

            neural_network_flow_normalized = LayerNormalization(epsilon=self.normalization_epsilon,
                                        name=f'norm2_block_{block_idx}')(neural_network_flow)

            ffn_dim = self.projection_dimension * 4
            classification_head_output = Dense(ffn_dim,
                                               activation=self.intermediary_activation,
                                               name=f'ffn1_block_{block_idx}')(neural_network_flow_normalized)

            classification_head_output = Dropout(self.dropout,
                                                 name=f'dropout2_block_{block_idx}')(classification_head_output)
            classification_head_output = Dense(self.projection_dimension,
                                               name=f'ffn2_block_{block_idx}')(classification_head_output)
            classification_head_output = Dropout(self.dropout,
                                                 name=f'dropout3_block_{block_idx}')(classification_head_output)
            neural_network_flow = Add(name=f'add2_block_{block_idx}')([classification_head_output, neural_network_flow])

        neural_network_flow = LayerNormalization(epsilon=self.normalization_epsilon, name='final_norm')(neural_network_flow)

        return neural_network_flow

    def build_model(self, number_patches: int = 8) -> tensorflow.keras.models.Model:
        """
        Build the complete Audio Spectrogram Transformer model.

        Model Architecture:
        1. Input patches processing and flattening
        2. Linear projection to embedding dimension
        3. Addition of CLS and/or distillation tokens
        4. Positional embeddings
        5. Transformer encoder blocks
        6. Classification heads with optional distillation

        Args:
            number_patches (int): Number of patches extracted from the spectrogram

        Returns:
            tensorflow.keras.models.Model: Compiled transformer model

        Raises:
            ValueError: If required components are not properly initialized
        """
        self.attention_layers = []

        inputs = Input(shape=(number_patches, self.patch_size[0], self.patch_size[1]))
        input_flatten = TimeDistributed(Flatten())(inputs)
        linear_projection = TimeDistributed(Dense(self.projection_dimension))(input_flatten)

        if self.use_distillation:
            neural_model_flow = DistillationCLSTokenLayer()(linear_projection)
            num_special_tokens = 2
        else:
            from Engine.Layers.CLSTokenLayer import CLSTokenLayer
            neural_model_flow = CLSTokenLayer()(linear_projection)
            num_special_tokens = 1

        positional_embeddings_layer = PositionalEmbeddingsLayer(number_patches + num_special_tokens - 1,
                                                                self.projection_dimension)(neural_model_flow)

        neural_model_flow = neural_model_flow + positional_embeddings_layer

        neural_model_flow = self.transformer_encoder(neural_model_flow)

        if self.use_distillation:
            cls_output = neural_model_flow[:, 0, :]
            dist_output = neural_model_flow[:, 1, :]

            cls_output = Dropout(self.dropout, name='cls_dropout')(cls_output)
            dist_output = Dropout(self.dropout, name='dist_dropout')(dist_output)

            cls_logits = Dense(self.projection_dimension,
                               activation=self.intermediary_activation,
                               name='cls_mlp_hidden')(cls_output)

            cls_logits = Dropout(self.dropout)(cls_logits)
            cls_logits = Dense(self.number_classes,
                               activation='softmax',
                               name='cls_head')(cls_logits)

            distillation_logits = Dense(self.projection_dimension,
                                        activation=self.intermediary_activation,
                                        name='dist_mlp_hidden')(dist_output)

            distillation_logits = Dropout(self.dropout)(distillation_logits)

            distillation_logits = Dense(self.number_classes,
                                        activation='softmax',
                                        name='dist_head')(distillation_logits)

            outputs = tensorflow.keras.layers.Average(name='average_predictions')([cls_logits, distillation_logits])

        else:
            cls_output = neural_model_flow[:, 0, :]
            cls_output = Dropout(self.dropout)(cls_output)

            feedforward_hidden = Dense(self.projection_dimension,
                                       activation=self.intermediary_activation,
                                       name='mlp_head_hidden')(cls_output)

            feedforward_hidden = Dropout(self.dropout)(feedforward_hidden)

            outputs = Dense(self.number_classes,
                            activation='softmax',
                            name='classification_head')(feedforward_hidden)

        self.neural_network_model = models.Model(inputs, outputs, name=self.model_name)
        self.neural_network_model.summary()

        return self.neural_network_model



    def build_attention_model(self):
        """
        Build a model specifically for extracting attention weights.

        Returns:
            tensorflow.keras.models.Model: Model that outputs attention scores from all layers

        Raises:
            ValueError: If main model hasn't been built yet
        """
        if self.neural_network_model is None:
            raise ValueError("The main model must be built before creating attention model.")

        attention_score_tensors = [scores for _, scores in self.attention_layers]

        if not attention_score_tensors:
            print("Warning: No attention scores found in the model.")
            return None

        self.attention_model = models.Model(inputs=self.neural_network_model.input,
                                            outputs=attention_score_tensors,
                                            name='attention_extractor')

        return self.attention_model



    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True,
                          use_early_stopping: bool = True,
                          early_stopping_monitor: str = 'val_loss',
                          early_stopping_patience: int = 10,
                          early_stopping_restore_best: bool = True,
                          early_stopping_min_delta: float = 0.0001):
        """
        Compile and train the Audio Spectrogram Transformer model.

        Args:
            train_data (np.ndarray): Training data samples
            train_labels (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_data (tuple, optional): (val_data, val_labels) for validation
            visualize_attention (bool): Whether to generate attention visualizations after training
            use_early_stopping (bool): Whether to use early stopping callback
            early_stopping_monitor (str): Metric to monitor for early stopping
                                         Options: 'val_loss', 'val_accuracy', 'loss', 'accuracy'
            early_stopping_patience (int): Number of epochs with no improvement after which
                                           training will be stopped
            early_stopping_restore_best (bool): Whether to restore model weights from the best epoch
            early_stopping_min_delta (float): Minimum change in monitored metric to qualify as improvement

        Returns:
            tensorflow.keras.callbacks.History: Training history object

        Example:
            ```python
            # Com early stopping ativo (padrão)
            history = model.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                visualize_attention=True,
                use_early_stopping=True,
                early_stopping_patience=15,
                early_stopping_monitor='val_loss'
            )

            # Sem early stopping
            history = model.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                use_early_stopping=False
            )

            # Monitorando acurácia de validação
            history = model.compile_and_train(
                train_data=X_train,
                train_labels=y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                early_stopping_monitor='val_accuracy',
                early_stopping_patience=20
            )
            ```
        """
        # Compilar o modelo
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

        # Configurar callbacks
        callbacks = []

        if use_early_stopping:
            # Criar callback de early stopping
            early_stopping = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=early_stopping_restore_best,
                min_delta=early_stopping_min_delta,
                verbose=1,
                mode='auto'  # Detecta automaticamente se deve maximizar ou minimizar
            )
            callbacks.append(early_stopping)

        # Treinar o modelo
        training_history = self.neural_network_model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if callbacks else None
        )

        # Opcional: Visualização de atenção após treinamento
        # if visualize_attention and validation_data is not None:
        #     val_data, val_labels = validation_data
        #
        #     self.visualize_attention_flow(val_data,
        #                                   labels=val_labels,
        #                                   num_samples=128,
        #                                   output_dir='Maps_AST')

        return training_history
    # Properties
    @property
    def head_size(self):
        return self._head_size

    @head_size.setter
    def head_size(self, head_size: int):
        self._head_size = head_size

    @property
    def number_heads(self):
        return self._number_heads

    @number_heads.setter
    def number_heads(self, num_heads: int):
        self._number_heads = num_heads

    @property
    def number_blocks(self):
        return self._number_blocks

    @number_blocks.setter
    def number_blocks(self, number_blocks: int):
        self._number_blocks = number_blocks

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, number_classes: int):
        self._number_classes = number_classes

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: tuple):
        self._patch_size = patch_size

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, dropout: float):
        self._dropout = dropout

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, optimizer_function: str):
        self._optimizer_function = optimizer_function

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: str):
        self._loss_function = loss_function

    @property
    def normalization_epsilon(self):
        return self._normalization_epsilon

    @normalization_epsilon.setter
    def normalization_epsilon(self, normalization_epsilon: float):
        self._normalization_epsilon = normalization_epsilon

    @property
    def last_activation_layer(self):
        return self._last_activation_layer

    @last_activation_layer.setter
    def last_activation_layer(self, last_activation_layer: str):
        self._last_activation_layer = last_activation_layer

    @property
    def projection_dimension(self):
        return self._projection_dimension

    @projection_dimension.setter
    def projection_dimension(self, projection_dimension: int):
        self._projection_dimension = projection_dimension

    @property
    def intermediary_activation(self):
        return self._intermediary_activation

    @intermediary_activation.setter
    def intermediary_activation(self, intermediary_activation: str):
        self._intermediary_activation = intermediary_activation

    @property
    def number_filters_spectrogram(self):
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, number_filters_spectrogram: int):
        self._number_filters_spectrogram = number_filters_spectrogram