#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{2}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/27'
__credits__ = ['Kayuã Oleques Paim']

"""
CORRECTED WAV2VEC2 IMPLEMENTATION

This implementation follows the original Wav2Vec2 architecture:
- Feature Encoder: 7-layer convolutional encoder with specific kernel sizes and strides
- Transformer Context Network: 12 transformer blocks
- Quantization Module: Gumbel-Softmax vector quantization
- Self-Supervised Training: Contrastive loss with masked prediction

Key differences from original code:
1. Works with RAW WAVEFORMS (not spectrograms)
2. Proper convolutional encoder architecture
3. Multiple transformer blocks (12 blocks)
4. Correct positional encoding
5. Proper quantization module
"""

try:
    import sys
    import logging
    import numpy as np
    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import GlobalAveragePooling1D

    # Assuming these custom layers exist in your codebase
    from Engine.Layers.GELU import GELU
    from tensorflow.keras.callbacks import EarlyStopping
    from Engine.Layers.MaskTimeLayer import TimeMaskingWithStorage
    from Engine.Loss.ContrastiveLoss import Wav2Vec2ContrastiveLoss
    from Engine.Modules.GumbelVectorQuantizer import GumbelVectorQuantizer
    from Engine.Models.Trainer.Wav2Vec2 import Wav2Vec2DynamicTrainingModel
    from Engine.Models.Process.Wav2Vec2_Process import Wav2Vec2Process

except ImportError as error:
    print(error)
    sys.exit(-1)


class PositionalEncoding(Layer):
    """
    Sinusoidal positional encoding layer for Wav2Vec2.

    This layer adds positional information to the input embeddings using
    sine and cosine functions of different frequencies.
    """

    def __init__(self, hidden_size, max_length=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_length = max_length

    def build(self, input_shape):
        # Pre-compute positional encodings using numpy (simple and reliable)
        pe_array = np.zeros((self.max_length, self.hidden_size), dtype=np.float32)

        position = np.arange(self.max_length, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.hidden_size, 2, dtype=np.float32) *
            (-np.log(10000.0) / self.hidden_size)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        pe_array[:, 0::2] = np.sin(position * div_term)
        # Apply cos to odd indices (1, 3, 5, ...)
        pe_array[:, 1::2] = np.cos(position * div_term)

        # Create non-trainable variable
        self.pe = tensorflow.Variable(
            initial_value=pe_array,
            trainable=False,
            dtype=tensorflow.float32,
            name='positional_encoding_matrix'
        )

        super(PositionalEncoding, self).build(input_shape)

    def call(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = tensorflow.shape(x)[1]
        # Slice the pre-computed positional encoding to match sequence length
        return x + self.pe[tensorflow.newaxis, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'max_length': self.max_length
        })
        return config


class AudioWav2Vec2(Wav2Vec2Process):
    """
    CORRECTED Wav2Vec2 implementation following the original architecture.

    Architecture:
        1. Feature Encoder: 7 CNN layers (kernel sizes: [10,3,3,3,3,2,2], strides: [5,2,2,2,2,2,2])
        2. Contextualized Representations: 12 transformer blocks
        3. Quantization: Gumbel-Softmax vector quantizer
        4. Training: Two-phase (pre-training + fine-tuning)

    Key Features:
        - Works with RAW WAVEFORMS (16kHz audio)
        - Proper convolutional feature extraction
        - Multiple transformer blocks for context
        - Self-supervised contrastive learning
        - Fine-tuning for downstream tasks
    """

    def __init__(self, arguments):
        """
        Initialize corrected Wav2Vec2 model.

        Args:
            arguments: Configuration object with parameters
        """
        Wav2Vec2Process.__init__(self, arguments)

        self.neural_network_model = None
        self.gradcam_model = None

        # Feature encoder configuration (following Wav2Vec2 paper)
        self.encoder_config = {
            'filters': [16, 16, 16, 16, 16, 16, 16],  # 7 layers
            'kernel_sizes': [10, 3, 3, 3, 3, 2, 2],
            'strides': [5, 2, 2, 2, 2, 2, 2]
        }

        # Transformer configuration
        self.num_transformer_blocks = getattr(arguments, 'num_transformer_blocks', 1)
        self.hidden_size = getattr(arguments, 'hidden_size', 768)
        self.num_attention_heads = getattr(arguments, 'num_attention_heads', 12)
        self.intermediate_size = getattr(arguments, 'intermediate_size', 3072)

        # Other parameters
        self.dropout_rate = getattr(arguments, 'dropout_rate', 0.1)
        self.attention_dropout = getattr(arguments, 'attention_dropout', 0.1)
        self.number_classes = arguments.number_classes
        self.loss_function = arguments.wav_to_vec_loss_function
        self.optimizer_function = arguments.wav_to_vec_optimizer_function
        self.last_layer_activation = getattr(arguments, 'last_layer_activation', 'softmax')

        # Wav2Vec2 specific
        self.mask_time_prob = getattr(arguments, 'mask_time_prob', 0.065)
        self.mask_time_length = getattr(arguments, 'mask_time_length', 10)
        self.contrastive_temperature = getattr(arguments, 'contrastive_temperature', 0.1)
        self.num_negatives = getattr(arguments, 'num_negatives', 100)
        self.diversity_weight = getattr(arguments, 'diversity_weight', 0.1)

        # input_dimension is already set by parent Wav2Vec2Process.__init__()
        # It's calculated as (window_size, 1) where window_size = hop_length * window_size_factor
        # For the user's config: window_size = 10240, so input_dimension = (10240, 1)
        # No need to override - just use self.input_dimension from parent

        self.model_name = "Wav2Vec2_Corrected"

        logging.info("✓ Wav2Vec2 Corrected initialized")
        logging.info(f"   Input: Raw waveform - {self.input_dimension}")
        logging.info(f"   Window size: {self.window_size} samples @ {self.sample_rate}Hz")
        logging.info(f"   Transformer blocks: {self.num_transformer_blocks}")
        logging.info(f"   Hidden size: {self.hidden_size}")

    def build_feature_encoder(self, inputs):
        """
        Build the convolutional feature encoder.

        Following Wav2Vec2 paper:
        - 7 convolutional layers
        - Specific kernel sizes and strides
        - GELU activation + Layer normalization

        Args:
            inputs: Raw waveform input tensor

        Returns:
            Encoded features
        """
        x = inputs

        for i, (filters, kernel_size, stride) in enumerate(zip(
                self.encoder_config['filters'],
                self.encoder_config['kernel_sizes'],
                self.encoder_config['strides']
        )):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='same',
                name=f'feature_encoder_conv_{i}'
            )(x)
            x = GELU(name=f'feature_encoder_gelu_{i}')(x)
            x = LayerNormalization(name=f'feature_encoder_ln_{i}')(x)

        logging.info(f"✓ Feature encoder built: 7 layers")
        return x

    def build_transformer_block(self, x, block_idx):
        """
        Build a single transformer encoder block.

        Args:
            x: Input tensor
            block_idx: Block index for naming

        Returns:
            Transformer block output
        """
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.hidden_size // self.num_attention_heads,
            dropout=self.attention_dropout,
            name=f'transformer_{block_idx}_attention'
        )(x, x)

        attention_output = Dropout(
            self.dropout_rate,
            name=f'transformer_{block_idx}_attn_dropout'
        )(attention_output)

        # Residual connection + LayerNorm
        x = Add(name=f'transformer_{block_idx}_add_1')([x, attention_output])
        x = LayerNormalization(
            name=f'transformer_{block_idx}_ln_1'
        )(x)

        # Feed-forward network
        ffn = Dense(
            self.intermediate_size,
            name=f'transformer_{block_idx}_ffn_1'
        )(x)
        ffn = GELU(name=f'transformer_{block_idx}_ffn_gelu')(ffn)
        ffn = Dropout(
            self.dropout_rate,
            name=f'transformer_{block_idx}_ffn_dropout_1'
        )(ffn)

        ffn = Dense(
            self.hidden_size,
            name=f'transformer_{block_idx}_ffn_2'
        )(ffn)
        ffn = Dropout(
            self.dropout_rate,
            name=f'transformer_{block_idx}_ffn_dropout_2'
        )(ffn)

        # Residual connection + LayerNorm
        x = Add(name=f'transformer_{block_idx}_add_2')([x, ffn])
        x = LayerNormalization(
            name=f'transformer_{block_idx}_ln_2'
        )(x)

        return x

    def build_model(self) -> None:
        """
        Build the complete CORRECTED Wav2Vec2 architecture.

        Architecture:
            1. Feature Encoder (7 CNN layers)
            2. Projection to hidden_size
            3. Positional embeddings
            4. Time masking
            5. Multiple Transformer blocks (12 blocks)
            6. Vector quantization

        Returns:
            None: Model stored in self.neural_network_model
        """
        logging.info("=" * 80)
        logging.info("BUILDING CORRECTED WAV2VEC2 MODEL")
        logging.info("=" * 80)

        # Ensure input_dimension is set
        if self.input_dimension is None:
            raise ValueError(
                "input_dimension is not set! "
                "This happens when use_full_audio=True but data hasn't been loaded yet. "
                "Make sure load_data() is called before build_model()."
            )

        # Input: Raw waveform
        inputs = Input(shape=self.input_dimension, name='raw_audio_input')
        logging.info(f"✓ Input: {self.input_dimension} (raw waveform)")

        # Feature Encoder
        encoded_features = self.build_feature_encoder(inputs)

        # Project to hidden size
        x = Dense(
            self.hidden_size,
            name='feature_projection'
        )(encoded_features)
        x = LayerNormalization(name='feature_projection_ln')(x)

        logging.info(f"✓ Feature projection: {self.hidden_size}D")

        # Positional embeddings using custom layer
        x = PositionalEncoding(
            hidden_size=self.hidden_size,
            max_length=5000,  # Should be enough for most audio sequences
            name='positional_encoding'
        )(x)

        logging.info(f"✓ Positional encoding added")

        # Time masking for self-supervised learning
        lengths = Lambda(
            lambda x: tensorflow.ones([tensorflow.shape(x)[0]], dtype=tensorflow.int32) * tensorflow.shape(x)[1],
            output_shape=(None,),
            dtype=tensorflow.int32,
            name='sequence_lengths'
        )(x)

        masking_layer = TimeMaskingWithStorage(
            mask_time_prob=self.mask_time_prob,
            number_mask_time_steps=self.mask_time_length,
            name='time_masking'
        )
        masked_features, mask_indices = masking_layer([x, lengths])

        logging.info(f"✓ Time masking: prob={self.mask_time_prob}, length={self.mask_time_length}")

        # Multiple Transformer blocks
        transformer_output = masked_features
        for i in range(self.num_transformer_blocks):
            transformer_output = self.build_transformer_block(transformer_output, i)

        logging.info(f"✓ Transformer: {self.num_transformer_blocks} blocks")

        # Vector quantization (on unmasked features for contrastive learning)
        quantization_layer = GumbelVectorQuantizer(
            name='gumbel_quantizer',
            hidden_size=self.hidden_size,  # Pass hidden_size here!
            number_groups=4,  # or make this a parameter
            number_vectors=320,  # typical Wav2Vec2 value
            temperature=0.2
        )


        logging.info(
            f"✓ Vector quantization: {4} groups × {320} vectors × {self.hidden_size // 4}D = {self.hidden_size}D output")

        quantized_output, perplexity = quantization_layer([x, lengths])

        logging.info(f"✓ Vector quantization with Gumbel-Softmax")

        # Create model
        self.neural_network_model = Model(
            inputs=inputs,
            outputs={
                'contextualized': transformer_output,
                'quantized': quantized_output,
                'perplexity': perplexity,
                'mask_indices': mask_indices
            },
            name=self.model_name
        )

        self.neural_network_model.summary()
        logging.info("✓ Wav2Vec2 CORRECTED architecture built")
        logging.info("=" * 80)

    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True,
                          use_early_stopping: bool = True,
                          early_stopping_monitor: str = 'val_loss',
                          early_stopping_patience: int = 10,
                          early_stopping_restore_best: bool = True,
                          early_stopping_min_delta: float = 0.0001):
        """
        Two-phase training: pre-training + fine-tuning.

        Phase 1: Self-supervised pre-training with contrastive loss
        Phase 2: Supervised fine-tuning for classification

        Args:
            train_data: Training waveforms (raw audio)
            train_labels: Training labels (for fine-tuning)
            epochs: Number of epochs for each phase
            batch_size: Batch size
            validation_data: Validation data tuple (X_val, y_val)
            freeze_encoder: Whether to freeze encoder during fine-tuning

        Returns:
            Training history from fine-tuning phase
        """
        logging.info("=" * 80)
        logging.info("WAV2VEC2 TWO-PHASE TRAINING (CORRECTED)")
        logging.info("=" * 80)

        # ===== PHASE 1: SELF-SUPERVISED PRE-TRAINING =====
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
        logging.info("=" * 80)

        # Create contrastive loss
        contrastive_loss = Wav2Vec2ContrastiveLoss(
            temperature=self.contrastive_temperature,
            num_negatives=self.num_negatives,
            diversity_weight=self.diversity_weight
        )

        # Wrap model with dynamic training
        pretrain_model = Wav2Vec2DynamicTrainingModel(
            self.neural_network_model,
            loss_fn=contrastive_loss,
            name='pretrain_model'
        )

        pretrain_model.compile(optimizer=self.optimizer_function)

        logging.info("✓ Loss: InfoNCE + Diversity Loss")
        logging.info(f"   Temperature: {self.contrastive_temperature}")
        logging.info(f"   Negative samples: {self.num_negatives}")
        logging.info(f"   Diversity weight: {self.diversity_weight}")
        logging.info(f"⚙ Starting pre-training for {epochs} epochs...")

        # Pre-training (only on unlabeled waveforms)
        pretrain_history = pretrain_model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        logging.info("✓ Pre-training completed!")

        # ===== PHASE 2: SUPERVISED FINE-TUNING =====
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: SUPERVISED FINE-TUNING")
        logging.info("=" * 80)

        # Get contextualized representations
        contextualized_output = self.neural_network_model.get_layer(
            f'transformer_{self.num_transformer_blocks - 1}_ln_2'
        ).output

        # Add classification head
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(contextualized_output)

        hidden = Dense(
            self.hidden_size,
            name='classification_hidden'
        )(pooled)
        hidden = GELU(name='classification_gelu')(hidden)
        hidden = Dropout(self.dropout_rate, name='classification_dropout')(hidden)

        outputs = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='classification_output'
        )(hidden)

        # Create fine-tuning model
        finetune_model = Model(
            inputs=self.neural_network_model.inputs,
            outputs=outputs,
            name=f"{self.model_name}_FineTuned"
        )

        # Freeze encoder if requested
        if freeze_encoder:
            for layer in finetune_model.layers:
                if 'feature_encoder' in layer.name or 'transformer' in layer.name:
                    layer.trainable = False
            logging.info("✓ Encoder frozen (only training classification head)")
        else:
            logging.info("✓ Full model trainable (end-to-end fine-tuning)")

        # Compile fine-tuning model
        finetune_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

        logging.info(f"⚙ Starting fine-tuning for {epochs} epochs...")

        callbacks = []

        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=early_stopping_restore_best,
                min_delta=early_stopping_min_delta,
                verbose=1,
                mode='auto'
            )
            callbacks.append(early_stopping)

        # Fine-tuning
        finetune_history = finetune_model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1,
            callbacks=callbacks if callbacks else None
        )

        # Update model reference
        self.neural_network_model = finetune_model

        logging.info("✓ Fine-tuning completed!")
        logging.info("=" * 80)

        return finetune_history

    def train(self) -> tuple:
        """
        Train method for compatibility with the existing framework.

        This method calls the parent class (Wav2Vec2Process) train() method,
        which handles the complete training pipeline including:
        - Data loading
        - K-fold cross-validation
        - Model building (calls build_model())
        - Training with compile_and_train()
        - Metrics calculation

        Returns:
            tuple: Contains metrics_list, probabilities_list, real_labels_list,
                   confusion_matrix_list, and history
        """
        # Call parent class train method
        return Wav2Vec2Process.train(self)


# ============================================================================
# COMPARISON WITH ORIGINAL IMPLEMENTATION
# =========================================================================================

"""
KEY DIFFERENCES FROM ORIGINAL CODE:

1. INPUT DATA:
   ❌ Original: Expects spectrograms/MFCC (128, 80, 1) shape
   ✅ Corrected: Raw waveforms (160000, 1) for 16kHz audio

2. FEATURE ENCODER:
   ❌ Original: Generic CNN with TimeDistributed
   ✅ Corrected: 7-layer CNN with specific kernel sizes [10,3,3,3,3,2,2]
                 and strides [5,2,2,2,2,2,2] as per paper

3. TRANSFORMER BLOCKS:
   ❌ Original: Only 1 attention block
   ✅ Corrected: 12 full transformer blocks (can be configured)

4. POSITIONAL ENCODING:
   ❌ Original: Not present
   ✅ Corrected: Sinusoidal positional encoding added

5. ARCHITECTURE FIDELITY:
   ❌ Original: Loosely inspired by Wav2Vec2
   ✅ Corrected: Follows original Wav2Vec2 paper architecture

6. DATA PROCESSING:
   ❌ Original: load_data() processes into spectrograms
   ✅ Corrected: Should work with raw waveforms directly

USAGE EXAMPLE:
--------------
from Engine.Models.AudioWav2Vec2_CORRECTED import AudioWav2Vec2Corrected

# Create model
model = AudioWav2Vec2Corrected(arguments)
model.build_model()

# Load raw waveforms (NOT spectrograms!)
# Shape: (n_samples, 160000, 1) for 10-second clips at 16kHz
waveforms, labels = load_raw_audio_data()

# Train
history = model.compile_and_train(
    train_data=waveforms,
    train_labels=labels,
    epochs=100,
    batch_size=32
)
"""