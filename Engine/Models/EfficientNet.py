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
    import sys
    import numpy
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom, gaussian_filter
    import seaborn as sns

    import tensorflow as tf
    import tensorflow
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import DepthwiseConv2D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Multiply

    from Engine.Models.Process.EfficientNet_Process import ProcessEfficientNet
    from Engine.GradientMap.EfficientNetGradientMaps import EfficientNetGradientMaps

except ImportError as error:
    print(error)
    sys.exit(-1)


# ============================================================================
# COMPONENTES PERSONALIZADOS DO EFFICIENTNET
# ============================================================================

class Swish(Layer):
    """
    Swish activation function: x * sigmoid(x)
    Também conhecida como SiLU (Sigmoid Linear Unit)
    """

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


class SEBlock(Layer):
    """
    Squeeze-and-Excitation Block

    Aplica atenção aos canais da feature map, permitindo que o modelo
    aprenda quais canais são mais importantes.

    Args:
        filters: Número de filtros de saída
        se_ratio: Razão de redução no squeeze (padrão: 0.25)
    """

    def __init__(self, filters, se_ratio=0.25, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio
        self.num_reduced_filters = max(1, int(filters * se_ratio))

    def build(self, input_shape):
        self.squeeze = GlobalAveragePooling2D(name=f'{self.name}_squeeze')
        self.excitation_1 = Dense(
            self.num_reduced_filters,
            activation='swish',
            name=f'{self.name}_excite_fc1'
        )
        self.excitation_2 = Dense(
            self.filters,
            activation='sigmoid',
            name=f'{self.name}_excite_fc2'
        )

    def call(self, inputs):
        # Squeeze: Global Average Pooling
        se = self.squeeze(inputs)
        # Excitation: FC -> Swish -> FC -> Sigmoid
        se = self.excitation_1(se)
        se = self.excitation_2(se)
        # Scale: Multiply input by channel weights
        se = tf.reshape(se, [-1, 1, 1, self.filters])
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'se_ratio': self.se_ratio
        })
        return config


class MBConvBlock(Layer):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv)

    Componente fundamental do EfficientNet. Consiste em:
    1. Expansion: Conv 1x1 para aumentar canais
    2. Depthwise Conv: Convolução espacial eficiente
    3. SE Block: Atenção aos canais
    4. Projection: Conv 1x1 para reduzir canais
    5. Skip connection (se aplicável)

    Args:
        filters_in: Canais de entrada
        filters_out: Canais de saída
        kernel_size: Tamanho do kernel depthwise
        strides: Stride da convolução depthwise
        expand_ratio: Fator de expansão (padrão: 6)
        se_ratio: Razão do SE block (padrão: 0.25)
        drop_rate: Taxa de dropout estocástico (padrão: 0)
    """

    def __init__(self, filters_in, filters_out, kernel_size, strides,
                 expand_ratio=6, se_ratio=0.25, drop_rate=0.0, **kwargs):
        super(MBConvBlock, self).__init__(**kwargs)

        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_rate = drop_rate

        # Se stride=1 e canais iguais, usa skip connection
        self.use_residual = (strides == 1 and filters_in == filters_out)

        # Número de filtros expandidos
        self.filters_expanded = filters_in * expand_ratio

    def build(self, input_shape):
        # 1. Expansion phase (se expand_ratio > 1)
        if self.expand_ratio != 1:
            self.expand_conv = Conv2D(
                self.filters_expanded,
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=f'{self.name}_expand_conv'
            )
            self.expand_bn = BatchNormalization(name=f'{self.name}_expand_bn')
            self.expand_activation = Swish(name=f'{self.name}_expand_swish')

        # 2. Depthwise Convolution
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            name=f'{self.name}_dwconv'
        )
        self.depthwise_bn = BatchNormalization(name=f'{self.name}_dwconv_bn')
        self.depthwise_activation = Swish(name=f'{self.name}_dwconv_swish')

        # 3. Squeeze-and-Excitation
        if self.se_ratio > 0:
            self.se_block = SEBlock(
                self.filters_expanded if self.expand_ratio != 1 else self.filters_in,
                se_ratio=self.se_ratio,
                name=f'{self.name}_se'
            )

        # 4. Projection phase
        self.project_conv = Conv2D(
            self.filters_out,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=f'{self.name}_project_conv'
        )
        self.project_bn = BatchNormalization(name=f'{self.name}_project_bn')

        # 5. Stochastic Depth (opcional)
        if self.drop_rate > 0 and self.use_residual:
            self.dropout = Dropout(
                self.drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=f'{self.name}_drop'
            )

    def call(self, inputs, training=None):
        x = inputs

        # 1. Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_activation(x)

        # 2. Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)

        # 3. Squeeze-and-Excitation
        if self.se_ratio > 0:
            x = self.se_block(x)

        # 4. Projection
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        # 5. Skip connection + Stochastic Depth
        if self.use_residual:
            if self.drop_rate > 0:
                x = self.dropout(x, training=training)
            x = Add(name=f'{self.name}_add')([inputs, x])

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters_in': self.filters_in,
            'filters_out': self.filters_out,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'expand_ratio': self.expand_ratio,
            'se_ratio': self.se_ratio,
            'drop_rate': self.drop_rate
        })
        return config


# ============================================================================
# CONFIGURAÇÕES DE SCALING PARA CADA VERSÃO DO EFFICIENTNET
# ============================================================================

EFFICIENTNET_CONFIGS = {
    'B0': {
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'resolution': 224,
        'dropout_rate': 0.2
    },
    'B1': {
        'width_coefficient': 1.0,
        'depth_coefficient': 1.1,
        'resolution': 240,
        'dropout_rate': 0.2
    },
    'B2': {
        'width_coefficient': 1.1,
        'depth_coefficient': 1.2,
        'resolution': 260,
        'dropout_rate': 0.3
    },
    'B3': {
        'width_coefficient': 1.2,
        'depth_coefficient': 1.4,
        'resolution': 300,
        'dropout_rate': 0.3
    },
    'B4': {
        'width_coefficient': 1.4,
        'depth_coefficient': 1.8,
        'resolution': 380,
        'dropout_rate': 0.4
    },
    'B5': {
        'width_coefficient': 1.6,
        'depth_coefficient': 2.2,
        'resolution': 456,
        'dropout_rate': 0.4
    },
    'B6': {
        'width_coefficient': 1.8,
        'depth_coefficient': 2.6,
        'resolution': 528,
        'dropout_rate': 0.5
    },
    'B7': {
        'width_coefficient': 2.0,
        'depth_coefficient': 3.1,
        'resolution': 600,
        'dropout_rate': 0.5
    }
}

# Configuração base dos blocos (B0)
BASE_BLOCKS_CONFIG = [
    # (expand_ratio, channels, num_blocks, kernel_size, stride)
    (1, 16, 1, 3, 1),  # Stage 1
    (6, 24, 2, 3, 2),  # Stage 2
    (6, 40, 2, 5, 2),  # Stage 3
    (6, 80, 3, 3, 2),  # Stage 4
    (6, 112, 3, 5, 1),  # Stage 5
    (6, 192, 4, 5, 2),  # Stage 6
    (6, 320, 1, 3, 1),  # Stage 7
]


def round_filters(filters, width_coefficient, divisor=8):
    """Arredonda o número de filtros baseado no width coefficient."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Garante que não diminui mais de 10%
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Arredonda o número de repetições baseado no depth coefficient."""
    return int(np.ceil(depth_coefficient * repeats))


# ============================================================================
# CLASSE PRINCIPAL DO EFFICIENTNET FROM SCRATCH
# ============================================================================

class EfficientNet(ProcessEfficientNet, EfficientNetGradientMaps):
    """
    EfficientNet FROM SCRATCH - Implementação Totalmente Personalizável

    Esta implementação constrói o EfficientNet do zero usando os blocos
    MBConv, SE blocks e compound scaling, sem depender do Keras Applications.

    CARACTERÍSTICAS:
    ===============
    1. ✅ Compound Scaling personalizável (width, depth, resolution)
    2. ✅ MBConv blocks com Squeeze-and-Excitation
    3. ✅ Suporte para todas as versões (B0-B7)
    4. ✅ Stochastic Depth para regularização
    5. ✅ Swish activation function
    6. ✅ Grad-CAM++ e XAI integrados
    7. ✅ Totalmente personalizável e extensível
    """

    def __init__(self, arguments):
        ProcessEfficientNet.__init__(self, arguments)
        self.neural_network_model = None
        self.gradcam_model = None
        self.loss_function = arguments.efficientnet_loss_function
        self.optimizer_function = arguments.efficientnet_optimizer_function
        self.number_filters_spectrogram = arguments.efficientnet_number_filters_spectrogram
        self.input_dimension = arguments.efficientnet_input_dimension
        self.efficientnet_version = arguments.efficientnet_version
        self.number_classes = arguments.number_classes
        self.dropout_rate = arguments.efficientnet_dropout_rate
        self.last_layer_activation = arguments.efficientnet_last_layer_activation
        self.model_name = f"EfficientNet{self.efficientnet_version}_Custom"

        # Parâmetros personalizáveis
        self.se_ratio = getattr(arguments, 'efficientnet_se_ratio', 0.25)
        self.stochastic_depth_rate = getattr(arguments, 'efficientnet_stochastic_depth', 0.2)

        # Custom scaling (opcional - sobrescreve config padrão)
        self.custom_width = getattr(arguments, 'efficientnet_custom_width', None)
        self.custom_depth = getattr(arguments, 'efficientnet_custom_depth', None)

        # Set modern style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def build_model(self) -> None:
        """
        Constrói o modelo EfficientNet from scratch.
        """
        print(f"\n{'=' * 70}")
        print(f"🔨 CONSTRUINDO EFFICIENTNET-{self.efficientnet_version} FROM SCRATCH")
        print(f"{'=' * 70}")

        # Obter configuração de scaling
        if self.efficientnet_version not in EFFICIENTNET_CONFIGS:
            raise ValueError(f"Versão inválida: {self.efficientnet_version}. "
                             f"Escolha entre: {list(EFFICIENTNET_CONFIGS.keys())}")

        config = EFFICIENTNET_CONFIGS[self.efficientnet_version]
        width_coef = self.custom_width or config['width_coefficient']
        depth_coef = self.custom_depth or config['depth_coefficient']

        print(f"📊 Width Coefficient: {width_coef}")
        print(f"📊 Depth Coefficient: {depth_coef}")
        print(f"📊 SE Ratio: {self.se_ratio}")
        print(f"📊 Stochastic Depth: {self.stochastic_depth_rate}")
        print(f"{'=' * 70}\n")

        inputs = Input(shape=self.input_dimension, name='input_layer')

        # Stem: Initial Convolution
        x = Conv2D(
            round_filters(32, width_coef),
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            name='stem_conv'
        )(inputs)
        x = BatchNormalization(name='stem_bn')(x)
        x = Swish(name='stem_swish')(x)

        # Building blocks
        block_num = 0
        total_blocks = sum([round_repeats(config[2], depth_coef)
                            for config in BASE_BLOCKS_CONFIG])

        for stage_idx, (expand_ratio, channels, num_blocks, kernel_size, stride) in enumerate(BASE_BLOCKS_CONFIG):
            # Aplicar scaling
            num_blocks = round_repeats(num_blocks, depth_coef)
            channels = round_filters(channels, width_coef)

            print(f"Stage {stage_idx + 1}: {num_blocks} blocks, {channels} channels")

            for block_idx in range(num_blocks):
                # Calcular drop rate progressivo (stochastic depth)
                drop_rate = self.stochastic_depth_rate * float(block_num) / total_blocks

                # Stride apenas no primeiro bloco de cada stage
                block_stride = stride if block_idx == 0 else 1

                # Canais de entrada
                filters_in = int(x.shape[-1])

                x = MBConvBlock(
                    filters_in=filters_in,
                    filters_out=channels,
                    kernel_size=kernel_size,
                    strides=block_stride,
                    expand_ratio=expand_ratio,
                    se_ratio=self.se_ratio,
                    drop_rate=drop_rate,
                    name=f'block{stage_idx + 1}{chr(97 + block_idx)}'
                )(x)

                block_num += 1

        # Head: Final Convolution
        x = Conv2D(
            round_filters(1280, width_coef),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name='head_conv'
        )(x)
        x = BatchNormalization(name='head_bn')(x)
        x = Swish(name='head_swish')(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)

        # Dropout
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, name='top_dropout')(x)

        # Classification head
        outputs = Dense(
            self.number_classes,
            activation=self.last_layer_activation,
            name='output_layer'
        )(x)

        # Criar modelo
        self.neural_network_model = Model(inputs=inputs, outputs=outputs, name=self.model_name)

        print(f"\n{'=' * 70}")
        print(f"✅ MODELO CONSTRUÍDO COM SUCESSO!")
        print(f"{'=' * 70}")

        self.neural_network_model.summary()

        # Contar parâmetros
        total_params = self.neural_network_model.count_params()
        print(f"\n📈 Total de parâmetros: {total_params:,}")
        print(f"{'=' * 70}\n")

    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True,
                          use_early_stopping: bool = True,
                          early_stopping_monitor: str = 'val_loss',
                          early_stopping_patience: int = 10,
                          early_stopping_restore_best: bool = True,
                          early_stopping_min_delta: float = 0.0001) -> tensorflow.keras.callbacks.History:
        """
        Compila e treina o modelo EfficientNet personalizado.
        """
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

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



        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if callbacks else None
        )

        # if validation_data is not None:
        #     print(f"Acurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")

        # if generate_gradcam and validation_data is not None:
        #     val_data, val_labels = validation_data
        #
        #     stats = self.generate_validation_visualizations(
        #         validation_data=val_data,
        #         validation_labels=val_labels,
        #         num_samples=128,
        #         output_dir='Maps_EfficientNet_Custom',
        #         xai_method=xai_method
        #     )

        return training_history

    def compile_model(self) -> None:
        """Compila o modelo EfficientNet."""
        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

    @staticmethod
    def smooth_heatmap(heatmap: numpy.ndarray, sigma: float = 2.0) -> numpy.ndarray:
        """Aplica suavização Gaussiana ao heatmap."""
        return gaussian_filter(heatmap, sigma=sigma)

    @staticmethod
    def interpolate_heatmap(heatmap: numpy.ndarray, target_shape: tuple,
                            smooth: bool = True) -> numpy.ndarray:
        """Interpola heatmap para o tamanho do espectrograma."""
        if not isinstance(heatmap, numpy.ndarray):
            heatmap = numpy.array(heatmap)

        if len(heatmap.shape) == 2:
            zoom_factors = (target_shape[0] / heatmap.shape[0], target_shape[1] / heatmap.shape[1])
            interpolated = zoom(heatmap, zoom_factors, order=3)

        elif len(heatmap.shape) == 1:
            temporal_interp = zoom(heatmap, (target_shape[1] / heatmap.shape[0],), order=3)
            freq_profile = numpy.linspace(1.0, 0.6, target_shape[0])
            interpolated = freq_profile[:, numpy.newaxis] * temporal_interp[numpy.newaxis, :]

        else:
            raise ValueError(f"Heatmap shape inesperado: {heatmap.shape}")

        if interpolated.shape != target_shape:
            zoom_factors_adjust = (target_shape[0] / interpolated.shape[0],
                                   target_shape[1] / interpolated.shape[1])
            interpolated = zoom(interpolated, zoom_factors_adjust, order=3)

        if smooth:
            interpolated = gaussian_filter(interpolated, sigma=2.0)

        return interpolated

    def plot_gradcam_modern(self, input_sample: np.ndarray, heatmap: np.ndarray,
                            class_idx: int, predicted_class: int, true_label: int = None,
                            confidence: float = None, xai_method: str = 'gradcam++',
                            save_path: str = None, show_plot: bool = True) -> None:
        """Visualização moderna do GradCAM."""
        if len(input_sample.shape) == 4:
            input_sample = input_sample[0]

        if len(input_sample.shape) == 3:
            input_sample_2d = input_sample[:, :, 0]
        else:
            input_sample_2d = input_sample

        interpolated_heatmap = self.interpolate_heatmap(heatmap, input_sample_2d.shape, smooth=True)

        fig = plt.figure(figsize=(20, 6), facecolor='white')
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        cmap_input = 'viridis'
        cmap_heatmap = 'jet'

        # 1. Espectrograma Original
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_sample_2d, cmap=cmap_input, aspect='auto', interpolation='bilinear')
        ax1.set_title('🎵 Spectrogram', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Temporal Frames', fontsize=10)
        ax1.set_ylabel('Frequency Bins', fontsize=10)
        ax1.grid(False)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'🔥 Activation Map ({xai_method.upper()})', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Temporal Frames', fontsize=10)
        ax2.set_ylabel('Frequency Bins', fontsize=10)
        ax2.grid(False)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        input_normalized = (input_sample_2d - input_sample_2d.min()) / (
                input_sample_2d.max() - input_sample_2d.min() + 1e-10)
        ax3.imshow(input_normalized, cmap='gray', aspect='auto', interpolation='bilinear')
        im3 = ax3.imshow(interpolated_heatmap, cmap=cmap_heatmap,
                         alpha=0.4, aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title('🎨 Overlap', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Temporal Frames', fontsize=10)
        ax3.set_ylabel('Frequency Bins', fontsize=10)
        ax3.grid(False)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=9)

        # 4. Temporal Importance Profile
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_importance = np.mean(interpolated_heatmap, axis=0)
        time_steps = np.arange(len(temporal_importance))
        temporal_smooth = gaussian_filter(temporal_importance, sigma=2)

        ax4.fill_between(time_steps, temporal_smooth, alpha=0.3, color='#FF6B6B')
        ax4.plot(time_steps, temporal_smooth, linewidth=2.5, color='#FF6B6B', label='Smoothed Profile')
        ax4.plot(time_steps, temporal_importance, linewidth=1, alpha=0.5,
                 color='#4ECDC4', linestyle='--', label='Original')

        ax4.set_xlabel('Temporal Frame', fontsize=10)
        ax4.set_ylabel('Average Importance', fontsize=10)
        ax4.set_title('📈 Temporal Importance', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_xlim([0, len(temporal_importance)])
        ax4.set_ylim([0, 1])

        # Super title
        pred_status = '✅' if predicted_class == true_label else '❌'
        conf_str = f' | Confidence: {confidence:.1%}' if confidence is not None else ''

        if true_label is not None:
            suptitle = f'{pred_status} Predicted: Class {predicted_class} | True: Class {true_label}{conf_str}'
        else:
            suptitle = f'Predicted: Class {predicted_class}{conf_str}'

        fig.suptitle(f'{suptitle} - {self.model_name}', fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

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
        """Gera visualizações XAI para amostras de validação."""
        import os
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

        for i, idx in enumerate(selected_indices):
            try:
                sample = validation_data[idx]
                true_label = true_labels[idx]
                predicted = predicted_classes[idx]
                confidence = confidences[idx]

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

                self.plot_gradcam_modern(sample, heatmap,
                                         predicted, predicted,
                                         true_label, confidence=confidence,
                                         xai_method=xai_method,
                                         save_path=save_path,
                                         show_plot=False)

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

        return stats

    # Properties
    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

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
    def number_filters_spectrogram(self):
        return self._number_filters_spectrogram

    @number_filters_spectrogram.setter
    def number_filters_spectrogram(self, value):
        self._number_filters_spectrogram = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def efficientnet_version(self):
        return self._efficientnet_version

    @efficientnet_version.setter
    def efficientnet_version(self, value):
        self._efficientnet_version = value

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