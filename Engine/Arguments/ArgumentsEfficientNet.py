#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/23'
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
    import os
    import sys
    import glob

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_OVERLAP = 1
DEFAULT_SIZE_BATCH = 32
DEFAULT_HOP_LENGTH = 256

DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_NUMBER_SPLITS = 5
DEFAULT_NUMBER_CLASSES = 4
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_NUMBER_EPOCHS = 10
DEFAULT_SAMPLE_RATE = 8000

DEFAULT_WINDOW_SIZE_FACTOR = 40
DEFAULT_FILE_EXTENSION = "*.wav"
DEFAULT_DECIBEL_SCALE_FACTOR = 80
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

DEFAULT_INPUT_DIMENSION = (512, 40, 1)  # Formato compatível com EfficientNet
DEFAULT_LAST_LAYER_ACTIVATION = 'softmax'
DEFAULT_NUMBER_FILTERS_SPECTROGRAM = 512
DEFAULT_EFFICIENTNET_VERSION = 'B0'
DEFAULT_LOSS_FUNCTION = 'sparse_categorical_crossentropy'

# Novos parâmetros para implementação from scratch
DEFAULT_SE_RATIO = 0.25  # Squeeze-and-Excitation ratio
DEFAULT_STOCHASTIC_DEPTH = 0.2  # Stochastic Depth drop rate


def add_efficientnet_arguments(parser):
    """
    Adiciona argumentos do EfficientNet FROM SCRATCH ao parser.

    PARÂMETROS PERSONALIZÁVEIS:
    ===========================
    - Versão do EfficientNet (B0-B7)
    - Compound Scaling personalizado (width, depth)
    - Squeeze-and-Excitation ratio
    - Stochastic Depth rate
    - Todas as configurações de treinamento

    EXEMPLO DE USO:
    ==============
    # EfficientNet-B3 padrão
    python main.py --efficientnet_version B3

    # EfficientNet customizado
    python main.py --efficientnet_version B0 \
                   --efficientnet_custom_width 1.5 \
                   --efficientnet_custom_depth 2.0 \
                   --efficientnet_se_ratio 0.3 \
                   --efficientnet_stochastic_depth 0.3
    """

    # ========================================================================
    # CONFIGURAÇÕES BÁSICAS
    # ========================================================================

    parser.add_argument(
        '--efficientnet_version',
        type=str,
        default=DEFAULT_EFFICIENTNET_VERSION,
        choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        help='Versão do EfficientNet (B0-B7). '
             'B0 é o menor e mais rápido, B7 é o maior e mais preciso.'
    )

    parser.add_argument(
        '--efficientnet_input_dimension',
        default=DEFAULT_INPUT_DIMENSION,
        help='Dimensões da entrada (altura, largura, canais). '
             'Formato: (H, W, C). EfficientNet usa 3 canais (RGB-like).'
    )

    parser.add_argument(
        '--efficientnet_number_filters_spectrogram',
        type=int,
        default=DEFAULT_NUMBER_FILTERS_SPECTROGRAM,
        help='Número de filtros (mel bins) no espectrograma. '
             'Valores típicos: 64, 128, 256.'
    )

    # ========================================================================
    # TREINAMENTO E OTIMIZAÇÃO
    # ========================================================================

    parser.add_argument(
        '--efficientnet_optimizer_function',
        type=str,
        default=DEFAULT_OPTIMIZER_FUNCTION,
        help='Otimizador para treinamento. '
             'Opções: adam, sgd, rmsprop, adamw.'
    )

    parser.add_argument(
        '--efficientnet_loss_function',
        type=str,
        default=DEFAULT_LOSS_FUNCTION,
        help='Função de perda. '
             'sparse_categorical_crossentropy para labels inteiros, '
             'categorical_crossentropy para one-hot encoding.'
    )

    parser.add_argument(
        '--efficientnet_dropout_rate',
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help='Taxa de dropout antes da camada de classificação. '
             'Ajuda a prevenir overfitting. Range: 0.0-0.5.'
    )

    parser.add_argument(
        '--efficientnet_last_layer_activation',
        type=str,
        default=DEFAULT_LAST_LAYER_ACTIVATION,
        help='Função de ativação da camada de saída. '
             'softmax para classificação multiclasse, '
             'sigmoid para classificação binária.'
    )

    # ========================================================================
    # PROCESSAMENTO DE ÁUDIO
    # ========================================================================

    parser.add_argument(
        '--efficientnet_hop_length',
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help='Hop length para STFT (Short-Time Fourier Transform). '
             'Determina a resolução temporal do espectrograma.'
    )

    parser.add_argument(
        '--efficientnet_window_size',
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help='Tamanho da janela FFT. '
             'Determina a resolução frequencial do espectrograma.'
    )

    parser.add_argument(
        '--efficientnet_window_size_factor',
        type=int,
        default=DEFAULT_WINDOW_SIZE_FACTOR,
        help='Fator multiplicador do tamanho da janela. '
             'Janela efetiva = hop_length * (factor - 1).'
    )

    parser.add_argument(
        '--efficientnet_overlap',
        type=int,
        default=DEFAULT_OVERLAP,
        help='Sobreposição entre janelas de processamento.'
    )

    parser.add_argument(
        '--efficientnet_decibel_scale_factor',
        type=float,
        default=DEFAULT_DECIBEL_SCALE_FACTOR,
        help='Fator de escala para conversão em decibéis. '
             'Normaliza o espectrograma.'
    )

    # ========================================================================
    # PARÂMETROS PERSONALIZÁVEIS DO EFFICIENTNET FROM SCRATCH
    # ========================================================================

    parser.add_argument(
        '--efficientnet_se_ratio',
        type=float,
        default=DEFAULT_SE_RATIO,
        help='Squeeze-and-Excitation ratio. '
             'Controla a redução de dimensionalidade no SE block. '
             'Valores menores = mais compressão. Range: 0.0-1.0. '
             'Recomendado: 0.25 (padrão), 0.0 desabilita SE.'
    )

    parser.add_argument(
        '--efficientnet_stochastic_depth',
        type=float,
        default=DEFAULT_STOCHASTIC_DEPTH,
        help='Taxa de Stochastic Depth (drop path). '
             'Regularização que droppa blocos inteiros durante treinamento. '
             'Range: 0.0-0.5. Maior = mais regularização. '
             'Recomendado: 0.2 (padrão).'
    )

    parser.add_argument(
        '--efficientnet_custom_width',
        type=float,
        default=None,
        help='Width coefficient personalizado (sobrescreve configuração padrão). '
             'Controla o número de filtros em cada camada. '
             'Valores maiores = modelo mais largo. '
             'Exemplo: 1.0 (B0), 1.4 (B4), 2.0 (B7).'
    )

    parser.add_argument(
        '--efficientnet_custom_depth',
        type=float,
        default=None,
        help='Depth coefficient personalizado (sobrescreve configuração padrão). '
             'Controla o número de blocos repetidos. '
             'Valores maiores = modelo mais profundo. '
             'Exemplo: 1.0 (B0), 1.8 (B4), 3.1 (B7).'
    )

    # ========================================================================
    # CONFIGURAÇÕES AVANÇADAS DE ARQUITETURA
    # ========================================================================

    parser.add_argument(
        '--efficientnet_stem_filters',
        type=int,
        default=32,
        help='Número de filtros na camada stem inicial. '
             'Será escalado pelo width coefficient.'
    )

    parser.add_argument(
        '--efficientnet_head_filters',
        type=int,
        default=1280,
        help='Número de filtros na camada head final. '
             'Será escalado pelo width coefficient.'
    )

    parser.add_argument(
        '--efficientnet_activation',
        type=str,
        default='swish',
        choices=['swish', 'relu', 'gelu'],
        help='Função de ativação interna. '
             'swish (padrão) oferece melhor performance.'
    )

    # ========================================================================
    # VISUALIZAÇÃO E XAI (Explainable AI)
    # ========================================================================

    parser.add_argument(
        '--efficientnet_xai_method',
        type=str,
        default='gradcam++',
        choices=['gradcam', 'gradcam++', 'scorecam'],
        help='Método de explicabilidade (XAI). '
             'gradcam++: Melhor localização. '
             'scorecam: Sem gradientes. '
             'gradcam: Versão básica.'
    )

    parser.add_argument(
        '--efficientnet_generate_gradcam',
        action='store_true',
        help='Gera mapas de ativação GradCAM após o treinamento.'
    )

    parser.add_argument(
        '--efficientnet_gradcam_samples',
        type=int,
        default=30,
        help='Número de amostras para gerar visualizações GradCAM.'
    )

    parser.add_argument(
        '--efficientnet_gradcam_output_dir',
        type=str,
        default='./mapas_de_ativacao',
        help='Diretório para salvar as visualizações GradCAM.'
    )

    # ========================================================================
    # INFORMAÇÕES E DEBUGGING
    # ========================================================================

    parser.add_argument(
        '--efficientnet_verbose',
        action='store_true',
        help='Mostra informações detalhadas durante a construção do modelo.'
    )

    parser.add_argument(
        '--efficientnet_print_architecture',
        action='store_true',
        help='Imprime a arquitetura completa do modelo após construção.'
    )

    return parser
