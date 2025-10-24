<div align="center">

# 🦟 Mosquito Audio Classification Models

**A comprehensive deep learning library for acoustic bioacoustics classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.4+](https://img.shields.io/badge/TensorFlow-2.4%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Supported-brightgreen.svg)](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Open Source](https://img.shields.io/badge/open%20source-yes-green.svg)]()

[Installation](#-installation) • [Quick Start](#-quick-start) • [Models](#-implemented-models) • [Documentation](#-documentation) • [Citation](#-citation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Implemented Models](#-implemented-models)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

This library provides state-of-the-art deep neural network implementations for mosquito audio classification and bioacoustics analysis. Built on TensorFlow/Keras, it offers a collection of modern architectures optimized for acoustic pattern recognition, supporting both research and production environments.

The models leverage advanced signal processing techniques and transformer-based architectures to achieve high accuracy in species classification from audio recordings. All implementations support GPU acceleration and include comprehensive evaluation tools.

![Audio Spectrograms](Layout/AudioSegments.png)

**Research Publication:** [View original publication on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1746809424004002)

---

## ✨ Key Features

- **🎯 Multiple Architectures**: Nine state-of-the-art models including transformers, CNNs, RNNs, and hybrid approaches
- **⚡ GPU Acceleration**: Full CUDA support for high-performance training and inference
- **🔊 Audio Processing Pipeline**: Built-in spectrogram generation and audio preprocessing
- **📊 Evaluation Framework**: Comprehensive metrics, confusion matrices, and visualization tools
- **🔄 Cross-Validation**: Built-in k-fold cross-validation support
- **🛠️ Flexible Configuration**: Extensive hyperparameter customization via CLI
- **📦 Easy Integration**: Simple API for quick prototyping and deployment
- **🐳 Docker Support**: Containerized execution for reproducibility

## 🏗️ Implemented Models

The library includes the following architectures, each optimized for acoustic classification tasks:

| Model | Type | Description | Key Features |
|-------|------|-------------|--------------|
| **AST** | Transformer | Audio Spectrogram Transformer | Self-attention on spectrogram patches |
| **Conformer** | Hybrid | Convolution-augmented Transformer | Combines CNN and transformer blocks |
| **ConvNetX** | CNN | Modern ConvNet architecture | Efficient convolutional design |
| **EfficientNet** | CNN | Compound scaling CNN | Optimized depth, width, and resolution |
| **LSTM** | RNN | Long Short-Term Memory | Sequential temporal modeling |
| **MLP** | Feedforward | Multi-Layer Perceptron | Fully connected baseline |
| **MobileNet** | CNN | Lightweight mobile architecture | Depthwise separable convolutions |
| **ResidualModel** | CNN | Residual Neural Network | Skip connections for deep training |
| **Wav2Vec2** | Transformer | Self-supervised audio encoder | Contrastive learning on raw waveforms |

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA 11.0+ for GPU support

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/kayua/Audio-Classification-Library
cd Mosquitoes-Classification-Models

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package
pip install .
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv ~/Python3venv/mosquito-classification
source ~/Python3venv/mosquito-classification/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package
pip install .
```

### Docker Installation

```bash
# Build Docker image
docker build -t mosquito-classification .

# Run container
docker run --gpus all -it mosquito-classification
```

---

## 🚀 Quick Start

### Basic Usage

```python
## 🚀 Basic Usage

### Quick Start
```python
from Engine.Models.AST import AudioSpectrogramTransformer
from Engine.Models.Conformer import Conformer
from Engine.Models.Wav2Vec2 import AudioWav2Vec2
from Engine.Models.LSTM import AudioLSTM
from Engine.Models.MLP import DenseModel
from Engine.Models.ResidualModel import ResidualModel

# Initialize the main training pipeline
main = Main()
main.__start__()

# Define the models you want to train
available_models = [
    AudioSpectrogramTransformer,
    Conformer,
    AudioWav2Vec2,
    AudioLSTM,
    DenseModel,
    ResidualModel
]

# Execute training and evaluation
main.__exec__(available_models, "Results")
```

### Training a Single Model
```python
# Initialize
main = Main()
main.__start__()

# Train only Wav2Vec2
available_models = [AudioWav2Vec2]

# Run training pipeline
main.__exec__(available_models, "Results")
```

### Output

After training, the following artifacts will be generated in the `Results/` directory:

- **Metrics**: Comparative performance metrics across models
- **Confusion Matrices**: Visual representation of classification results
- **Loss Curves**: Training and validation loss over epochs
- **ROC Curves**: Receiver Operating Characteristic curves
- **Results.pdf**: Comprehensive report with all visualizations

### Prerequisites
```bash
# Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Configuration

The training pipeline uses command-line arguments for configuration. Customize your training by passing arguments when initializing:
```python
# Example with custom arguments
python main.py --epochs 100 --batch_size 32 --learning_rate 0.001
```
```

### Training from Command Line

```bash
# Train AST model
python train.py \
    --model_name ast \
    --dataset_directory ./data \
    --number_epochs 100 \
    --batch_size 32 \
    --number_splits 5 \
    --output_directory ./results
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py \
    --model_path ./results/best_model.h5 \
    --test_data ./data/test \
    --output_directory ./evaluation
```

---

## 🖥️ System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Any modern x86_64 | Multi-core (Intel i5/Ryzen 5+) |
| **RAM** | 4 GB | 8 GB or more |
| **Storage** | 10 GB free space | SSD with 20 GB free |
| **GPU** | None (CPU only) | NVIDIA GPU with CUDA 11+ |

### Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Ubuntu 22.04+ | Linux-based distributions |
| **Python** | 3.8.10+ | Virtual environment recommended |
| **TensorFlow** | 2.4.1+ | GPU version for acceleration |
| **CUDA** | 11.0+ | Optional, for GPU support |
| **Docker** | 27.2.1+ | Optional, for containerization |

> **Note:** GPU support is optional but significantly improves training speed. CPU-only execution is supported but may be slower for large datasets.


## 📚 API Reference

### Common Parameters

All models share a common set of training parameters:

```python
# General parameters
--dataset_directory          # Path to dataset
--number_epochs              # Training epochs (default: 100)
--batch_size                 # Batch size (default: 32)
--number_splits              # K-fold splits (default: 5)
--loss                       # Loss function (categorical_crossentropy)
--sample_rate                # Audio sample rate (Hz)
--number_classes             # Number of output classes
--output_directory           # Results directory
```

### Model-Specific Parameters

#### Audio Spectrogram Transformer (AST)

```python
--ast_projection_dimension        # Embedding dimension (default: 256)
--ast_number_heads                # Attention heads (default: 8)
--ast_number_blocks               # Transformer blocks (default: 6)
--ast_patch_size                  # Spectrogram patch size
--ast_hop_length                  # STFT hop length
--ast_size_fft                    # FFT window size
--ast_dropout                     # Dropout rate
--ast_optimizer_function          # Optimizer (adam, sgd, rmsprop)
```

#### Conformer

```python
--conformer_embedding_dimension   # Embedding dimension
--conformer_number_heads          # Attention heads
--conformer_number_conformer_blocks  # Conformer blocks
--conformer_kernel_size           # Convolution kernel size
--conformer_dropout_rate          # Dropout rate
--conformer_max_length            # Max sequence length
```

#### LSTM

```python
--lstm_list_lstm_cells            # LSTM units per layer [128, 64, 32]
--lstm_dropout_rate               # Dropout rate
--lstm_recurrent_activation       # Recurrent activation (tanh, sigmoid)
--lstm_optimizer_function         # Optimizer
```

#### Residual Network

```python
--residual_number_layers          # Number of residual blocks
--residual_filters_per_block      # Filters per block
--residual_dropout_rate           # Dropout rate
--residual_size_pooling           # Pooling layer size
--residual_size_convolutional_filters  # Conv filter size
```

#### Wav2Vec 2.0

```python
--wav_to_vec_number_heads         # Attention heads
--wav_to_vec_context_dimension    # Context dimension
--wav_to_vec_projection_mlp_dimension  # MLP projection dimension
--wav_to_vec_quantization_bits    # Quantization bits
--wav_to_vec_list_filters_encoder # Encoder filter sizes
```

For complete API documentation, see the [full parameter reference](docs/API.md).


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Mosquito Classification Models

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Research community for innovative model architectures
- Contributors and users for feedback and improvements
- [ScienceDirect](https://www.sciencedirect.com) for publication platform

---

## 📞 Support

- **Documentation**: [Read the Docs](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/kayua/Mosquitoes-Classification-Models/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kayua/Mosquitoes-Classification-Models/discussions)
- **Email**: support@example.com

---

## 🔗 Related Projects

- [TensorFlow Audio](https://www.tensorflow.org/io/tutorials/audio)
- [Hugging Face Audio Transformers](https://huggingface.co/docs/transformers/tasks/audio_classification)
- [PyTorch Audio](https://pytorch.org/audio/stable/index.html)
- [Librosa](https://librosa.org/)

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

Made with ❤️ by the Mosquito Classification Team

[Website](https://example.com) • [Documentation](https://docs.example.com) • [Blog](https://blog.example.com)

</div>
