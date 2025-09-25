# PixelRNN Implementation

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://tensorflow.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)

Implementation of Pixel Recurrent Neural Networks (PixelRNN) based on the paper by van den Oord et al. (2016). This project implements and compares three generative models: **PixelCNN**, **Row LSTM**, and **Diagonal BiLSTM** on the CIFAR-10 dataset.

## 🎯 Project Overview

This project provides a complete implementation of the PixelRNN paper, featuring three different autoregressive generative models for image generation. Each model offers unique advantages in terms of training efficiency, generation quality, and computational complexity.

### Key Features
- 🔬 **Complete PixelRNN implementation** with all three architectures
- 🎨 **CIFAR-10 image generation** with high-quality samples
- ⚡ **GPU-optimized training** with TensorFlow/Keras
- 📊 **Comprehensive evaluation metrics** and model comparison
- 🎯 **Research-grade implementation** following the original paper

## 📋 Table of Contents

- [Paper Reference](#paper-reference)
- [Models Implemented](#models-implemented)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Key Features of Each Model](#key-features-of-each-model)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Paper Reference

**Pixel Recurrent Neural Networks**  
Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu  
arXiv:1601.06759, 2016  
[Link to Paper](https://arxiv.org/abs/1601.06759)

## Models Implemented

### 1. PixelCNN
- Uses masked convolutions for autoregressive generation
- Mask Type A for first layer, Mask Type B for subsequent layers
- Fully parallelizable during training
- Bounded receptive field

### 2. Row LSTM
- Processes images row by row using LSTM
- Triangular receptive field
- Sequential processing for generation
- Good balance between efficiency and quality

### 3. Diagonal BiLSTM
- Processes images diagonally using bidirectional LSTM
- Global receptive field
- Skewing and unskewing operations
- Best theoretical performance

## Features

- ✅ Complete implementation of all three architectures
- ✅ CIFAR-10 dataset loading and preprocessing
- ✅ Training with negative log-likelihood loss
- ✅ Comprehensive evaluation metrics
- ✅ Sample generation and visualization
- ✅ Model comparison and analysis
- ✅ TensorBoard logging
- ✅ Configurable hyperparameters

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Q3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have a compatible GPU with CUDA support (recommended).

## Usage

### Training All Models
```bash
python main.py --model all --mode train_eval --epochs 100 --batch_size 32
```

### Training Specific Model
```bash
# Train PixelCNN only
python main.py --model pixelcnn --mode train_eval --epochs 50

# Train Row LSTM only
python main.py --model row_lstm --mode train_eval --epochs 50

# Train Diagonal BiLSTM only
python main.py --model diagonal_bilstm --mode train_eval --epochs 50
```

### Evaluation Only
```bash
python main.py --model all --mode eval
```

### Command Line Arguments

- `--model`: Model to train ('pixelcnn', 'row_lstm', 'diagonal_bilstm', 'all')
- `--mode`: Mode ('train', 'eval', 'train_eval')
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)

## Project Structure

```
Q3/
├── main.py                 # Main training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── explanation.md         # Detailed project explanation
├── configs/
│   └── config.py          # Configuration parameters
├── src/
│   ├── data_loader.py     # CIFAR-10 data loading and preprocessing
│   ├── models.py          # Model implementations
│   ├── trainer.py         # Training logic and utilities
│   └── evaluator.py       # Evaluation metrics and visualization
├── models/                # Saved model checkpoints
├── results/               # Evaluation results and metrics
├── plots/                 # Generated plots and visualizations
└── logs/                  # Training logs and TensorBoard files
```

## Configuration

Edit `configs/config.py` to modify:
- Model architecture parameters
- Training hyperparameters
- Evaluation settings
- File paths and directories

## Evaluation Metrics

- **Negative Log-Likelihood (NLL)**: Primary metric for model comparison
- **Bits per Dimension**: NLL normalized by image dimensions
- **Generated Sample Quality**: Visual inspection and statistical analysis
- **Training Efficiency**: Time and memory usage comparison

## Results

The implementation will generate:
- Training history plots
- Generated sample visualizations
- Model comparison charts
- Evaluation reports
- TensorBoard logs

## Key Features of Each Model

### PixelCNN
- **Advantages**: Fast training, fully parallelizable, good baseline performance
- **Disadvantages**: Limited receptive field, may miss long-range dependencies
- **Best for**: Quick experimentation, baseline comparisons

### Row LSTM
- **Advantages**: Good balance of efficiency and quality, captures row-wise dependencies
- **Disadvantages**: Triangular receptive field, sequential generation
- **Best for**: Balanced performance and efficiency

### Diagonal BiLSTM
- **Advantages**: Global receptive field, best theoretical performance
- **Disadvantages**: Most computationally expensive, complex implementation
- **Best for**: Best possible quality, research applications

## Technical Details

### Data Preprocessing
- CIFAR-10 dataset (32×32×3 RGB images)
- Discrete pixel values (0-255) for discrete distribution modeling
- Proper train/validation/test splits

### Training
- Adam optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Gradient clipping for stability

### Generation
- Pixel-by-pixel autoregressive generation
- Temperature scaling for sampling diversity
- Proper masking to prevent information leakage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model size
2. **Training Instability**: Check learning rate and gradient clipping
3. **Poor Generation Quality**: Increase model capacity or training epochs
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Tips

1. Use GPU for training (significantly faster)
2. Adjust batch size based on available memory
3. Monitor training with TensorBoard
4. Use early stopping to prevent overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please cite the original PixelRNN paper if you use this implementation in your research.

## Acknowledgments

- Original PixelRNN paper by van den Oord et al.
- TensorFlow/Keras framework
- CIFAR-10 dataset creators

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.
