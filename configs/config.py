"""
Configuration file for PixelRNN implementation
Based on the Pixel Recurrent Neural Networks paper by van den Oord et al. (2016)
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    
    # Common model parameters
    image_size: Tuple[int, int] = (32, 32)  # CIFAR-10 size
    num_channels: int = 3  # RGB
    num_pixel_values: int = 256  # Discrete pixel values (0-255)
    
    # PixelCNN parameters
    pixelcnn_layers: int = 12
    pixelcnn_filters: int = 128
    pixelcnn_kernel_size: int = 7  # First layer
    pixelcnn_kernel_size_later: int = 3  # Later layers
    
    # Row LSTM parameters
    row_lstm_layers: int = 12
    row_lstm_hidden_size: int = 128
    row_lstm_kernel_size: int = 3  # k x 1 kernel
    
    # Diagonal BiLSTM parameters
    diagonal_bilstm_layers: int = 12
    diagonal_bilstm_hidden_size: int = 128
    diagonal_bilstm_kernel_size: Tuple[int, int] = (2, 1)  # 2 x 1 kernel
    
    # Residual connections
    use_residual: bool = True
    residual_features: int = 256  # 2h features for residual connections

@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Dataset
    dataset: str = "cifar10"
    batch_size: int = 32
    num_epochs: int = 100
    
    # Optimization
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    
    # Training settings
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 10
    tensorboard_log_dir: str = "logs/tensorboard"
    model_save_dir: str = "models"
    results_dir: str = "results"
    plots_dir: str = "plots"

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Metrics
    compute_nll: bool = True  # Negative log-likelihood
    compute_bits_per_dim: bool = True  # Bits per dimension
    generate_samples: bool = True  # Generate sample images
    num_generated_samples: int = 16
    
    # Visualization
    plot_training_history: bool = True
    plot_generated_samples: bool = True
    plot_model_comparison: bool = True

# Global configuration instances
model_config = ModelConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
