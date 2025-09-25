"""
Data loader for CIFAR-10 dataset
Implements proper preprocessing for PixelRNN models
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Optional
import os

class CIFAR10DataLoader:
    """
    Data loader for CIFAR-10 dataset with preprocessing for PixelRNN models
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (32, 32),
                 num_channels: int = 3,
                 validation_split: float = 0.1,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize the data loader
        
        Args:
            image_size: Size of images (height, width)
            num_channels: Number of color channels
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
        """
        self.image_size = image_size
        self.num_channels = num_channels
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """Load CIFAR-10 dataset"""
        print("Loading CIFAR-10 dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        
        print(f"Training data shape: {self.x_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Number of classes: {len(np.unique(self.y_train))}")
        
    def _preprocess_data(self):
        """Preprocess the data for PixelRNN models"""
        print("Preprocessing data...")
        
        # Keep pixel values as discrete integers (0-255) for discrete distribution
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        
        # Flatten labels for easier handling
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()
        
        print(f"Preprocessed training data shape: {self.x_train.shape}")
        print(f"Preprocessed test data shape: {self.x_test.shape}")
        print(f"Pixel value range: [{self.x_train.min()}, {self.x_train.max()}]")
        
    def get_train_val_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split training data into train and validation sets
        
        Returns:
            Tuple of (x_train, y_train, x_val, y_val)
        """
        # Calculate split index
        num_train = int(len(self.x_train) * (1 - self.validation_split))
        
        # Shuffle indices
        indices = np.arange(len(self.x_train))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Split data
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        x_train_split = self.x_train[train_indices]
        y_train_split = self.y_train[train_indices]
        x_val_split = self.x_train[val_indices]
        y_val_split = self.y_train[val_indices]
        
        print(f"Training samples: {len(x_train_split)}")
        print(f"Validation samples: {len(x_val_split)}")
        
        return x_train_split, y_train_split, x_val_split, y_val_split
    
    def create_tf_dataset(self, 
                         x_data: np.ndarray, 
                         y_data: Optional[np.ndarray] = None,
                         is_training: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from numpy arrays
        
        Args:
            x_data: Input images
            y_data: Labels (optional, for supervised learning)
            is_training: Whether this is training data
            
        Returns:
            TensorFlow dataset
        """
        # For PixelRNN, we use the input images as both input and target
        dataset = tf.data.Dataset.from_tensor_slices((x_data, x_data))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(x_data))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_data_generators(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get data generators for training, validation, and test sets
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Get train/val split
        x_train, y_train, x_val, y_val = self.get_train_val_split()
        
        # Create datasets
        train_dataset = self.create_tf_dataset(x_train, y_train, is_training=True)
        val_dataset = self.create_tf_dataset(x_val, y_val, is_training=False)
        test_dataset = self.create_tf_dataset(self.x_test, self.y_test, is_training=False)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_sample_batch(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a sample batch from the dataset
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Tuple of (images, labels) or (images, None) if no labels
        """
        for batch in dataset.take(1):
            if isinstance(batch, tuple):
                return batch[0].numpy(), batch[1].numpy()
            else:
                return batch.numpy(), None
    
    def visualize_samples(self, 
                         images: np.ndarray, 
                         labels: Optional[np.ndarray] = None,
                         num_samples: int = 16,
                         save_path: Optional[str] = None):
        """
        Visualize sample images
        
        Args:
            images: Array of images
            labels: Array of labels (optional)
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Convert back to [0, 1] range for visualization
            img = images[i] / 255.0
            axes[i].imshow(img)
            axes[i].axis('off')
            
            if labels is not None:
                label_idx = int(labels[i]) if hasattr(labels[i], 'item') else int(labels[i])
                axes[i].set_title(class_names[label_idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()

def create_masked_conv2d(inputs, 
                        filters: int, 
                        kernel_size: int, 
                        mask_type: str = 'B',
                        strides: int = 1,
                        padding: str = 'same',
                        name: str = None):
    """
    Create a masked 2D convolution layer
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        kernel_size: Size of the convolution kernel
        mask_type: Type of mask ('A' or 'B')
        strides: Stride size
        padding: Padding type
        name: Layer name
        
    Returns:
        Masked convolution layer
    """
    # Create regular convolution
    conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        name=name
    )(inputs)
    
    # Apply mask
    if mask_type == 'A':
        # Mask A: Center pixel and future pixels are masked
        mask = create_mask_A(kernel_size, filters)
    elif mask_type == 'B':
        # Mask B: Only future pixels are masked
        mask = create_mask_B(kernel_size, filters)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    # Apply mask to weights
    conv.kernel.assign(conv.kernel * mask)
    
    return conv

def create_mask_A(kernel_size: int, filters: int) -> tf.Tensor:
    """
    Create mask A for the first layer
    
    Args:
        kernel_size: Size of the kernel
        filters: Number of filters
        
    Returns:
        Mask tensor
    """
    mask = tf.ones((kernel_size, kernel_size, 3, filters))
    
    # Mask the center pixel
    center = kernel_size // 2
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[center, center, 0, i] for i in range(filters)],
        [0.0] * filters
    )
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[center, center, 1, i] for i in range(filters)],
        [0.0] * filters
    )
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[center, center, 2, i] for i in range(filters)],
        [0.0] * filters
    )
    
    # Mask future pixels (right and below center)
    for i in range(center, kernel_size):
        for j in range(center, kernel_size):
            if i > center or j > center:
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    [[i, j, k, l] for k in range(3) for l in range(filters)],
                    [0.0] * (3 * filters)
                )
    
    return mask

def create_mask_B(kernel_size: int, filters: int) -> tf.Tensor:
    """
    Create mask B for subsequent layers
    
    Args:
        kernel_size: Size of the kernel
        filters: Number of filters
        
    Returns:
        Mask tensor
    """
    mask = tf.ones((kernel_size, kernel_size, filters, filters))
    
    # Mask future pixels (right and below center)
    center = kernel_size // 2
    for i in range(center, kernel_size):
        for j in range(center, kernel_size):
            if i > center or j > center:
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    [[i, j, k, l] for k in range(filters) for l in range(filters)],
                    [0.0] * (filters * filters)
                )
    
    return mask
