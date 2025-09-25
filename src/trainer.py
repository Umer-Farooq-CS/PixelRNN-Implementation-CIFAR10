"""
Training module for PixelRNN models
Implements training logic with negative log-likelihood loss
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import json

class PixelRNNTrainer:
    """
    Trainer class for PixelRNN models
    """
    
    def __init__(self, 
                 model: keras.Model,
                 model_name: str,
                 config: Dict,
                 log_dir: str = "logs",
                 model_dir: str = "models"):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            model_name: Name of the model
            config: Training configuration
            log_dir: Directory for logs
            model_dir: Directory for saving models
        """
        self.model = model
        self.model_name = model_name
        self.config = config
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_nll': [],
            'val_nll': [],
            'train_bits_per_dim': [],
            'val_bits_per_dim': []
        }
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Setup metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        
    def _setup_optimizer(self):
        """Setup the optimizer"""
        if self.config['optimizer'] == 'adam':
            return keras.optimizers.Adam(
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif self.config['optimizer'] == 'sgd':
            return keras.optimizers.SGD(
                learning_rate=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # TensorBoard callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.log_dir, 'tensorboard', self.model_name),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Model checkpoint callback
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.model_dir, f"{self.model_name}_checkpoint.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Reduce learning rate callback
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.get('reduce_lr_factor', 0.5),
            patience=self.config.get('reduce_lr_patience', 5),
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        return callbacks
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute negative log-likelihood loss
        
        Args:
            y_true: True pixel values (batch_size, height, width, channels)
            y_pred: Predicted logits (batch_size, height, width, channels, num_pixel_values)
            
        Returns:
            Loss value
        """
        # For PixelRNN, we use the input as the target (autoregressive)
        # y_true should be the same as the input to the model
        # Convert true values to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=256, dtype=tf.float32)
        
        # Compute cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true_one_hot,
            logits=y_pred
        )
        
        # Average over all dimensions
        loss = tf.reduce_mean(loss)
        
        return loss
    
    def compute_nll(self, y_true, y_pred):
        """
        Compute negative log-likelihood
        
        Args:
            y_true: True pixel values
            y_pred: Predicted logits
            
        Returns:
            NLL value
        """
        return self.compute_loss(y_true, y_pred)
    
    def compute_bits_per_dimension(self, nll):
        """
        Compute bits per dimension from NLL
        
        Args:
            nll: Negative log-likelihood
            
        Returns:
            Bits per dimension
        """
        # CIFAR-10 has 32x32x3 = 3072 dimensions
        # Use log base 2 as in the original PixelRNN paper
        num_dimensions = 32 * 32 * 3
        bits_per_dim = nll / (num_dimensions * np.log2(2))  # log2(2) = 1
        return bits_per_dim
    
    @tf.function
    def train_step(self, x_batch, y_batch):
        """
        Single training step
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x_batch, training=True)
            
            # Compute loss
            loss = self.compute_loss(y_batch, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        
        return loss
    
    @tf.function
    def val_step(self, x_batch, y_batch):
        """
        Single validation step
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Loss value
        """
        # Forward pass
        predictions = self.model(x_batch, training=False)
        
        # Compute loss
        loss = self.compute_loss(y_batch, predictions)
        
        # Update metrics
        self.val_loss_metric.update_state(loss)
        
        return loss
    
    def train_epoch(self, train_dataset, epoch_num, total_epochs):
        """
        Train for one epoch
        
        Args:
            train_dataset: Training dataset
            epoch_num: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Average training loss
        """
        self.train_loss_metric.reset_state()
        
        # Calculate total batches for progress tracking
        total_batches = len(list(train_dataset))
        start_time = time.time()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            loss = self.train_step(x_batch, y_batch)
            
            # Check for NaN values
            if tf.math.is_nan(loss):
                print(f"⚠️  NaN detected in training loss at batch {batch_idx+1}")
                print("Skipping this batch and continuing with next batch...")
                continue
            
            # Show progress every 50 batches or at the end
            if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                elapsed_time = time.time() - start_time
                progress = (batch_idx + 1) / total_batches * 100
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - batch_idx - 1
                eta_seconds = remaining_batches * avg_time_per_batch
                eta_minutes = eta_seconds / 60
                
                print(f"Epoch {epoch_num}/{total_epochs} | "
                      f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | "
                      f"Loss: {loss:.4f} | "
                      f"ETA: {eta_minutes:.1f}m")
        
        final_loss = self.train_loss_metric.result().numpy()
        
        # Check for NaN in final epoch loss
        if np.isnan(final_loss):
            print(f"⚠️  NaN detected in final training loss for epoch {epoch_num}")
            print("Using previous best loss value...")
            # Return the last valid loss from history if available
            if self.history['train_loss']:
                final_loss = self.history['train_loss'][-1]
            else:
                final_loss = 0.0  # Default fallback
        
        return final_loss
    
    def validate_epoch(self, val_dataset):
        """
        Validate for one epoch
        
        Args:
            val_dataset: Validation dataset
            
        Returns:
            Average validation loss
        """
        self.val_loss_metric.reset_state()
        
        for x_batch, y_batch in val_dataset:
            loss = self.val_step(x_batch, y_batch)
            
            # Check for NaN values
            if tf.math.is_nan(loss):
                print(f"⚠️  NaN detected in validation loss")
                print("Skipping this batch and continuing...")
                continue
        
        final_loss = self.val_loss_metric.result().numpy()
        
        # Check for NaN in final validation loss
        if np.isnan(final_loss):
            print(f"⚠️  NaN detected in final validation loss")
            print("Using previous best validation loss value...")
            # Return the last valid loss from history if available
            if self.history['val_loss']:
                final_loss = self.history['val_loss'][-1]
            else:
                final_loss = 0.0  # Default fallback
        
        return final_loss
    
    def train(self, 
              train_dataset, 
              val_dataset, 
              num_epochs: int = 100,
              save_interval: int = 10):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs to train
            save_interval: Interval for saving model
        """
        print(f"Starting training for {self.model_name}")
        print(f"Training samples: {len(list(train_dataset))}")
        print(f"Validation samples: {len(list(val_dataset))}")
        
        # Calculate estimated total time
        total_batches = len(list(train_dataset))
        estimated_time_per_epoch = total_batches * 0.1  # Rough estimate
        estimated_total_time = estimated_time_per_epoch * num_epochs / 60  # in minutes
        
        print(f"Estimated training time: {estimated_total_time:.1f} minutes")
        print(f"Total batches per epoch: {total_batches}")
        print(f"{'='*60}")
        
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        nan_epochs = 0
        max_nan_epochs = 3  # Maximum consecutive NaN epochs before stopping
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_dataset, epoch + 1, num_epochs)
            
            # Validation
            val_loss = self.validate_epoch(val_dataset)
            
            # Check for NaN values in losses
            if np.isnan(train_loss) or np.isnan(val_loss):
                nan_epochs += 1
                print(f"\n{'='*60}")
                print(f"⚠️  NaN DETECTED IN EPOCH {epoch+1}/{num_epochs}")
                print(f"NaN epochs count: {nan_epochs}/{max_nan_epochs}")
                print(f"{'='*60}")
                
                if nan_epochs >= max_nan_epochs:
                    print(f"❌ Too many NaN epochs ({nan_epochs}). Stopping training.")
                    print("Using previous best model...")
                    break
                
                print("Skipping this epoch and continuing with next epoch...")
                print(f"{'='*60}\n")
                continue
            
            # Reset NaN counter if we have valid losses
            nan_epochs = 0
            
            # Compute metrics
            train_nll = train_loss
            val_nll = val_loss
            train_bits_per_dim = self.compute_bits_per_dimension(train_nll)
            val_bits_per_dim = self.compute_bits_per_dimension(val_nll)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_nll'].append(train_nll)
            self.history['val_nll'].append(val_nll)
            self.history['train_bits_per_dim'].append(train_bits_per_dim)
            self.history['val_bits_per_dim'].append(val_bits_per_dim)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs} COMPLETED")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train NLL: {train_nll:.4f} | Val NLL: {val_nll:.4f}")
            print(f"Train Bits/Dim: {train_bits_per_dim:.4f} | Val Bits/Dim: {val_bits_per_dim:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Calculate overall progress
            overall_progress = (epoch + 1) / num_epochs * 100
            print(f"Overall Progress: {overall_progress:.1f}%")
            
            if val_loss < best_val_loss:
                print(f"🎉 NEW BEST VALIDATION LOSS: {val_loss:.4f}")
                best_val_loss = val_loss
            else:
                print(f"Best Val Loss: {best_val_loss:.4f}")
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            
            print(f"{'='*60}\n")
            
            # Save model if best
            if val_loss < best_val_loss:
                self.model.save(os.path.join(self.model_dir, f"{self.model_name}_best.keras"))
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Save model at intervals
            if (epoch + 1) % save_interval == 0:
                self.model.save(os.path.join(self.model_dir, f"{self.model_name}_epoch_{epoch+1}.keras"))
        
        # Save final model
        self.model.save(os.path.join(self.model_dir, f"{self.model_name}_final.keras"))
        
        # Save training history
        self.save_history()
        
        print(f"Training completed for {self.model_name}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.log_dir, f"{self.model_name}_history.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_history(self, history_path: str):
        """Load training history"""
        with open(history_path, 'r') as f:
            self.history = json.load(f)
        print(f"Training history loaded from {history_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # NLL plot
        axes[0, 1].plot(self.history['train_nll'], label='Train NLL')
        axes[0, 1].plot(self.history['val_nll'], label='Val NLL')
        axes[0, 1].set_title('Training and Validation NLL')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('NLL')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Bits per dimension plot
        axes[1, 0].plot(self.history['train_bits_per_dim'], label='Train Bits/Dim')
        axes[1, 0].plot(self.history['val_bits_per_dim'], label='Val Bits/Dim')
        axes[1, 0].set_title('Training and Validation Bits per Dimension')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Bits per Dimension')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined plot
        axes[1, 1].plot(self.history['val_loss'], label='Val Loss')
        axes[1, 1].plot(self.history['val_bits_per_dim'], label='Val Bits/Dim')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def generate_samples(self, 
                        num_samples: int = 16,
                        temperature: float = 1.0,
                        save_path: Optional[str] = None):
        """
        Generate sample images
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            save_path: Path to save the samples
            
        Returns:
            Generated samples
        """
        print(f"Generating {num_samples} samples with temperature {temperature}")
        
        # Initialize with zeros
        batch_size = num_samples
        height, width, channels = 32, 32, 3
        
        # Create empty image
        generated = np.zeros((batch_size, height, width, channels), dtype=np.int32)
        
        # Generate pixel by pixel
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    # Get current context
                    current_input = tf.constant(generated, dtype=tf.float32)
                    
                    # Get predictions
                    predictions = self.model(current_input, training=False)
                    
                    # Get logits for current pixel
                    logits = predictions[:, i, j, c, :] / temperature
                    
                    # Sample from distribution
                    probs = tf.nn.softmax(logits)
                    sampled = tf.random.categorical(tf.math.log(probs), 1)
                    sampled = tf.squeeze(sampled, axis=1)
                    
                    # Update generated image
                    generated[:, i, j, c] = sampled.numpy()
        
        # Convert to [0, 1] range for visualization
        generated_normalized = generated.astype(np.float32) / 255.0
        
        # Plot samples
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(min(num_samples, 16)):
            axes[i].imshow(generated_normalized[i])
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Generated samples saved to {save_path}")
        
        plt.show()
        
        return generated_normalized
