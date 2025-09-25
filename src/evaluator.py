"""
Evaluation module for PixelRNN models
Implements evaluation metrics and visualization
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
import pandas as pd

class PixelRNNEvaluator:
    """
    Evaluator class for PixelRNN models
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 model_name: str,
                 results_dir: str = "results",
                 plots_dir: str = "plots"):
        """
        Initialize the evaluator
        
        Args:
            model: The trained model
            model_name: Name of the model
            results_dir: Directory for results
            plots_dir: Directory for plots
        """
        self.model = model
        self.model_name = model_name
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize results
        self.results = {}
        
    def evaluate_model(self, 
                      test_dataset: tf.data.Dataset,
                      compute_nll: bool = True,
                      compute_bits_per_dim: bool = True,
                      generate_samples: bool = True,
                      num_samples: int = 16) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            test_dataset: Test dataset
            compute_nll: Whether to compute negative log-likelihood
            compute_bits_per_dim: Whether to compute bits per dimension
            generate_samples: Whether to generate sample images
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary of evaluation results
        """
        print(f"Evaluating {self.model_name}...")
        
        results = {}
        
        if compute_nll or compute_bits_per_dim:
            nll_results = self.compute_nll_metrics(test_dataset)
            results.update(nll_results)
        
        if generate_samples:
            sample_results = self.generate_and_evaluate_samples(num_samples)
            results.update(sample_results)
        
        # Save results
        self.results = results
        self.save_results()
        
        return results
    
    def compute_nll_metrics(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Compute negative log-likelihood metrics
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of NLL metrics
        """
        print("Computing NLL metrics...")
        
        total_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in test_dataset:
            # Forward pass
            predictions = self.model(x_batch, training=False)
            
            # Compute loss
            loss = self.compute_loss(y_batch, predictions)
            
            total_loss += loss.numpy()
            num_batches += 1
        
        # Average loss
        avg_loss = total_loss / num_batches
        
        # Compute bits per dimension
        num_dimensions = 32 * 32 * 3  # CIFAR-10 dimensions
        # Use log base 2 as in the original PixelRNN paper
        bits_per_dim = avg_loss / (num_dimensions * np.log2(2))  # log2(2) = 1
        
        results = {
            'nll': float(avg_loss),
            'bits_per_dimension': float(bits_per_dim),
            'num_test_batches': num_batches
        }
        
        print(f"NLL: {avg_loss:.4f}")
        print(f"Bits per dimension: {bits_per_dim:.4f}")
        
        return results
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute negative log-likelihood loss
        
        Args:
            y_true: True pixel values
            y_pred: Predicted logits
            
        Returns:
            Loss value
        """
        # Convert true values to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=256, dtype=tf.float32)
        
        # Compute cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true_one_hot,
            logits=y_pred
        )
        
        # Average over spatial dimensions and batch
        loss = tf.reduce_mean(loss)
        
        return loss
    
    def generate_and_evaluate_samples(self, num_samples: int = 16) -> Dict:
        """
        Generate and evaluate sample images
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary of sample evaluation results
        """
        print(f"Generating {num_samples} samples...")
        
        # Generate samples
        samples = self.generate_samples(num_samples)
        
        # Evaluate samples
        sample_metrics = self.evaluate_generated_samples(samples)
        
        # Visualize samples
        self.visualize_samples(samples, num_samples)
        
        results = {
            'generated_samples': samples.tolist(),
            'sample_metrics': sample_metrics
        }
        
        return results
    
    def generate_samples(self, 
                        num_samples: int = 16,
                        temperature: float = 1.0) -> np.ndarray:
        """
        Generate sample images
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Generated samples
        """
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
        
        return generated
    
    def evaluate_generated_samples(self, samples: np.ndarray) -> Dict:
        """
        Evaluate generated samples
        
        Args:
            samples: Generated samples
            
        Returns:
            Dictionary of sample metrics
        """
        # Convert to [0, 1] range
        samples_normalized = samples.astype(np.float32) / 255.0
        
        # Compute basic statistics
        mean_pixel_value = np.mean(samples_normalized)
        std_pixel_value = np.std(samples_normalized)
        
        # Compute per-channel statistics
        channel_means = np.mean(samples_normalized, axis=(0, 1, 2))
        channel_stds = np.std(samples_normalized, axis=(0, 1, 2))
        
        # Compute diversity metrics
        pixel_diversity = np.std(samples_normalized)
        
        metrics = {
            'mean_pixel_value': float(mean_pixel_value),
            'std_pixel_value': float(std_pixel_value),
            'channel_means': channel_means.tolist(),
            'channel_stds': channel_stds.tolist(),
            'pixel_diversity': float(pixel_diversity)
        }
        
        return metrics
    
    def visualize_samples(self, 
                         samples: np.ndarray, 
                         num_samples: int = 16,
                         save_path: Optional[str] = None):
        """
        Visualize generated samples
        
        Args:
            samples: Generated samples
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        # Convert to [0, 1] range
        samples_normalized = samples.astype(np.float32) / 255.0
        
        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, 16)):
            axes[i].imshow(samples_normalized[i])
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.suptitle(f'Generated Samples - {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f"{self.model_name}_generated_samples.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to {save_path}")
        plt.show()
    
    def compare_models(self, 
                      other_results: List[Dict],
                      other_model_names: List[str],
                      save_path: Optional[str] = None):
        """
        Compare multiple models
        
        Args:
            other_results: Results from other models
            other_model_names: Names of other models
            save_path: Path to save the comparison plot
        """
        # Prepare data for comparison
        model_names = [self.model_name] + other_model_names
        all_results = [self.results] + other_results
        
        # Extract metrics
        nll_values = []
        bits_per_dim_values = []
        
        for results in all_results:
            if 'nll' in results:
                nll_values.append(results['nll'])
            else:
                nll_values.append(None)
            
            if 'bits_per_dimension' in results:
                bits_per_dim_values.append(results['bits_per_dimension'])
            else:
                bits_per_dim_values.append(None)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # NLL comparison
        valid_nll = [(name, nll) for name, nll in zip(model_names, nll_values) if nll is not None]
        if valid_nll:
            names, nlls = zip(*valid_nll)
            axes[0].bar(names, nlls)
            axes[0].set_title('Negative Log-Likelihood Comparison')
            axes[0].set_ylabel('NLL')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Bits per dimension comparison
        valid_bits = [(name, bits) for name, bits in zip(model_names, bits_per_dim_values) if bits is not None]
        if valid_bits:
            names, bits = zip(*valid_bits)
            axes[1].bar(names, bits)
            axes[1].set_title('Bits per Dimension Comparison')
            axes[1].set_ylabel('Bits per Dimension')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "model_comparison.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
        plt.show()
    
    def plot_training_history(self, 
                             history: Dict,
                             save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # NLL plot
        if 'train_nll' in history and 'val_nll' in history:
            axes[0, 1].plot(history['train_nll'], label='Train NLL')
            axes[0, 1].plot(history['val_nll'], label='Val NLL')
            axes[0, 1].set_title('Training and Validation NLL')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('NLL')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Bits per dimension plot
        if 'train_bits_per_dim' in history and 'val_bits_per_dim' in history:
            axes[1, 0].plot(history['train_bits_per_dim'], label='Train Bits/Dim')
            axes[1, 0].plot(history['val_bits_per_dim'], label='Val Bits/Dim')
            axes[1, 0].set_title('Training and Validation Bits per Dimension')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Bits per Dimension')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Combined plot
        if 'val_loss' in history and 'val_bits_per_dim' in history:
            axes[1, 1].plot(history['val_loss'], label='Val Loss')
            axes[1, 1].plot(history['val_bits_per_dim'], label='Val Bits/Dim')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle(f'Training History - {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f"{self.model_name}_training_history.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()
    
    def save_results(self):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"{self.model_name}_evaluation_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
    
    def load_results(self, results_path: str):
        """Load evaluation results"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        print(f"Evaluation results loaded from {results_path}")
    
    def create_evaluation_report(self, save_path: Optional[str] = None):
        """
        Create a comprehensive evaluation report
        
        Args:
            save_path: Path to save the report
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, f"{self.model_name}_evaluation_report.txt")
        
        with open(save_path, 'w') as f:
            f.write(f"Evaluation Report for {self.model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model performance
            if 'nll' in self.results:
                f.write(f"Negative Log-Likelihood: {self.results['nll']:.4f}\n")
            
            if 'bits_per_dimension' in self.results:
                f.write(f"Bits per Dimension: {self.results['bits_per_dimension']:.4f}\n")
            
            f.write("\n")
            
            # Sample metrics
            if 'sample_metrics' in self.results:
                f.write("Generated Sample Metrics:\n")
                sample_metrics = self.results['sample_metrics']
                for key, value in sample_metrics.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n")
            
            # Additional information
            f.write("Model Architecture: PixelRNN\n")
            f.write("Dataset: CIFAR-10\n")
            f.write("Image Size: 32x32x3\n")
            f.write("Pixel Values: 256 (discrete)\n")
        
        print(f"Evaluation report saved to {save_path}")

def compare_all_models(model_results: List[Dict], 
                      model_names: List[str],
                      save_path: Optional[str] = None):
    """
    Compare all models and create comprehensive comparison
    
    Args:
        model_results: List of results from all models
        model_names: List of model names
        save_path: Path to save the comparison
    """
    # Create comparison DataFrame
    comparison_data = []
    
    for name, results in zip(model_names, model_results):
        row = {'Model': name}
        
        if 'nll' in results:
            row['NLL'] = results['nll']
        
        if 'bits_per_dimension' in results:
            row['Bits_per_Dimension'] = results['bits_per_dimension']
        
        if 'sample_metrics' in results:
            sample_metrics = results['sample_metrics']
            row['Mean_Pixel_Value'] = sample_metrics.get('mean_pixel_value', None)
            row['Pixel_Diversity'] = sample_metrics.get('pixel_diversity', None)
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison
    if save_path is None:
        save_path = "model_comparison.csv"
    
    df.to_csv(save_path, index=False)
    print(f"Model comparison saved to {save_path}")
    
    # Print comparison
    print("\nModel Comparison:")
    print(df.to_string(index=False))
    
    return df
