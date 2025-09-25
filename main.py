"""
Main script for PixelRNN implementation
Trains and evaluates PixelCNN, Row LSTM, and Diagonal BiLSTM models
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from configs.config import model_config, training_config, evaluation_config
from src.data_loader import CIFAR10DataLoader
from src.models import PixelCNN, RowLSTM, DiagonalBiLSTM, MaskedConv2D, ResidualBlock
from src.trainer import PixelRNNTrainer
from src.evaluator import PixelRNNEvaluator, compare_all_models

def setup_gpu():
    """Setup GPU configuration"""
    print("Setting up GPU...")
    
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
            print(f"GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

def create_model(model_type: str, config: dict) -> tf.keras.Model:
    """
    Create a model based on the specified type
    
    Args:
        model_type: Type of model ('pixelcnn', 'row_lstm', 'diagonal_bilstm')
        config: Model configuration
        
    Returns:
        The created model
    """
    print(f"Creating {model_type} model...")
    
    if model_type == 'pixelcnn':
        model = PixelCNN(
            image_size=config['image_size'],
            num_channels=config['num_channels'],
            num_pixel_values=config['num_pixel_values'],
            num_layers=config['pixelcnn_layers'],
            filters=config['pixelcnn_filters'],
            kernel_size_first=config['pixelcnn_kernel_size'],
            kernel_size_later=config['pixelcnn_kernel_size_later'],
            use_residual=config['use_residual'],
            residual_features=config['residual_features']
        )
    elif model_type == 'row_lstm':
        model = RowLSTM(
            image_size=config['image_size'],
            num_channels=config['num_channels'],
            num_pixel_values=config['num_pixel_values'],
            num_layers=config['row_lstm_layers'],
            hidden_size=config['row_lstm_hidden_size'],
            kernel_size=config['row_lstm_kernel_size'],
            use_residual=config['use_residual'],
            residual_features=config['residual_features']
        )
    elif model_type == 'diagonal_bilstm':
        model = DiagonalBiLSTM(
            image_size=config['image_size'],
            num_channels=config['num_channels'],
            num_pixel_values=config['num_pixel_values'],
            num_layers=config['diagonal_bilstm_layers'],
            hidden_size=config['diagonal_bilstm_hidden_size'],
            kernel_size=config['diagonal_bilstm_kernel_size'],
            use_residual=config['use_residual'],
            residual_features=config['residual_features']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def train_model(model_type: str, 
                model: tf.keras.Model,
                train_dataset: tf.data.Dataset,
                val_dataset: tf.data.Dataset,
                config: dict) -> PixelRNNTrainer:
    """
    Train a model
    
    Args:
        model_type: Type of model
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        
    Returns:
        The trainer object
    """
    print(f"Training {model_type} model...")
    
    # Create trainer
    trainer = PixelRNNTrainer(
        model=model,
        model_name=model_type,
        config=config,
        log_dir=training_config.tensorboard_log_dir,
        model_dir=training_config.model_save_dir
    )
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config['num_epochs'],
        save_interval=config['save_interval']
    )
    
    return trainer

def evaluate_model(model_type: str,
                  model: tf.keras.Model,
                  test_dataset: tf.data.Dataset,
                  config: dict) -> PixelRNNEvaluator:
    """
    Evaluate a model
    
    Args:
        model_type: Type of model
        model: The trained model
        test_dataset: Test dataset
        config: Evaluation configuration
        
    Returns:
        The evaluator object
    """
    print(f"Evaluating {model_type} model...")
    
    # Create evaluator
    evaluator = PixelRNNEvaluator(
        model=model,
        model_name=model_type,
        results_dir=training_config.results_dir,
        plots_dir=training_config.plots_dir
    )
    
    # Evaluate the model
    results = evaluator.evaluate_model(
        test_dataset=test_dataset,
        compute_nll=config['compute_nll'],
        compute_bits_per_dim=config['compute_bits_per_dim'],
        generate_samples=config['generate_samples'],
        num_samples=config['num_generated_samples']
    )
    
    return evaluator

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PixelRNN Implementation')
    parser.add_argument('--model', type=str, default='all',
                       choices=['pixelcnn', 'row_lstm', 'diagonal_bilstm', 'all'],
                       help='Model to train and evaluate')
    parser.add_argument('--mode', type=str, default='train_eval',
                       choices=['train', 'eval', 'train_eval'],
                       help='Mode: train, eval, or train_eval')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(
        image_size=model_config.image_size,
        num_channels=model_config.num_channels,
        validation_split=training_config.validation_split,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    train_dataset, val_dataset, test_dataset = data_loader.get_data_generators()
    
    # Visualize sample data
    print("Visualizing sample data...")
    sample_images, sample_labels = data_loader.get_sample_batch(train_dataset)
    data_loader.visualize_samples(
        sample_images[:16], 
        None,  # Skip labels for now
        num_samples=16,
        save_path=os.path.join(training_config.plots_dir, "cifar10_samples.png")
    )
    
    # Update configs with command line arguments
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    
    # Display configuration settings
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Optimizer: {training_config.optimizer}")
    print(f"Validation Split: {training_config.validation_split}")
    print(f"Early Stopping Patience: {training_config.early_stopping_patience}")
    print(f"Reduce LR Patience: {training_config.reduce_lr_patience}")
    print(f"Reduce LR Factor: {training_config.reduce_lr_factor}")
    print(f"GPU Available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")
    print(f"{'='*80}\n")
    
    # Convert configs to dictionaries
    model_config_dict = {
        'image_size': model_config.image_size,
        'num_channels': model_config.num_channels,
        'num_pixel_values': model_config.num_pixel_values,
        'pixelcnn_layers': model_config.pixelcnn_layers,
        'pixelcnn_filters': model_config.pixelcnn_filters,
        'pixelcnn_kernel_size': model_config.pixelcnn_kernel_size,
        'pixelcnn_kernel_size_later': model_config.pixelcnn_kernel_size_later,
        'row_lstm_layers': model_config.row_lstm_layers,
        'row_lstm_hidden_size': model_config.row_lstm_hidden_size,
        'row_lstm_kernel_size': model_config.row_lstm_kernel_size,
        'diagonal_bilstm_layers': model_config.diagonal_bilstm_layers,
        'diagonal_bilstm_hidden_size': model_config.diagonal_bilstm_hidden_size,
        'diagonal_bilstm_kernel_size': model_config.diagonal_bilstm_kernel_size,
        'use_residual': model_config.use_residual,
        'residual_features': model_config.residual_features
    }
    
    training_config_dict = {
        'num_epochs': training_config.num_epochs,
        'batch_size': training_config.batch_size,
        'learning_rate': training_config.learning_rate,
        'optimizer': training_config.optimizer,
        'weight_decay': training_config.weight_decay,
        'validation_split': training_config.validation_split,
        'early_stopping_patience': training_config.early_stopping_patience,
        'reduce_lr_patience': training_config.reduce_lr_patience,
        'reduce_lr_factor': training_config.reduce_lr_factor,
        'log_interval': training_config.log_interval,
        'save_interval': training_config.save_interval
    }
    
    evaluation_config_dict = {
        'compute_nll': evaluation_config.compute_nll,
        'compute_bits_per_dim': evaluation_config.compute_bits_per_dim,
        'generate_samples': evaluation_config.generate_samples,
        'num_generated_samples': evaluation_config.num_generated_samples
    }
    
    # Determine which models to train/evaluate
    if args.model == 'all':
        models_to_process = ['pixelcnn', 'row_lstm', 'diagonal_bilstm']
    else:
        models_to_process = [args.model]
    
    # Store results for comparison
    all_results = []
    all_model_names = []
    
    # Process each model
    for model_type in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Create or load model
        if args.mode in ['eval'] and os.path.exists(os.path.join(training_config.model_save_dir, f"{model_type}_best.keras")):
            print(f"Loading trained {model_type} model...")
            try:
                # Load the trained model
                model = tf.keras.models.load_model(
                    os.path.join(training_config.model_save_dir, f"{model_type}_best.keras"),
                    custom_objects={
                        'PixelCNN': PixelCNN,
                        'RowLSTM': RowLSTM,
                        'DiagonalBiLSTM': DiagonalBiLSTM,
                        'MaskedConv2D': MaskedConv2D,
                        'ResidualBlock': ResidualBlock
                    }
                )
                print(f"Successfully loaded trained {model_type} model!")
            except Exception as e:
                print(f"Failed to load trained model: {e}")
                print("Creating new model instead...")
                model = create_model(model_type, model_config_dict)
                # Build the model with a sample input
                sample_input = tf.random.normal((1, 32, 32, 3))
                _ = model(sample_input)
        else:
            # Create new model
            model = create_model(model_type, model_config_dict)
            # Build the model with a sample input
            sample_input = tf.random.normal((1, 32, 32, 3))
            _ = model(sample_input)
        
        # Print model summary
        print(f"\n{model_type.upper()} Model Summary:")
        model.summary()
        
        # Train model if requested
        trainer = None
        if args.mode in ['train', 'train_eval']:
            trainer = train_model(
                model_type=model_type,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=training_config_dict
            )
            
            # Plot training history
            if trainer.history:
                trainer.plot_training_history(
                    save_path=os.path.join(training_config.plots_dir, f"{model_type}_training_history.png")
                )
        
        # Evaluate model if requested
        evaluator = None
        if args.mode in ['eval', 'train_eval']:
            evaluator = evaluate_model(
                model_type=model_type,
                model=model,
                test_dataset=test_dataset,
                config=evaluation_config_dict
            )
            
            # Store results for comparison
            all_results.append(evaluator.results)
            all_model_names.append(model_type)
    
    # Compare all models if multiple models were processed
    if len(models_to_process) > 1 and args.mode in ['eval', 'train_eval']:
        print(f"\n{'='*60}")
        print("COMPARING ALL MODELS")
        print(f"{'='*60}")
        
        # Create comparison
        comparison_df = compare_all_models(
            model_results=all_results,
            model_names=all_model_names,
            save_path=os.path.join(training_config.results_dir, "model_comparison.csv")
        )
        
        # Create comparison plot
        if len(all_results) > 1:
            # Use the first evaluator to create comparison plot
            first_evaluator = None
            for model_type in models_to_process:
                if model_type in all_model_names:
                    first_evaluator = PixelRNNEvaluator(
                        model=None,  # We don't need the model for plotting
                        model_name=model_type,
                        results_dir=training_config.results_dir,
                        plots_dir=training_config.plots_dir
                    )
                    break
            
            if first_evaluator:
                first_evaluator.compare_models(
                    other_results=all_results[1:],
                    other_model_names=all_model_names[1:],
                    save_path=os.path.join(training_config.plots_dir, "model_comparison.png")
                )
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {training_config.results_dir}")
    print(f"Plots saved to: {training_config.plots_dir}")
    print(f"Models saved to: {training_config.model_save_dir}")
    print(f"Logs saved to: {training_config.tensorboard_log_dir}")

if __name__ == "__main__":
    main()
