#!/usr/bin/env python3
"""
Generate comprehensive final report for PixelRNN models
Creates training history plots, detailed comparison tables, and summary reports
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_evaluation_results(results_dir: str) -> Dict[str, Dict]:
    """Load evaluation results for all models"""
    results = {}
    
    # Find all evaluation result files
    for filename in os.listdir(results_dir):
        if '_evaluation_' in filename and filename.endswith('.json'):
            # Extract model name from filename
            model_name = filename.split('_evaluation_')[0]
            
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
    
    return results

def create_comprehensive_comparison_table(results: Dict[str, Dict], save_path: str):
    """Create a comprehensive comparison table"""
    
    # Prepare data for comparison
    comparison_data = []
    
    for model_name, result in results.items():
        row = {
            'Model': model_name.upper(),
            'NLL': result.get('nll', 'N/A'),
            'Bits_per_Dimension': result.get('bits_per_dimension', 'N/A'),
            'Test_Batches': result.get('num_test_batches', 'N/A')
        }
        
        # Add sample metrics if available
        if 'sample_metrics' in result:
            sample_metrics = result['sample_metrics']
            row.update({
                'Mean_Pixel_Value': sample_metrics.get('mean_pixel_value', 'N/A'),
                'Pixel_Diversity': sample_metrics.get('pixel_diversity', 'N/A'),
                'Channel_Mean_R': sample_metrics.get('channel_means', [0, 0, 0])[0],
                'Channel_Mean_G': sample_metrics.get('channel_means', [0, 0, 0])[1],
                'Channel_Mean_B': sample_metrics.get('channel_means', [0, 0, 0])[2],
                'Channel_Std_R': sample_metrics.get('channel_stds', [0, 0, 0])[0],
                'Channel_Std_G': sample_metrics.get('channel_stds', [0, 0, 0])[1],
                'Channel_Std_B': sample_metrics.get('channel_stds', [0, 0, 0])[2]
            })
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by NLL (lower is better)
    if 'NLL' in df.columns:
        df = df.sort_values('NLL')
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    # Create formatted table for display
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*100)
    print(df.to_string(index=False, float_format='%.6f'))
    print("="*100)
    
    return df

def create_detailed_metrics_plot(results: Dict[str, Dict], save_path: str):
    """Create detailed metrics comparison plot"""
    
    # Extract metrics
    models = list(results.keys())
    nll_values = [results[model].get('nll', 0) for model in models]
    bits_per_dim = [results[model].get('bits_per_dimension', 0) for model in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # NLL comparison
    axes[0, 0].bar(models, nll_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Negative Log-Likelihood Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('NLL (Lower is Better)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(nll_values):
        axes[0, 0].text(i, v + max(nll_values) * 0.01, f'{v:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Bits per dimension comparison
    axes[0, 1].bar(models, bits_per_dim, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title('Bits per Dimension Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Bits per Dimension (Lower is Better)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(bits_per_dim):
        axes[0, 1].text(i, v + max(bits_per_dim) * 0.01, f'{v:.6f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Sample quality metrics
    mean_pixel_values = []
    pixel_diversities = []
    
    for model in models:
        if 'sample_metrics' in results[model]:
            mean_pixel_values.append(results[model]['sample_metrics'].get('mean_pixel_value', 0))
            pixel_diversities.append(results[model]['sample_metrics'].get('pixel_diversity', 0))
        else:
            mean_pixel_values.append(0)
            pixel_diversities.append(0)
    
    # Mean pixel value comparison
    axes[1, 0].bar(models, mean_pixel_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Mean Pixel Value in Generated Samples', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Pixel Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mean_pixel_values):
        axes[1, 0].text(i, v + max(mean_pixel_values) * 0.01, f'{v:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Pixel diversity comparison
    axes[1, 1].bar(models, pixel_diversities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_title('Pixel Diversity in Generated Samples', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Pixel Diversity (Higher is Better)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(pixel_diversities):
        axes[1, 1].text(i, v + max(pixel_diversities) * 0.01, f'{v:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed metrics plot saved to {save_path}")
    plt.show()

def create_performance_ranking(results: Dict[str, Dict], save_path: str):
    """Create performance ranking visualization"""
    
    # Calculate composite score (lower NLL and bits per dim is better)
    rankings = []
    
    for model_name, result in results.items():
        nll = result.get('nll', float('inf'))
        bits_per_dim = result.get('bits_per_dimension', float('inf'))
        
        # Normalize scores (lower is better, so we use negative values for ranking)
        # Use a weighted combination
        composite_score = -nll - (bits_per_dim * 1000)  # Weight bits per dim more heavily
        
        rankings.append({
            'Model': model_name.upper(),
            'NLL': nll,
            'Bits_per_Dimension': bits_per_dim,
            'Composite_Score': composite_score
        })
    
    # Sort by composite score (higher is better)
    rankings.sort(key=lambda x: x['Composite_Score'], reverse=True)
    
    # Create ranking plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['Model'] for r in rankings]
    scores = [r['Composite_Score'] for r in rankings]
    
    bars = ax.barh(models, scores, color=['#2ca02c', '#ff7f0e', '#1f77b4'])
    ax.set_xlabel('Composite Performance Score (Higher is Better)')
    ax.set_title('Model Performance Ranking', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance ranking plot saved to {save_path}")
    plt.show()
    
    return rankings

def create_summary_report(results: Dict[str, Dict], rankings: List[Dict], save_path: str):
    """Create a comprehensive summary report"""
    
    with open(save_path, 'w') as f:
        f.write("PIXELRNN MODELS COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write("This report presents a comprehensive evaluation of three PixelRNN model architectures:\n")
        f.write("1. PixelCNN - Masked convolutional neural network\n")
        f.write("2. Row LSTM - Row-wise LSTM processing\n")
        f.write("3. Diagonal BiLSTM - Diagonal bidirectional LSTM processing\n\n")
        
        f.write("PERFORMANCE RANKING\n")
        f.write("-" * 20 + "\n")
        for i, ranking in enumerate(rankings, 1):
            f.write(f"{i}. {ranking['Model']}\n")
            f.write(f"   NLL: {ranking['NLL']:.6f}\n")
            f.write(f"   Bits per Dimension: {ranking['Bits_per_Dimension']:.6f}\n")
            f.write(f"   Composite Score: {ranking['Composite_Score']:.2f}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 20 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")
            f.write(f"  Negative Log-Likelihood: {result.get('nll', 'N/A'):.6f}\n")
            f.write(f"  Bits per Dimension: {result.get('bits_per_dimension', 'N/A'):.6f}\n")
            f.write(f"  Test Batches Evaluated: {result.get('num_test_batches', 'N/A')}\n")
            
            if 'sample_metrics' in result:
                sample_metrics = result['sample_metrics']
                f.write(f"  Sample Quality Metrics:\n")
                f.write(f"    Mean Pixel Value: {sample_metrics.get('mean_pixel_value', 'N/A'):.6f}\n")
                f.write(f"    Pixel Diversity: {sample_metrics.get('pixel_diversity', 'N/A'):.6f}\n")
                f.write(f"    Channel Means (R,G,B): {sample_metrics.get('channel_means', 'N/A')}\n")
                f.write(f"    Channel Stds (R,G,B): {sample_metrics.get('channel_stds', 'N/A')}\n")
        
        f.write("\n\nKEY FINDINGS\n")
        f.write("-" * 20 + "\n")
        
        # Find best performing model
        best_model = rankings[0]
        f.write(f"• Best Performing Model: {best_model['Model']}\n")
        f.write(f"  - Achieved lowest NLL of {best_model['NLL']:.6f}\n")
        f.write(f"  - Achieved lowest bits per dimension of {best_model['Bits_per_Dimension']:.6f}\n\n")
        
        # Performance gaps
        if len(rankings) > 1:
            nll_gap = rankings[1]['NLL'] - best_model['NLL']
            bits_gap = rankings[1]['Bits_per_Dimension'] - best_model['Bits_per_Dimension']
            f.write(f"• Performance Gap:\n")
            f.write(f"  - NLL difference from second best: {nll_gap:.6f}\n")
            f.write(f"  - Bits per dimension difference: {bits_gap:.6f}\n\n")
        
        f.write("TECHNICAL NOTES\n")
        f.write("-" * 20 + "\n")
        f.write("• All models were trained on CIFAR-10 dataset (32x32x3 images)\n")
        f.write("• Evaluation metrics include negative log-likelihood and bits per dimension\n")
        f.write("• Generated samples were evaluated for quality and diversity\n")
        f.write("• Lower NLL and bits per dimension indicate better model performance\n")
        f.write("• Composite score combines both metrics for overall ranking\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Use {best_model['Model']} for best overall performance\n")
        f.write("• Consider computational efficiency vs. performance trade-offs\n")
        f.write("• Further hyperparameter tuning may improve results\n")
        f.write("• Consider ensemble methods for even better performance\n")
    
    print(f"Summary report saved to {save_path}")

def main():
    """Main function to generate comprehensive report"""
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    plots_dir = os.path.join(base_dir, 'plots')
    
    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading evaluation results...")
    results = load_evaluation_results(results_dir)
    
    if not results:
        print("No evaluation results found!")
        return
    
    print(f"Found results for {len(results)} models: {list(results.keys())}")
    
    # Create comprehensive comparison table
    print("\nCreating comprehensive comparison table...")
    comparison_path = os.path.join(results_dir, 'comprehensive_model_comparison.csv')
    df = create_comprehensive_comparison_table(results, comparison_path)
    
    # Create detailed metrics plot
    print("\nCreating detailed metrics plot...")
    metrics_plot_path = os.path.join(plots_dir, 'detailed_metrics_comparison.png')
    create_detailed_metrics_plot(results, metrics_plot_path)
    
    # Create performance ranking
    print("\nCreating performance ranking...")
    ranking_plot_path = os.path.join(plots_dir, 'performance_ranking.png')
    rankings = create_performance_ranking(results, ranking_plot_path)
    
    # Create summary report
    print("\nCreating summary report...")
    report_path = os.path.join(results_dir, 'comprehensive_evaluation_report.txt')
    create_summary_report(results, rankings, report_path)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE REPORT GENERATION COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Files created:")
    print(f"  - comprehensive_model_comparison.csv")
    print(f"  - detailed_metrics_comparison.png")
    print(f"  - performance_ranking.png")
    print(f"  - comprehensive_evaluation_report.txt")

if __name__ == "__main__":
    main()
