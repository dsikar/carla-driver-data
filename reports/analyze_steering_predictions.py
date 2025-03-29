"""
Script to analyze steering angle predictions from inference results.
Loads .npy file, computes metrics, and generates visualizations.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze steering angle predictions from .npy file')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to .npy file with inference results (labels, predictions)')
    parser.add_argument('--output_dir', type=str, default='./reports',
                        help='Directory to save analysis outputs')
    return parser.parse_args()

def calculate_metrics(labels, predictions):
    """Calculate performance metrics."""
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

def plot_predictions_vs_labels(labels, predictions, output_path):
    """Plot actual steering angles over array index."""
    plt.figure(figsize=(12, 6))
    
    # Use only labels, with index as x-axis
    indices = np.arange(len(labels))  # Array indices: 0 to 24,999
    plt.plot(indices, labels, 'b-', linewidth=0.5, label='Actual Steering Angles')
    
    # Customize plot
    plt.title('Actual Steering Angles Over Dataset Index')
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Angle')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Sensible x-axis ticks for 25k values
    plt.xticks(np.arange(0, len(labels), step=5000))  # Ticks every 5,000 samples
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Higher DPI for clarity
    plt.close()

def plot_error_distribution(labels, predictions, output_path):
    """Plot distribution of prediction errors."""
    errors = predictions - labels
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function to analyze predictions."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load inference results
    results = np.load(args.input_file)
    labels = results[:, 0]      # First column: actual steering angles
    predictions = results[:, 1] # Second column: predicted steering angles
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)
    print("Analysis Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
    
    # Generate and save plots
    plot_predictions_vs_labels(
        labels, predictions,
        os.path.join(args.output_dir, 'predictions_vs_labels.png')
    )
    plot_error_distribution(
        labels, predictions,
        os.path.join(args.output_dir, 'error_distribution.png')
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
