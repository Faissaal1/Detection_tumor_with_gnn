import json
import matplotlib.pyplot as plt
import numpy as np
import os

def safe_json_load(file_path):
    """Safely load JSON data, handling potential truncation"""
    with open(file_path, 'r') as f:
        content = f.read()
        # Find the last complete array or object
        last_complete = content.rfind(']')
        if last_complete == -1:
            last_complete = content.rfind('}')
        if last_complete != -1:
            content = content[:last_complete + 1] + '}'
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON file is truncated. Using available data up to error position.")
            # Try to load partial data
            partial_content = content[:e.pos]
            if partial_content:
                return json.loads(partial_content + '}')
            raise

def plot_training_metrics(metrics_file):
    try:
        # Read metrics from JSON file with error handling
        metrics = safe_json_load(metrics_file)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return
    
    # Create plots directory
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Plot Loss Curves
    plt.figure(figsize=(12, 6))
    if 'train_loss' in metrics and metrics['train_loss']:
        plt.plot(metrics['train_loss'], label='Training Loss', linewidth=2, color='blue')
    if 'val_loss' in metrics and metrics['val_loss']:
        plt.plot(metrics['val_loss'], label='Validation Loss', linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot F1 Score Curves
    plt.figure(figsize=(12, 6))
    if 'train_f1' in metrics and metrics['train_f1']:
        plt.plot(metrics['train_f1'], label='Training F1', linewidth=2, color='green')
    if 'val_f1' in metrics and metrics['val_f1']:
        plt.plot(metrics['val_f1'], label='Validation F1', linewidth=2, color='purple')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Validation F1 Score Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot Learning Rate
    plt.figure(figsize=(12, 6))
    if 'learning_rates' in metrics and metrics['learning_rates']:
        plt.plot(metrics['learning_rates'], linewidth=2, color='orange')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot Class Distribution
    if 'class_distribution' in metrics:
        plt.figure(figsize=(15, 5))
        distributions = metrics['class_distribution']
        
        # Plot full dataset distribution
        if 'full_dataset' in distributions:
            plt.subplot(1, 3, 1)
            counts = list(distributions['full_dataset'].values())
            labels = list(distributions['full_dataset'].keys())
            plt.bar(labels, counts, color='skyblue')
            plt.title('Full Dataset Distribution')
            plt.xticks(rotation=45)
        
        # Plot training split distribution
        if 'train_split' in distributions:
            plt.subplot(1, 3, 2)
            counts = list(distributions['train_split'].values())
            labels = list(distributions['train_split'].keys())
            plt.bar(labels, counts, color='lightgreen')
            plt.title('Training Split Distribution')
            plt.xticks(rotation=45)
        
        # Plot validation split distribution
        if 'val_split' in distributions:
            plt.subplot(1, 3, 3)
            counts = list(distributions['val_split'].values())
            labels = list(distributions['val_split'].keys())
            plt.bar(labels, counts, color='salmon')
            plt.title('Validation Split Distribution')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print summary statistics
    print("\nTraining Summary:")
    if 'best_val_f1' in metrics:
        print(f"Best Validation F1 Score: {metrics['best_val_f1']:.4f}")
    if 'best_epoch' in metrics:
        print(f"Best Epoch: {metrics['best_epoch']}")
    if 'train_loss' in metrics and metrics['train_loss']:
        print(f"Final Training Loss: {metrics['train_loss'][-1]:.4f}")
    if 'val_loss' in metrics and metrics['val_loss']:
        print(f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}")
    if 'train_f1' in metrics and metrics['train_f1']:
        print(f"Final Training F1: {metrics['train_f1'][-1]:.4f}")
    if 'val_f1' in metrics and metrics['val_f1']:
        print(f"Final Validation F1: {metrics['val_f1'][-1]:.4f}")

if __name__ == "__main__":
    # Updated path to the metrics file
    metrics_file = 'metrics/20250613_094103/training_metrics.json'
    plot_training_metrics(metrics_file) 