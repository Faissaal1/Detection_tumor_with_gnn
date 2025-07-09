import torch
from torch_geometric.data import DataLoader
from model import SimpleTumorGNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def plot_confusion_matrix(y_true, y_pred, class_names):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    # Add labels and title
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save metrics to JSON
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        'overall_accuracy': float((y_pred == y_true).mean())
    }
    
    with open('plots/confusion_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_graphs = torch.load('data/test_graph_wlabels.pt')
    test_loader = DataLoader(test_graphs, batch_size=8)
    
    # Load best model
    print("Loading best model...")
    checkpoint = torch.load('checkpoints/best_model.pt')
    model = SimpleTumorGNN(in_channels=5, hidden_channels=64, num_classes=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Class names
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']
    
    # Lists to store predictions and true labels
    all_preds = []
    all_targets = []
    
    # Evaluate model
    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot confusion matrix and save metrics
    print("Generating confusion matrix and saving metrics...")
    plot_confusion_matrix(all_targets, all_preds, class_names)
    
    # Print overall accuracy
    accuracy = (all_preds == all_targets).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 