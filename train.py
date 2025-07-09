import torch
from torch_geometric.data import DataLoader
from model import SimpleTumorGNN
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import random
import json
from datetime import datetime

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def calculate_metrics(preds, targets):
    """Calculate precision, recall, and F1 score for each class"""
    metrics = {}
    for i in range(4):  # 4 classes
        tp = ((preds == i) & (targets == i)).sum().item()
        fp = ((preds == i) & (targets != i)).sum().item()
        fn = ((preds != i) & (targets == i)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return metrics

def save_metrics(metrics, filename):
    """Save metrics to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_metrics(metrics, save_dir):
    """Plot and save training metrics"""
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_f1'], label='Train F1')
    plt.plot(metrics['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'f1_plot.png'))
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Time')
    plt.savefig(os.path.join(save_dir, 'lr_plot.png'))
    plt.close()

def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize metrics dictionary
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': [],
        'best_val_f1': 0,
        'best_epoch': 0,
        'class_distribution': {},
        'final_metrics': {}
    }
    
    # Load datasets
    print('Loading training data...')
    train_graphs = torch.load('data/train_graph_wlabels.pt')
    test_graphs = torch.load('data/test_graph_wlabels.pt')
    
    # Debug: Print unique labels in dataset
    train_labels = [g.y.item() for g in train_graphs]
    unique_labels = np.unique(train_labels)
    print(f"\nUnique labels in dataset: {unique_labels}")
    
    # Print full dataset distribution
    full_class_counts = np.bincount(train_labels, minlength=4)
    print("\nFull Dataset Class Distribution:")
    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    for i, count in enumerate(full_class_counts):
        print(f"{class_names[i]}: Count = {count}")
    
    # Shuffle the data before splitting
    indices = list(range(len(train_graphs)))
    random.shuffle(indices)
    train_graphs = [train_graphs[i] for i in indices]
    
    # Split training data
    train_size = int(0.8 * len(train_graphs))
    train_dataset = train_graphs[:train_size]
    val_dataset = train_graphs[train_size:]
    
    print(f'\nDataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_graphs)}')
    
    # Calculate class weights using the full dataset
    class_weights = np.ones(4)  # 4 classes
    total_samples = np.sum(full_class_counts)
    for i in range(4):
        if full_class_counts[i] > 0:
            class_weights[i] = total_samples / (4 * full_class_counts[i])
    
    class_weights = torch.FloatTensor(class_weights)
    
    # Print training split distribution
    train_labels = [g.y.item() for g in train_dataset]
    train_class_counts = np.bincount(train_labels, minlength=4)
    print("\nTraining Split Class Distribution:")
    for i, (count, weight) in enumerate(zip(train_class_counts, class_weights)):
        print(f"{class_names[i]}: Count = {count}, Weight = {weight:.4f}")
    
    # Print validation split distribution
    val_labels = [g.y.item() for g in val_dataset]
    val_class_counts = np.bincount(val_labels, minlength=4)
    print("\nValidation Split Class Distribution:")
    for i, count in enumerate(val_class_counts):
        print(f"{class_names[i]}: Count = {count}")
    
    # Create DataLoaders with smaller batch size to handle memory
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_graphs, batch_size=8)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTumorGNN(in_channels=5, hidden_channels=64, num_classes=4).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    
    # Store class distribution in metrics
    metrics['class_distribution'] = {
        'full_dataset': {name: count for name, count in zip(class_names, full_class_counts)},
        'train_split': {name: count for name, count in zip(class_names, train_class_counts)},
        'val_split': {name: count for name, count in zip(class_names, val_class_counts)}
    }
    
    # Training loop
    num_epochs = 20
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    print("\nStarting Training...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            
            del batch, out, loss, pred
            torch.cuda.empty_cache()
        
        train_loss = total_loss / len(train_loader)
        train_metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        train_f1 = np.mean([m['f1'] for m in train_metrics.values()])
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                all_val_preds.extend(pred.cpu().numpy())
                all_val_targets.extend(batch.y.cpu().numpy())
                
                del batch, out, loss, pred
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(np.array(all_val_preds), np.array(all_val_targets))
        val_f1 = np.mean([m['f1'] for m in val_metrics.values()])
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_f1)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            metrics['best_val_f1'] = best_val_f1
            metrics['best_epoch'] = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_weights': class_weights
            }, 'checkpoints/best_model.pt')
            print(f"\nNew best model saved! Validation F1: {val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join('metrics', timestamp)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics to JSON
    save_metrics(metrics, os.path.join(metrics_dir, 'training_metrics.json'))
    
    # Plot metrics
    plot_metrics(metrics, metrics_dir)
    
    print(f"\nTraining metrics saved to {metrics_dir}")
    
    # Load best model for final evaluation
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Final evaluation
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_test_preds.extend(pred.cpu().numpy())
            all_test_targets.extend(batch.y.cpu().numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_test_targets, all_test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Print final results
    print("\nFinal Test Results:")
    test_metrics = calculate_metrics(np.array(all_test_preds), np.array(all_test_targets))
    for i, class_name in enumerate(class_names):
        metrics = test_metrics[f'class_{i}']
        print(f"\n{class_name}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Generate classification report
    report = classification_report(all_test_targets, all_test_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    main() 