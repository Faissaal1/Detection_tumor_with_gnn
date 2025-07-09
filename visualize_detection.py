import torch
import cv2
import numpy as np
from model import SimpleTumorGNN
from graph import image_to_graph
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
import os
import glob
from torch.serialization import add_safe_globals
from scipy import stats
import seaborn as sns

add_safe_globals(['numpy._core.multiarray.scalar'])

def load_model(model_path, device):
    """Load the trained model."""
    model = SimpleTumorGNN(in_channels=5, hidden_channels=64, num_classes=4).to(device)
    try:
        # First try loading with weights_only=True
        checkpoint = torch.load(model_path, weights_only=True)
    except Exception as e:
        print("Warning: Could not load with weights_only=True, falling back to standard loading")
        # Fall back to standard loading if weights_only fails
        checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_image(image_path, target_size=(128, 128)):
    """Process image for model input."""
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original image for visualization
    original_image = image.copy()
    
    # Resize for model input
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert to C,H,W format
    
    return image, original_image

def get_attention_weights(model, graph, device):
    """Extract attention weights from the GAT layer."""
    with torch.no_grad():
        # Get attention weights from first GAT layer
        x, edge_index = graph.x.to(device), graph.edge_index.to(device)
        # GAT returns (output, (edge_index, attention_weights))
        _, (_, attention_weights) = model.conv1(x, edge_index, return_attention_weights=True)
        return attention_weights

def analyze_attention_patterns(attention_weights):
    """Analyze attention patterns and return statistics."""
    # Convert to numpy for analysis
    attention_np = attention_weights.mean(dim=1).cpu().numpy()
    
    # Calculate statistics
    stats_dict = {
        'mean': np.mean(attention_np),
        'std': np.std(attention_np),
        'max': np.max(attention_np),
        'min': np.min(attention_np),
        'median': np.median(attention_np),
        'skewness': stats.skew(attention_np),
        'kurtosis': stats.kurtosis(attention_np)
    }
    
    # Calculate attention distribution
    hist, bins = np.histogram(attention_np, bins=50, density=True)
    
    return stats_dict, hist, bins

def create_attention_map(attention_weights, original_shape):
    """Create attention map from attention weights."""
    # Get the number of nodes
    num_nodes = int(np.sqrt(attention_weights.shape[0]))
    
    # Reshape attention weights to 2D
    attention_map = attention_weights.mean(dim=1).cpu().numpy()
    attention_map = attention_map.reshape(num_nodes, num_nodes)
    
    # Resize to original image dimensions
    attention_map = cv2.resize(attention_map, (original_shape[1], original_shape[0]))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    return attention_map

def find_test_images(num_images=2):
    """Find multiple test images from the training data."""
    # Look for images in all tumor directories
    tumor_dirs = [
        'data/training/glioma_tumor',
        'data/training/meningioma_tumor',
        'data/training/pituitary_tumor',
        'data/training/no_tumor'
    ]
    
    test_images = []
    for dir_path in tumor_dirs:
        if os.path.exists(dir_path):
            # Get all jpg/png files in the directory
            image_files = glob.glob(os.path.join(dir_path, '*.jpg')) + \
                         glob.glob(os.path.join(dir_path, '*.png'))
            if image_files:
                test_images.extend(image_files[:num_images])
                if len(test_images) >= num_images:
                    break
    
    if not test_images:
        raise FileNotFoundError("No test images found in the training directories")
    
    return test_images[:num_images]

def visualize_detection(image_paths=None, model_path='checkpoints/best_model.pt', output_dir='visualizations'):
    """Visualize tumor detection with bounding boxes for multiple images."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, device)
    
    # If no image paths provided, find test images
    if image_paths is None:
        image_paths = find_test_images(num_images=2)
        print(f"Using test images: {image_paths}")
    
    # Class names
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']
    
    # Create a figure for all images
    plt.figure(figsize=(20, 12 * len(image_paths)))
    
    for idx, image_path in enumerate(image_paths):
        # Process image
        image, original_image = process_image(image_path)
        
        # Convert to graph
        graph = image_to_graph(image, label=0)  # Label doesn't matter for inference
        
        # Get model prediction
        with torch.no_grad():
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index, graph.batch)
            pred = out.argmax(dim=1).item()
            confidence = torch.softmax(out, dim=1)[0][pred].item()
            
            # Get attention weights
            attention_weights = get_attention_weights(model, graph, device)
        
        predicted_class = class_names[pred]
        
        # Analyze attention patterns
        stats_dict, hist, bins = analyze_attention_patterns(attention_weights)
        
        # Calculate subplot positions for this image
        # Each image gets 6 subplots in two rows
        row = idx * 2 + 1
        cols = 3
        
        # First row of plots for this image
        # Plot original image
        plt.subplot(len(image_paths) * 2, cols, (row-1)*cols + 1)
        plt.imshow(original_image)
        plt.title(f'Original Image {idx+1}')
        plt.axis('off')
        
        # Plot attention heatmap
        plt.subplot(len(image_paths) * 2, cols, (row-1)*cols + 2)
        attention_map = create_attention_map(attention_weights, original_image.shape)
        plt.imshow(attention_map, cmap='hot', alpha=0.7)
        plt.imshow(original_image, alpha=0.3)
        plt.title(f'Attention Heatmap {idx+1}\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        
        # Plot attention distribution
        plt.subplot(len(image_paths) * 2, cols, (row-1)*cols + 3)
        plt.hist(bins[:-1], bins, weights=hist, alpha=0.7)
        plt.title(f'Attention Distribution {idx+1}')
        plt.xlabel('Attention Weight')
        plt.ylabel('Density')
        
        # Second row of plots for this image
        # Plot attention statistics
        plt.subplot(len(image_paths) * 2, cols, row*cols + 1)
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats_dict.items()])
        plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
        plt.axis('off')
        plt.title(f'Attention Statistics {idx+1}')
        
        # Plot attention map with threshold
        plt.subplot(len(image_paths) * 2, cols, row*cols + 2)
        threshold = np.percentile(attention_map, 95)  # Use top 5% attention values
        thresholded_map = np.where(attention_map > threshold, attention_map, 0)
        plt.imshow(thresholded_map, cmap='hot', alpha=0.7)
        plt.imshow(original_image, alpha=0.3)
        plt.title(f'High Attention Regions {idx+1}\n(> {threshold:.2f})')
        plt.axis('off')
        
        # Plot attention map with contours
        plt.subplot(len(image_paths) * 2, cols, row*cols + 3)
        plt.imshow(original_image)
        plt.contour(attention_map, levels=[threshold], colors='red', alpha=0.7)
        plt.title(f'Attention Contours {idx+1}')
        plt.axis('off')
        
        # Print results for this image
        print(f"\nResults for Image {idx+1}:")
        print(f"Image path: {image_path}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("Attention Statistics:")
        for k, v in stats_dict.items():
            print(f"{k}: {v:.4f}")
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparative_detection.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nComparative visualization saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Run visualization with automatic test image selection
    visualize_detection() 