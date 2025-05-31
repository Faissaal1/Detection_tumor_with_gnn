import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Comment out automatic data loading and processing for testing
train_data = np.load('data/preprocessed_train_images.npy')  
test_data = np.load('data/preprocessed_test_images.npy')

def image_to_graph(image, patch_size=8):
    C, H, W = image.shape
    num_nodes = H*W     

    # Move tensors to GPU if available
    x = torch.tensor(image.reshape(C, -1).T, dtype=torch.float).to(device)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float).to(device)
    
    print("Creating node positions...")
    for y in tqdm(range(H), desc="Processing rows"):
        for x_ in range(W):
            idx = y * W + x_
            positions[idx] = torch.tensor([x_/W, y/H], dtype=torch.float).to(device)
    
    x = torch.cat([x, positions], dim=1)

    edge_index = []
    edge_attr = []

    print("Creating edges and edge features...")
    for y in tqdm(range(H), desc="Creating edges"):
        for x_ in range(W):
            idx = y * W + x_
            
            def add_edge(i, j):
                if 0 <= i < num_nodes and 0 <= j < num_nodes:
                    edge_index.append((i, j))
                    edge_index.append((j, i))
                    
                    pixel_i = image[:, i//W, i%W]
                    pixel_j = image[:, j//W, j%W]
                    intensity_diff = torch.norm(torch.tensor(pixel_i - pixel_j, dtype=torch.float))
                    distance = torch.tensor([((i//W - j//W)/H)**2 + ((i%W - j%W)/W)**2], dtype=torch.float)
                    edge_features = torch.cat([intensity_diff.unsqueeze(0), distance])
                    
                    edge_attr.append(edge_features)
                    edge_attr.append(edge_features)

            if x_ < W - 1:  # Right
                add_edge(idx, idx + 1)
            if y < H - 1:  # Down
                add_edge(idx, idx + W)
            if x_ < W - 1 and y < H - 1:  # Down-right
                add_edge(idx, idx + W + 1)
            if x_ > 0 and y < H - 1:  # Down-left
                add_edge(idx, idx + W - 1)

    print("Finalizing graph structure...")
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.stack(edge_attr).to(device)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

print("Processing test data...")
test_graphs = []
for i, img in enumerate(tqdm(test_data, desc="Processing test images")):
    test_graphs.append(image_to_graph(img))
    if (i + 1) % 10 == 0:  # Save progress every 10 images
        torch.save(test_graphs, 'data/test_graph.pt')
torch.save(test_graphs, 'data/test_graph.pt')

print("Processing training data...")
train_graphs = []
for i, img in enumerate(tqdm(train_data, desc="Processing training images")):
    train_graphs.append(image_to_graph(img))
    if (i + 1) % 10 == 0:  # Save progress every 10 images
        torch.save(train_graphs, 'data/train_graph.pt')
torch.save(train_graphs, 'data/train_graph.pt')