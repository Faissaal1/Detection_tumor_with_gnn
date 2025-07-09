import torch
from torch_geometric.data import Data
import numpy as np
import os
from tqdm import tqdm
import cv2

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Comment out automatic data loading and processing for testing
train_data = np.load('data/preprocessed_train_images.npy')  
test_data = np.load('data/preprocessed_test_images.npy')

# Label mapping
LABEL_MAP = {
    'glioma_tumor': 0,
    'meningioma_tumor': 1,
    'pituitary_tumor': 2,
    'no_tumor': 3
}

def image_to_graph(image, label, patch_size=8):
    C, H, W = image.shape
    num_nodes = H*W
    x = torch.tensor(image.reshape(C, -1).T, dtype=torch.float)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float)
    for y in range(H):
        for x_ in range(W):
            idx = y * W + x_
            positions[idx] = torch.tensor([x_/W, y/H], dtype=torch.float)
    x = torch.cat([x, positions], dim=1)
    edge_index = []
    edge_attr = []
    for y in range(H):
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
            if x_ < W - 1:
                add_edge(idx, idx + 1)
            if y < H - 1:
                add_edge(idx, idx + W)
            if x_ < W - 1 and y < H - 1:
                add_edge(idx, idx + W + 1)
            if x_ > 0 and y < H - 1:
                add_edge(idx, idx + W - 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.y = torch.tensor([label], dtype=torch.long)
    return data

def process_dataset(root_dir, out_file):
    graphs = []
    for label_name, label in LABEL_MAP.items():
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in tqdm(os.listdir(class_dir), desc=f"Processing {label_name}"):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            graph = image_to_graph(img, label)
            graphs.append(graph)
    torch.save(graphs, out_file)
    print(f"Saved {len(graphs)} graphs to {out_file}")

if __name__ == "__main__":
    process_dataset('data/training', 'data/train_graph_wlabels.pt')
    process_dataset('data/testing', 'data/test_graph_wlabels.pt')