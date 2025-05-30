import torch
from torch_geometric.data import Data
import numpy as np

train_data = np.load('data/preprocessed_train_images.npy')  
test_data = np.load('data/preprocessed_test_images.npy')
def image_to_graph(image, patch_size=8):

    C, H, W = image.shape
    num_nodes = H*W     

    x = torch.tensor(image.reshape(C, -1).T, dtype=torch.float)
    edge_index = []

    for y in range(H):
        for x_ in range(W):
            idx = y * W + x_
            if x_ < W - 1:  
                edge_index.append((idx, idx + 1))
                edge_index.append((idx + 1, idx))
            if y < H - 1:
                edge_index.append((idx, idx + W))
                edge_index.append((idx + W, idx))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

test_graphs = [image_to_graph(img) for img in test_data]
train_graphs = [image_to_graph(img) for img in train_data]
torch.save(test_graphs, 'data/test_graph.pt')
torch.save(train_graphs, 'data/train_graph.pt')