import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool

class SimpleTumorGNN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, num_classes=4):
        super(SimpleTumorGNN, self).__init__()
        
        # GNN layers with different architectures
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)  # Reduced heads
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)  # Adjusted for 2 heads
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization
        self.bn1 = BatchNorm(hidden_channels * 2)  # Adjusted for 2 heads
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(0.3)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),  # *2 for concatenated pooling
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch):
        # First GNN layer with attention
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x 