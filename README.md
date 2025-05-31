# Brain Tumor Detection using Graph Neural Networks

This project implements a sophisticated brain tumor detection model using Graph Attention Networks (GAT) and Graph Neural Networks (GNN) architectures. The model processes medical imaging data represented as graphs to achieve accurate tumor detection and classification.

## Features

- Graph-based representation of medical imaging data
- Implementation of Graph Attention Networks (GAT)
- Advanced Graph Neural Network architectures
- Support for multi-class tumor segmentation
- Integration with PyTorch and PyTorch Geometric
- Requirements - Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tumor-detection-gnn.git
cd tumor-detection-gnn
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added as the project develops]

## Model Architecture

The project implements a novel approach to brain tumor detection by:
- Converting medical imaging data into graph representations
- Utilizing Graph Attention Networks for feature extraction
- Implementing advanced GNN architectures for classification
- Supporting multi-class tumor segmentation

### Medical Image Preprocessing
1. **Image Normalization**
   - Intensity normalization to [0,1] range
   - Standardization using mean and standard deviation
   - Contrast enhancement for better feature visibility

2. **Image Registration**
   - Alignment of multi-modal MRI scans
   - Spatial normalization to standard template
   - Resolution standardization

3. **Noise Reduction**
   - Application of Gaussian filtering
   - Removal of artifacts and noise
   - Edge preservation using anisotropic diffusion

### Graph Construction Pipeline
1. **Node Generation**
   - Superpixel segmentation of MRI images
   - Feature extraction for each superpixel:
     - Intensity statistics
     - Texture features (GLCM, LBP)
     - Spatial coordinates
     - Shape characteristics

2. **Edge Construction**
   - K-nearest neighbors (KNN) based on spatial proximity
   - Feature similarity-based connections
   - Multi-scale edge construction for hierarchical features

3. **Graph Representation**
   - Adjacency matrix construction
   - Feature matrix creation
   - Edge weight computation based on:
     - Spatial distance
     - Feature similarity
     - Intensity differences

4. **Graph Optimization**
   - Pruning of redundant edges
   - Edge weight normalization
   - Graph sparsification for computational efficiency

### Implementation Details
```python
# Example graph construction parameters
graph_params = {
    'k_neighbors': 8,          # Number of nearest neighbors
    'feature_dim': 128,        # Dimension of node features
    'edge_threshold': 0.5,     # Similarity threshold for edges
    'max_nodes': 1000,         # Maximum nodes per graph
    'spatial_weight': 0.7,     # Weight for spatial features
    'feature_weight': 0.3      # Weight for intensity features
}
```

## References

1. Patel, D., Patel, D., Saxena, R., & Akilan, T., (2023). Multi-class Brain Tumor Segmentation using Graph Attention Network. https://arxiv.org/pdf/2302.05598

2. Ravinder, M., et al. (2023). Enhanced brain tumor classification using graph convolutional neural network architecture. Scientific Reports, 13, 14938. https://doi.org/10.1038/s41598-023-41407-8 

3. Gürsoy, E., & Kaya, Y. (2024). Brain-GCN-Net: Graph-Convolutional Neural Network for brain tumor identification. Computers in Biology and Medicine, 180, 108971. https://doi.org/10.1016/j.compbiomed.2024.108971 

4. Mohammadi, S., & Allali, M. (2024). Advancing Brain Tumor Segmentation with Spectral–Spatial Graph Neural Networks. Applied Sciences, 14(8), 3424. https://doi.org/10.3390/app14083424 

5. Saueressig, C., et al. (2021). A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation. arXiv preprint arXiv:2109.05580. https://arxiv.org/abs/2109.05580

6. Rukiye, D., Fatih, G., Ahmet, S. (2025). Advanced Brain Tumor Classification in MR Images Using Transfer Learning and Pre-Trained Deep CNN Models. https://pubmed.ncbi.nlm.nih.gov/39796749/

## License

[License information to be added]

## Contributing

[Contribution guidelines to be added]
