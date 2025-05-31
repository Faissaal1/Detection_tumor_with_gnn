print("Script started")  # First debug print

import cv2
print("cv2 imported")  # Debug print after cv2

import numpy as np
print("numpy imported")  # Debug print after numpy

import torch
print("torch imported")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

from graph import image_to_graph
print("graph module imported")  # Debug print after graph import

import time
print("time imported")  # Debug print after time import

print("Starting image processing...")
# Load a single test image
test_image_path = 'data/training/no_tumor/1.jpg'
print(f"Loading image from: {test_image_path}")
image = cv2.imread(test_image_path)
print(f"Image loaded, shape: {image.shape if image is not None else 'None'}")

if image is None:
    print("Error: Failed to load image")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))  # Convert to C,H,W format
print("Image loaded and preprocessed successfully")

print("\nStarting graph construction...")
start_time = time.time()
# Create graph
graph = image_to_graph(image)
end_time = time.time()
print(f"Graph construction completed in {end_time - start_time:.2f} seconds")

# Print graph information
print("\nGraph Statistics:")
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.edge_index.shape[1]}")
print(f"Node feature dimension: {graph.x.shape[1]}")
print(f"Edge feature dimension: {graph.edge_attr.shape[1]}")
print("\nDetailed Shapes:")
print("Node features shape:", graph.x.shape)
print("Edge index shape:", graph.edge_index.shape)
print("Edge attributes shape:", graph.edge_attr.shape)

# Print device information for tensors
print("\nDevice Information:")
print(f"Node features device: {graph.x.device}")
print(f"Edge index device: {graph.edge_index.device}")
print(f"Edge attributes device: {graph.edge_attr.device}") 