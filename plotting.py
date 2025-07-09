import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths
DIRECTORIES = {
    'Glioma Tumor': 'data/training/glioma_tumor',
    'Meningioma Tumor': 'data/training/meningioma_tumor',
    'No Tumor': 'data/training/no_tumor',
    'Pituitary Tumor': 'data/training/pituitary_tumor'
}

def validate_image(img_path):
    """Validate if the file is a valid image."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.warning(f"Invalid image file {img_path}: {e}")
        return False

def plot_images_from_directory(directory, num_images, title):
    """
    Plot images from a directory with proper error handling and validation.
    
    Args:
        directory (str): Path to the directory containing images
        num_images (int): Number of images to plot
        title (str): Title for the plot
    """
    images = []
    valid_paths = []
    
    # Collect valid image paths
    for root, _, files in os.walk(directory):
        for filename in files:
            if len(valid_paths) >= num_images:
                break
            img_path = os.path.join(root, filename)
            if validate_image(img_path):
                valid_paths.append(img_path)
    
    if not valid_paths:
        logging.error(f"No valid images found in directory: {directory}")
        return
    
    # Read and store images
    for img_path in valid_paths:
        try:
            img = plt.imread(img_path)
            images.append(img)
        except Exception as e:
            logging.error(f"Failed to read image {img_path}: {e}")
    
    if not images:
        logging.error(f"Failed to load any images from {directory}")
        return
    
    # Create the plot
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
    
    # Plot each image
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    
    # Add title and adjust layout
    plt.suptitle(f"{title} Samples", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_all_classes(num_images=1):
    """Plot sample images from all classes."""
    for class_name, directory in DIRECTORIES.items():
        logging.info(f"Plotting images from {class_name} directory:")
        plot_images_from_directory(directory, num_images, class_name)

if __name__ == "__main__":
    # Plot one image from each class
    plot_all_classes(num_images=1)
    
    # Uncomment to plot multiple images from each class
    # plot_all_classes(num_images=3)