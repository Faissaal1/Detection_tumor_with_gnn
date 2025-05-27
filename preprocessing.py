import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd
import random


training_dir = 'data/training'
test_dir = 'data/testing'

transform = A.Compose([
    A.HorizontalFlip(p=0.5),  
    A.VerticalFlip(p=0.5),  
    A.RandomRotate90(p=0.5), 
], seed =42)

def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def augment_image(image):

    augmented = transform(image=image)
    return augmented['image']

def image_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = augment_image(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0  
    image = np.transpose(image, (2, 0, 1)) 
    return image

def gaussian_filtering(image, sigma=1.0):
    return cv2.GaussianBlur(image, (0, 0), sigma)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def load_and_preprocess_images(folder):
    images = load_images_from_folder(folder)
    preprocessed_images = []
    for img in images:
        img = image_transform(img)
        preprocessed_images.append(img)
    return np.array(preprocessed_images)

if __name__ == "__main__":
   
    train_images = load_and_preprocess_images(training_dir)
    test_images = load_and_preprocess_images(test_dir)

    print(f"Training images shape: {train_images.shape}")
    print(f"Testing images shape: {test_images.shape}")
    

    np.save('data/preprocessed_train_images.npy', train_images)
    np.save('data/preprocessed_test_images.npy', test_images)