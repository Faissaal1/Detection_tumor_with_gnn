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
    A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
], seed=42)

def validate_image(image):
    if image is None:
        return False
    if image.size == 0:
        return False
    if np.isnan(image).any():
        return False
    return True

def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            try:
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None and validate_image(img):
                    images.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
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

def load_and_preprocess_images(folder, batch_size=32):
    images = []
    for root, dirs, files in os.walk(folder):
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_images = []
            for filename in batch_files:
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None and validate_image(img):
                    processed_img = image_transform(img)
                    if processed_img is not None:
                        batch_images.append(processed_img)
            images.extend(batch_images)
    return np.array(images)

if __name__ == "__main__":
    try:
        train_images = load_and_preprocess_images(training_dir)
        test_images = load_and_preprocess_images(test_dir)

        print(f"Training images shape: {train_images.shape}")
        print(f"Testing images shape: {test_images.shape}")

        np.save('data/preprocessed_train_images.npy', train_images)
        np.save('data/preprocessed_test_images.npy', test_images)
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")