# src/preprocessing/image_utils.py
# This script provides utilities for loading and preprocessing images. 
# It includes functions to load images, resize them, normalize pixel values, 
# and ensure compatibility with machine learning models.

from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np

def load_image(filename, base_folder='data', target_size=(224, 224)):
    for subdir in os.listdir(base_folder):
        subpath = os.path.join(base_folder, subdir)
        image_path = os.path.join(subpath, filename)
        if os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, target_size)  # Resize to target size
            return img / 255.0  # Normalize pixel values
    return None

def load_images_in_parallel(image_filenames, base_folder='data', target_size=(224, 224)):
    def process_image(filename):
        return load_image(filename, base_folder, target_size)
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_filenames))
    return [img for img in images if img is not None]

def preprocess_images_in_batches(image_filenames, batch_size=100, base_folder='data', target_size=(224, 224)):
    """
    Preprocess images in batches to reduce memory usage.
    """
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i + batch_size]
        yield load_images_in_parallel(batch_filenames, base_folder, target_size)

def save_preprocessed_images(images, output_dir='data/preprocessed', batch_index=0):
    """
    Save preprocessed images to disk as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'batch_{batch_index}.npy')
    np.save(output_path, images)
    print(f"Saved batch {batch_index} to {output_path}")

def load_cached_images(cache_dir='data/preprocessed'):
    """
    Load preprocessed images from cached .npy files.
    """
    if not os.path.exists(cache_dir):
        return []
    cached_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.npy')]
    images = []
    for file in cached_files:
        images.extend(np.load(file, allow_pickle=True))
    print(f"Loaded {len(images)} images from cache.")
    return images
