# src/preprocessing/image_utils.py
# This script provides utilities for loading and preprocessing images. 
# It includes functions to load images, resize them, normalize pixel values, 
# and ensure compatibility with machine learning models.

from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
import logging

logging.basicConfig(filename='image_loading_errors.log', level=logging.ERROR)

def load_image(filename, base_folder='data/images', target_size=(224, 224)):
    if not isinstance(filename, str):
        raise TypeError(f"Expected filename to be a string, but got {type(filename)}")
    for subdir in os.listdir(base_folder):
        subpath = os.path.join(base_folder, subdir)
        image_path = os.path.join(subpath, filename)
        if os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                continue
            # Preprocess the image
            img = preprocess_image(img)  # Apply preprocessing filters
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, target_size)  # Resize to target size
            return img / 255.0  # Normalize pixel values
    return None

def load_images_in_parallel(image_filenames, base_folder='data/images', target_size=(224, 224)):
    def process_image(filename):
        # Explicitly handle numpy arrays and lists
        if isinstance(filename, np.ndarray):
            filename = filename.item() if filename.size == 1 else filename.flatten()[0]
        elif isinstance(filename, (list, tuple)):
            filename = filename[0]
        
        # Final safety check
        if not isinstance(filename, str):
            filename = str(filename)

        return load_image(filename, base_folder, target_size)


    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_filenames))
    return [img for img in images if img is not None]

def preprocess_images_in_batches(image_filenames, batch_size=100, base_folder='data/images', target_size=(224, 224)):
    """
    Preprocess images in batches to reduce memory usage.
    """
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i + batch_size]
        yield load_images_in_parallel(batch_filenames, base_folder, target_size)

def save_preprocessed_images(images, filenames, output_dir='data/preprocessed', batch_index=0):
    """
    Save preprocessed images and their filenames to disk as a batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'batch_{batch_index}.npy')
    np.save(output_path, {'images': images, 'filenames': filenames})
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

def get_cached_filenames(cache_dir='data/preprocessed'):
    """
    Retrieve all filenames from cached batches.
    """
    cached_filenames = []
    for batch_file in os.listdir(cache_dir):
        if batch_file.endswith('.npy'):
            batch_data = np.load(os.path.join(cache_dir, batch_file), allow_pickle=True).item()
            cached_filenames.extend(batch_data['filenames'])
    return set(cached_filenames)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to reduce noise while preserving edges.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def fast_nl_means_denoising(image, h=10):
    """
    Apply Non-Local Means denoising.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def preprocess_image(image, filter_type='bilateral'):
    """
    Preprocess image by applying the chosen filter (bilateral or fastNlmeans).
    """
    if filter_type == 'bilateral':
        return bilateral_filter(image)
    elif filter_type == 'nl_means':
        return fast_nl_means_denoising(image)
    return image
