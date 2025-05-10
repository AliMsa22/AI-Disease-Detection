import os
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(filename='image_loading_errors.log', level=logging.ERROR)

failed_images = 0

def load_image(filename, base_folder='data/images', target_size=(512, 512)):
    global failed_images
    for subdir in os.listdir(base_folder):
        image_path = os.path.join(base_folder, subdir, filename)
        if os.path.exists(image_path):
            try:
                # Load grayscale image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.sum() == 0:
                    failed_images += 1
                    logging.error(f"Corrupt image: {image_path}")
                    return None
                
                # Apply CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)

                # Denoise
                img = cv2.fastNlMeansDenoising(img, h=10)

                # Resize to target size (512, 512)
                img = cv2.resize(img, target_size)

                # Normalize to [-1, 1]
                img = img / 255.0  # Normalize to [0, 1]
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                
                return img
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                return None
    return None

# Function to load images in parallel
def load_images_in_parallel(image_filenames, base_folder='data/images', target_size=(512, 512)):
    def process_image(filename):
        if not isinstance(filename, str):
            filename = str(filename)
        return load_image(filename, base_folder, target_size)
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_filenames))
    return [img for img in images if img is not None]

# Save preprocessed images as numpy arrays
def save_preprocessed_images(images, filenames, output_dir='data/preprocessed', batch_index=0):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'batch_{batch_index}.npy')
    np.save(output_path, {'images': images, 'filenames': filenames})
    logging.info(f"Saved batch {batch_index} to {output_path}")
    print(f"Saved batch {batch_index} to {output_path}")

def get_cached_filenames(cache_dir='data/preprocessed'):
    cached_filenames = []
    for batch_file in os.listdir(cache_dir):
        if batch_file.endswith('.npy'):
            batch_data = np.load(os.path.join(cache_dir, batch_file), allow_pickle=True).item()
            cached_filenames.extend(batch_data['filenames'])
    return set(cached_filenames)
