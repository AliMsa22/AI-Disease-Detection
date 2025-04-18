# src/dataset/chestxray_dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class ChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-ray images and multi-label classification.
    """
    def __init__(self, csv_file, image_root='data/images', cache_dir='data/preprocessed',
                 use_preprocessed=True, target_size=(224, 224), train=True):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            image_root (str): Root directory containing image subdirectories.
            cache_dir (str): Directory containing preprocessed images as .npy files.
            use_preprocessed (bool): Whether to use preprocessed images from cache.
            target_size (tuple): Target size for resizing images.
            train (bool): Whether the dataset is for training (applies augmentations if True).
        """
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.use_preprocessed = use_preprocessed
        self.target_size = target_size
        self.disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                               'Mass', 'Nodule', 'No Finding']  # Define the list of diseases

        # Define transformations
        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert NumPy array to PyTorch tensor
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                transforms.RandomRotation(10),  # Randomly rotate the image by ±10 degrees
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert NumPy array to PyTorch tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])

        # Load preprocessed batch metadata if using preprocessed images
        if self.use_preprocessed:
            self.preprocessed_batches = self._load_preprocessed_metadata()

    def _load_preprocessed_metadata(self):
        """
        Load metadata for preprocessed batches without loading images into memory.
        Returns:
            List of dictionaries containing batch file paths and their filenames.
        """
        preprocessed_batches = []
        for batch_file in sorted(os.listdir(self.cache_dir)):
            if batch_file.endswith('.npy'):
                batch_path = os.path.join(self.cache_dir, batch_file)
                batch_data = np.load(batch_path, allow_pickle=True).item()
                preprocessed_batches.append({
                    'path': batch_path,
                    'filenames': batch_data['filenames']  # Only load filenames
                })
        return preprocessed_batches

    def _get_preprocessed_image(self, filename):
        """
        Retrieve a preprocessed image by filename.
        Args:
            filename (str): Name of the image file.
        Returns:
            Preprocessed image as a NumPy array.
        """
        for batch in self.preprocessed_batches:
            if filename in batch['filenames']:
                index = batch['filenames'].index(filename)
                batch_data = np.load(batch['path'], allow_pickle=True).item()  # Load only the required batch
                return batch_data['images'][index]
        raise FileNotFoundError(f"Preprocessed image {filename} not found in any batch.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image tensor.
            label (int): Class index for the disease.
        """
        row = self.data.iloc[idx]
        image_name = row['Image Index']

        # Load image (either preprocessed or raw)
        if self.use_preprocessed:
            image = self._get_preprocessed_image(image_name)
            if len(image.shape) == 2:  # Grayscale image
                image = np.expand_dims(image, axis=-1)  # Add channel dimension
        else:
            image = self.load_image(image_name)

        # Apply transformations
        if self.transform:
            if isinstance(image, np.ndarray):  # Convert NumPy array to PIL image
                image = Image.fromarray(image.astype('uint8'))
            image = self.transform(image)

        # Map 'Finding Labels' to a class index
        finding_label = row['Finding Labels']  # Single disease label
        label = self.disease_labels.index(finding_label)  # Convert label to class index

        return image, label

    def load_image(self, filename):
        """
        Loads an image from the dataset.
        Args:
            filename (str): Name of the image file.
        Returns:
            img (PIL.Image): Loaded and resized image.
        """
        for folder in os.listdir(self.image_root):  # Iterate through subdirectories
            folder_path = os.path.join(self.image_root, folder)
            image_path = os.path.join(folder_path, filename)
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                img = img.resize(self.target_size)
                return img
        raise FileNotFoundError(f"Image {filename} not found in {self.image_root}.")
