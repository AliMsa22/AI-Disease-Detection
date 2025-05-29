import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import random
import torchio as tio


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1, p=0.3):  # Add probability (p) parameter
        self.mean = mean
        self.std = std
        self.p = p  # Probability of adding noise

    def __call__(self, image):
        if random.random() < self.p:  # Apply noise with probability p
            if not isinstance(image, torch.Tensor):
                raise TypeError("AddGaussianNoise expects a tensor input. Ensure it is applied after ToTensor().")
            noise = torch.randn(image.size(), device=image.device) * self.std + self.mean
            noisy_image = image + noise
            return torch.clamp(noisy_image, 0.0, 1.0)
        return image  # No noise applied if the random probability is not met

class ElasticDeformation(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            transform = tio.RandomElasticDeformation()
            return transform(image)
        return image


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_root='data/images', cache_dir='data/preprocessed',
                 use_preprocessed=True, target_size=(512, 512), train=True):
        """
        Args:
            csv_file (str): Path to CSV file (train.csv/val.csv)
            image_root (str): Root directory containing images
            cache_dir (str): Directory for preprocessed images
            use_preprocessed (bool): Whether to use cached preprocessed images
            target_size (tuple): Target image dimensions
            train (bool): Whether to apply training augmentations
        """
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.use_preprocessed = use_preprocessed
        self.target_size = target_size
        self.disease_labels = ['Effusion', 'Mass', 'Nodule', 'No Finding']

        # Transformations
        if train:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=448),  # Crop the central region
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),  # Convert to tensor before applying noise
                AddGaussianNoise(mean=0.0, std=0.05, p=0.3),  # Add Gaussian noise with probability
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=448),  # Consistently crop the central region
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
            ])

        if self.use_preprocessed:
            self.preprocessed_batches = self._load_preprocessed_metadata()

    def _load_preprocessed_metadata(self):
        """Load metadata for cached preprocessed images"""
        preprocessed_batches = []
        for batch_file in sorted(os.listdir(self.cache_dir)):
            if batch_file.endswith('.npy'):
                batch_path = os.path.join(self.cache_dir, batch_file)
                batch_data = np.load(batch_path, allow_pickle=True).item()
                preprocessed_batches.append({
                    'path': batch_path,
                    'filenames': batch_data['filenames']
                })
        return preprocessed_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['Image Index']

        # Load image (preprocessed or raw)
        if self.use_preprocessed:
            image = self._get_preprocessed_image(image_name)
        else:
            image = self._load_raw_image(image_name)

        # Apply transformations
        image = self.transform(image)
        label = self.disease_labels.index(row['Finding Labels'])
        return image, label

    def _get_preprocessed_image(self, filename):
        """Retrieve preprocessed image from cache and ensure it is grayscale"""
        for batch in self.preprocessed_batches:
            if filename in batch['filenames']:
                idx = batch['filenames'].index(filename)
                image = np.load(batch['path'], allow_pickle=True).item()['images'][idx]

                # Ensure the array has the correct shape
                if len(image.shape) == 3 and image.shape[-1] == 1:  # Shape (H, W, 1)
                    image = np.squeeze(image, axis=-1)  # Convert to (H, W)
                elif len(image.shape) == 2:  # Shape (H, W)
                    pass  # Already valid
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")

                # Denormalize the image back to [0, 255] if the image is in [-1, 1]
                if image.min() < 0:  # Check if the image is in the [-1, 1] range
                    image = (image + 1) * 0.5 * 255  # Reverse the normalization to [0, 255]

                # Ensure the image is correctly scaled back to [0, 255]
                image = np.clip(image, 0, 255).astype(np.uint8)

                # Convert to PIL Image for further transformations (if needed)
                image = Image.fromarray(image)  # Return as PIL.Image

                return image

        raise FileNotFoundError(f"Preprocessed image {filename} not found")

    def _load_raw_image(self, filename):
        """Load raw grayscale image"""
        for folder in os.listdir(self.image_root):
            path = os.path.join(self.image_root, folder, filename)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if img is None:
                    raise ValueError(f"Failed to load image: {path}")

                # Resize to target size
                img = cv2.resize(img, self.target_size)

                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0

                return Image.fromarray((img * 255).astype(np.uint8))  # Return as PIL.Image
        raise FileNotFoundError(f"Image {filename} not found")