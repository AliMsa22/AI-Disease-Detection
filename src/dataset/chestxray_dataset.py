# src/dataset/chestxray_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms

class ChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-ray images and multi-label classification.
    """
    def __init__(self, csv_file, image_root='data', transform=None, target_size=(224, 224)):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            image_root (str): Root directory containing image subdirectories.
            transform (callable, optional): Transform to apply to the images.
            target_size (tuple): Target size for resizing images.
        """
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.target_size = target_size
        self.disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                               'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image tensor.
            label (Tensor): Multi-label binary vector.
        """
        row = self.data.iloc[idx]
        image_name = row['Image Index']
        image = self.load_image(image_name)

        # Create label vector
        label = torch.tensor([row[d] for d in self.disease_labels], dtype=torch.float32)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

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
