from torchvision import transforms
from PIL import Image
import torch
import os

def preprocess_image(image_path, image_size=256):
    """
    Preprocess image for PneumoniaCNN to match notebook validation transform

    Args:
        image_path (str): Path to the image file
        image_size (int): Desired resize dimension (default 256)

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, image_size, image_size]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert grayscale image to RGB
    image = Image.open(image_path).convert("RGB")

    # Use same transform as validation/test in notebook
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor
