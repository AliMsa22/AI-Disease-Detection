import numpy as np
import cv2
import pydicom
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess image for PyTorch model prediction
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image as numpy array
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if image_path.lower().endswith('.dcm'):
            # Read DICOM file
            dicom = pydicom.dcmread(image_path)
            img = dicom.pixel_array
        else:
            # Read regular image file
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
        # Resize to model input size
        img = cv2.resize(img, (448, 448))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # Add dimensions for PyTorch: (batch_size, channels, height, width)
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        logger.info(f"Preprocessed image shape: {img.shape}")  # Should show (1, 1, 448, 448)
        return img
    
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise