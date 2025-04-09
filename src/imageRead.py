import cv2
import os
import matplotlib.pyplot as plt

data_dir = "data"
image_folders = [f for f in os.listdir(data_dir) if f.startswith("images_")]

# Initialize an empty list to store images
images = []

for folder in image_folders:
    folder_path = os.path.join(data_dir, folder)
    image_files = os.listdir(folder_path)[:10]  # Limit to the first 10 images
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            images.append(img)
            print(f"Loaded: {image_path}")  # Debugging print
        else:
            print(f"Failed to load: {image_path}")  # Debugging print

# Display the first image as a sample
if images:
    plt.imshow(images[0], cmap='gray')
    plt.title("Sample Image")
    plt.axis('off')
    plt.show()

print(f"Total images loaded: {len(images)}")