# src/preprocessing/load_labels.py
# This script processes the dataset CSV file to encode disease labels 
# into binary format for machine learning tasks.
import os
import pandas as pd

DISEASES = ['Effusion', 'Mass', 'Nodule', 'No Finding']

def generate_labels_from_folders(base_folder='data/images'):
    """
    Generate a DataFrame with image filenames and their corresponding labels
    based on the folder structure.
    """
    # Map folder names to labels
    folder_to_label = {
        'No Finding': 'No Finding',  # Updated folder name
        'Nodule': 'Nodule',
        'Effusion': 'Effusion',
        'Mass': 'Mass',
    }

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    data = []
    for folder_name, label in folder_to_label.items():
        disease_folder = os.path.join(base_folder, folder_name)
        if os.path.exists(disease_folder):
            if not os.listdir(disease_folder):
                print(f"Warning: Folder '{folder_name}' is empty.")
            for filename in os.listdir(disease_folder):
                if filename.endswith(valid_extensions):
                    data.append({'Image Index': filename, 'Finding Labels': label})
        else:
            print(f"Warning: Folder for disease '{folder_name}' not found in {base_folder}.")
    
    df = pd.DataFrame(data)
    return df
