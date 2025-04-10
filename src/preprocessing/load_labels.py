# src/preprocessing/load_labels.py
# This script processes the dataset CSV file to encode disease labels 
# into a binary format suitable for multi-label classification.

import pandas as pd

DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
            'Fibrosis', 'Pleural_Thickening', 'Hernia']

def load_and_encode_labels(csv_path):
    df = pd.read_csv(csv_path)
    if 'Finding Labels' not in df.columns:
        raise ValueError("The CSV file does not contain the 'Finding Labels' column.")
    for label in DISEASES:
        df[label] = df['Finding Labels'].apply(lambda x: int(label in x.split('|')))
    return df
