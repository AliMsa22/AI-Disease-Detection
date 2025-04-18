# src/preprocessing/split_dataset.py
# This script handles splitting the dataset into training and test sets.
# It also filters out rows where the corresponding image file does not exist.

from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

def split_dataframe(df, image_loader=None):
    if image_loader:
        # Parallelize the filtering process
        with ThreadPoolExecutor() as executor:
            df['exists'] = list(executor.map(lambda x: image_loader(x) is not None, df['Image Index']))
        df = df[df['exists']]
    
    if df.empty:
        raise ValueError("No valid images found in the dataset.")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df
