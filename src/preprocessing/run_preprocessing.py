# src/preprocessing/run_preprocessing.py
# This script orchestrates the preprocessing pipeline for the dataset.
# It loads and encodes labels, splits the dataset into training, validation, 
# and test sets, and saves the resulting splits to CSV files.

import os
import logging
from load_labels import load_and_encode_labels
from image_utils import load_images_in_parallel, preprocess_images_in_batches, save_preprocessed_images, load_cached_images
from split_dataset import split_dataframe

logging.basicConfig(level=logging.INFO)

def main():
    csv_path = 'data/Data_Entry_2017.csv'
    cache_dir = 'data/preprocessed'
    try:
        df = load_and_encode_labels(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    logging.info("Loaded and encoded labels.")

    # Check if cached images exist
    cached_images = load_cached_images(cache_dir)
    if cached_images:
        logging.info(f"Using cached images: {len(cached_images)}")
    else:
        # Use batch preprocessing and save to cache
        image_filenames = df['Image Index'].tolist()
        batch_index = 0
        for batch in preprocess_images_in_batches(image_filenames, batch_size=100):
            save_preprocessed_images(batch, output_dir=cache_dir, batch_index=batch_index)
            batch_index += 1

    # Filter dataframe based on cached images
    train_df, val_df, test_df = split_dataframe(df, image_loader=lambda x: os.path.exists(os.path.join(cache_dir, f"{x}.npy")))

    logging.info("Splits:")
    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    logging.info("Saved splits to CSV.")

if __name__ == "__main__":
    main()
