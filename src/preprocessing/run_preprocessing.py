# src/preprocessing/run_preprocessing.py
# This script orchestrates the preprocessing pipeline for the dataset.
# It generates labels from folder structure, splits the dataset into training, 
# validation, and test sets, and saves the resulting splits to CSV files.

import os
import logging
from image_utils import load_images_in_parallel,preprocess_images_in_batches, save_preprocessed_images, load_cached_images, get_cached_filenames
from split_dataset import split_dataframe
from load_labels import generate_labels_from_folders  # Replace load_and_encode_labels

logging.basicConfig(level=logging.INFO)

def main():
    base_folder = 'data/images'
    cache_dir = 'data/preprocessed'

    # Generate labels from folder structure
    df = generate_labels_from_folders(base_folder)
    logging.info("Generated labels from folder structure.")

    # Ensure 'Image Index' column is of type string
    df['Image Index'] = df['Image Index'].astype(str)

    # Check if cached images exist
    cached_images = load_cached_images(cache_dir)
    if cached_images:
        logging.info(f"Using cached images: {len(cached_images)}")
    else:
        # Use batch preprocessing and save to cache
        image_filenames = df['Image Index'].tolist()
        logging.info(f"Image filenames: {image_filenames[:5]}")  # Print the first 5 filenames
        batch_index = 0
        for batch_filenames in [image_filenames[i:i + 100] for i in range(0, len(image_filenames), 100)]:
            batch_images = load_images_in_parallel(batch_filenames, base_folder=base_folder, target_size=(224, 224))
            save_preprocessed_images(batch_images, batch_filenames, output_dir=cache_dir, batch_index=batch_index)
            logging.info(f"Processed and saved batch {batch_index} with {len(batch_images)} images.")
            batch_index += 1

    # Filter dataframe based on cached images
    cached_filenames = get_cached_filenames(cache_dir)
    df = df[df['Image Index'].isin(cached_filenames)]
    logging.info(f"Filtered dataframe. Remaining rows: {len(df)}")

    # Split the filtered dataframe
    train_df, test_df = split_dataframe(df)

    logging.info("Splits:")
    logging.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    logging.info("Saved splits to CSV.")

if __name__ == "__main__":
    main()
