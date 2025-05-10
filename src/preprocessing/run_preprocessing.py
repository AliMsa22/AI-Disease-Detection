import os
import logging
from image_utils import load_images_in_parallel, save_preprocessed_images, get_cached_filenames, failed_images  # Import the global counter
from split_dataset import split_dataframe
from load_labels import generate_labels_from_folders

logging.basicConfig(level=logging.INFO)

def main():
    base_folder = 'data/images'
    cache_dir = 'data/preprocessed'
    batch_size = 100  # Default batch size

    df = generate_labels_from_folders(base_folder)
    df['Image Index'] = df['Image Index'].astype(str)

    cached_filenames = get_cached_filenames(cache_dir)
    if not cached_filenames:
        image_filenames = df['Image Index'].tolist()
        for i, batch in enumerate([image_filenames[i:i+batch_size] for i in range(0, len(image_filenames), batch_size)]):
            batch_images = load_images_in_parallel(batch, target_size=(512, 512))
            save_preprocessed_images(batch_images, batch, batch_index=i)
            logging.info(f"Processed {len(batch_images)} images in batch {i}.")
        
        # Log the total number of failed images
        logging.info(f"Total failed images: {failed_images}")
        print(f"Total failed images: {failed_images}")

    df = df[df['Image Index'].isin(get_cached_filenames(cache_dir))]
    split_dataframe(df)
    

if __name__ == "__main__":
    main()