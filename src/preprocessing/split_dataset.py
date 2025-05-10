from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_dataframe(df, output_dir='data'):
    """Saves train/val/test splits to CSV"""
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
