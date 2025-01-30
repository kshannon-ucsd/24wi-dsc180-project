"""Module for handling dataset splitting and organization.

This module provides functionality for splitting data into training and validation sets,
and organizing the dataset structure.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_splits(metadata_path, train_size=0.8):
    """Split dataset into training and validation sets.

    Args:
        metadata_path (str): Path to the metadata CSV file
        train_size (float): Proportion of data to use for training
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df) Pandas DataFrames containing split data
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Perform stratified split to maintain class distribution
    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df['Abnormal']
    )
    
    # Create preprocessed directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessed_dir = os.path.join(base_dir, 'data', 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Save split datasets
    train_df.to_csv(os.path.join(preprocessed_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(preprocessed_dir, 'test_metadata.csv'), index=False)
    
    return train_df, val_df

def save_split_metadata(train_df, val_df, output_dir):
    """Save split datasets to separate CSV files.

    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        output_dir (str): Directory to save the split files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)