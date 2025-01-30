"""Module for handling dataset splitting and organization.

This module provides functionality for splitting data into training and validation sets,
and organizing the dataset structure.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_splits(metadata_path=None, metadata_df=None, train_size=0.8, output_dir=None, random_state=42):
    """Split dataset into training and validation sets and optionally save them.

    Args:
        metadata_path (str, optional): Path to the metadata CSV file
        metadata_df (pd.DataFrame, optional): DataFrame containing metadata
        train_size (float): Proportion of data to use for training (between 0 and 1)
        output_dir (str, optional): Directory to save the split files. If None,
            files will be saved in the default preprocessed directory
        random_state (int, optional): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df) Pandas DataFrames containing split data

    Raises:
        ValueError: If neither metadata_path nor metadata_df is provided
        ValueError: If train_size is not between 0 and 1
    """
    if metadata_df is None and metadata_path is not None:
        metadata_df = pd.read_csv(metadata_path)
    elif metadata_df is None:
        raise ValueError("Either metadata_path or metadata_df must be provided")
    
    # Validate train_size parameter
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")
    
    # Perform stratified split to maintain class distribution
    train_df, val_df = train_test_split(
        metadata_df,
        train_size=train_size,
        stratify=metadata_df['Abnormal'],
        random_state=random_state
    )
    
    if output_dir is None:
        # Use default preprocessed directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(base_dir, 'data', 'preprocessed')
    
    # Save split datasets
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)
    
    return train_df, val_df