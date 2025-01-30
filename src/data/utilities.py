"""Utility functions for data handling and dataset generation.

This module provides utilities for data handling, dataset preparation, and batch generation
for training the X-ray classification model.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_dcm

def validate_dicom_path(file_path):
    """Validate that a DICOM file exists and is accessible.

    Args:
        file_path (str): Path to the DICOM file

    Returns:
        bool: True if file exists and is accessible, False otherwise
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)

def create_directory_structure(base_path):
    """Create the necessary directory structure for the project.

    Args:
        base_path (str): Base path where directories should be created
    """
    directories = [
        'data/raw',
        'data/preprocessed',
        'results/models',
        'results/metrics',
        'results/figures',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

def prepare_dataset(metadata, test_size=0.2):
    """Split the dataset into train and test sets.
    
    Args:
        metadata (pd.DataFrame): DataFrame containing image metadata.
        test_size (float): Proportion of dataset to include in the test split.
    
    Returns:
        tuple: (train_data, test_data) DataFrames containing split metadata
    """
    train_data, test_data = train_test_split(
        metadata, 
        test_size=test_size, 
        stratify=metadata['Abnormal']
    )
    return train_data, test_data

class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing X-ray images in batches."""
    
    def __init__(self, data_dir, metadata, batch_size=16, img_size=(224, 224), shuffle=True):
        self.data_dir = data_dir
        self.metadata = metadata
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(metadata))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # Initialize cache for preprocessed images
        self.image_cache = {}
        self.prefetch_indexes = []
        self.prefetch_size = 2

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        batch_size = min(self.batch_size, len(self.metadata) - idx * self.batch_size)
        batch_images = np.zeros((batch_size,) + self.img_size + (1,), dtype=np.float32)
        num_positions = len(self.metadata['ViewPosition'].unique())
        batch_positions = np.zeros((batch_size, num_positions), dtype=np.float32)
        batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

        valid_samples = 0
        current_idx = idx * self.batch_size
        max_attempts = len(self.metadata)
        attempts = 0

        while valid_samples < batch_size and attempts < max_attempts:
            if current_idx >= len(self.metadata):
                current_idx = 0

            idx_in_metadata = self.indexes[current_idx]
            row = self.metadata.iloc[idx_in_metadata]
            file_path = os.path.join(self.data_dir, os.path.basename(row['DicomPath']))
            
            if os.path.exists(file_path):
                try:
                    image = preprocess_dcm(file_path, self.img_size)
                    if image is not None:
                        batch_images[valid_samples] = image
                        batch_positions[valid_samples] = tf.keras.utils.to_categorical(
                            self.metadata['ViewPosition'].astype('category').cat.codes[idx_in_metadata],
                            num_classes=num_positions
                        )
                        batch_labels[valid_samples] = row['Abnormal']
                        valid_samples += 1
                except Exception:
                    pass

            current_idx += 1
            attempts += 1

        if valid_samples < batch_size:
            batch_images = batch_images[:valid_samples]
            batch_positions = batch_positions[:valid_samples]
            batch_labels = batch_labels[:valid_samples]

        return [batch_images, batch_positions], batch_labels