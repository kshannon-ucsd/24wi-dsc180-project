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

class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing X-ray images in batches."""
    
    def __init__(self, data_dir, metadata, batch_size=16, img_size=(224, 224), shuffle=True, num_positions=2):
        """Initialize the data generator.
        
        Args:
            data_dir (str): Directory containing the image files.
            metadata (pd.DataFrame): DataFrame containing image metadata.
            batch_size (int): Number of samples per batch.
            img_size (tuple): Target size for the images (height, width).
            shuffle (bool): Whether to shuffle the data between epochs.
            num_positions (int): Number of unique view positions.
        """
        self.data_dir = data_dir
        self.metadata = metadata
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.num_positions = num_positions
        self.indexes = np.arange(len(metadata))
        self.on_epoch_end()
        
        # Initialize cache for preprocessed images
        self.image_cache = {}
        self.max_cache_size = 1000

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.metadata))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_images = []
        batch_positions = []
        batch_labels = []
        
        # Keep track of valid samples
        valid_samples = 0
        max_retries = 3
        current_retry = 0
        
        while valid_samples < 1 and current_retry < max_retries:
            # Fill batch arrays
            for i, idx_in_metadata in enumerate(batch_indexes):
                row = self.metadata.iloc[idx_in_metadata]
                file_path = os.path.join(self.data_dir, os.path.basename(row['DicomPath']))
                
                # Try to get image from cache first
                image = self.image_cache.get(file_path)
                if image is None and os.path.exists(file_path):
                    image = preprocess_dcm(file_path, self.img_size)
                    if image is not None and len(self.image_cache) < self.max_cache_size:
                        self.image_cache[file_path] = image
                
                if image is not None:
                    batch_images.append(image)
                    # Get view position and convert to categorical
                    view_position = self.metadata.iloc[idx_in_metadata]['ViewPosition']
                    position_categories = self.metadata['ViewPosition'].astype('category').cat.categories
                    position_idx = np.where(position_categories == view_position)[0][0]
                    batch_positions.append(
                        tf.keras.utils.to_categorical(
                            position_idx,
                            num_classes=self.num_positions
                        )
                    )
                    batch_labels.append(float(row['Abnormal']))
                    valid_samples += 1
            
            if valid_samples < 1:
                print(f"Warning: No valid samples in batch {idx}, retry {current_retry + 1}/{max_retries}")
                current_retry += 1
                # Try next set of indexes
                start_idx = (idx + current_retry) * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.metadata))
                batch_indexes = self.indexes[start_idx:end_idx]
                if end_idx > len(self.metadata):
                    # Wrap around to the beginning of the dataset
                    remaining = self.batch_size - (len(self.metadata) - start_idx)
                    batch_indexes = np.concatenate([self.indexes[start_idx:], self.indexes[:remaining]])
        
        if valid_samples < 1:
            print(f"Error: Failed to find valid samples after {max_retries} retries")
            # Return a minimal valid batch to prevent training failure
            return {
                'image_input': tf.zeros((1, *self.img_size, 1), dtype=tf.float32),
                'position_input': tf.zeros((1, self.num_positions), dtype=tf.float32)
            }, tf.zeros((1,), dtype=tf.float32)
        
        # Convert lists to numpy arrays
        batch_images = np.array(batch_images)
        batch_positions = np.array(batch_positions)
        batch_labels = np.array(batch_labels)
        
        return {
            'image_input': tf.convert_to_tensor(batch_images, dtype=tf.float32),
            'position_input': tf.convert_to_tensor(batch_positions, dtype=tf.float32)
        }, tf.convert_to_tensor(batch_labels, dtype=tf.float32)
    
    def on_epoch_end(self):
        """Update indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)