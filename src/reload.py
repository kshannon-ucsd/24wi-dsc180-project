import os
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_dcm(file_path, img_size=(224, 224)):
    """
    Load and preprocess a DICOM image for model input. This function performs several
    key preprocessing steps to ensure consistent image format and normalization:
    1. Loads DICOM file and extracts pixel array
    2. Normalizes pixel values to [0,1] range for better model convergence
    3. Ensures correct dimensionality (adds channel dimension if needed)
    4. Resizes image to specified dimensions
    
    Args:
        file_path (str): Path to the DICOM file.
        img_size (tuple): Target size for resizing images (height, width).
    
    Returns:
        np.ndarray: Preprocessed image array with shape (height, width, 1).
                   Returns None if processing fails.
    """
    try:
        # Read DICOM file with specific transfer syntax handling
        dicom = pydicom.dcmread(file_path, force=True)
        
        # Handle different transfer syntaxes
        if hasattr(dicom, 'file_meta') and hasattr(dicom.file_meta, 'TransferSyntaxUID'):
            transfer_syntax = dicom.file_meta.TransferSyntaxUID
            if transfer_syntax.is_compressed:
                dicom.decompress()
        
        # Convert to float and normalize
        image = dicom.pixel_array.astype(float)
        
        # Check for invalid pixel values
        if np.all(image == 0) or np.isnan(image).any() or np.isinf(image).any():
            print(f"Invalid pixel values in {file_path}")
            return None
            
        # Normalize to [0,1] range
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max > image_min:  # Avoid division by zero
            image = (image - image_min) / (image_max - image_min)
        else:
            print(f"No pixel value range in {file_path}")
            return None
        
        # Ensure image has correct dimensions (H, W, 1) before resizing
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize image
        image = tf.image.resize(image, img_size)
        image = image.numpy()
        
        # Ensure final shape is (H, W, 1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_dataset(metadata, test_size=0.2):
    """
    Split the dataset into train and test sets.
    
    Args:
        metadata (pd.DataFrame): DataFrame containing image metadata.
        test_size (float): Proportion of dataset to include in the test split.
    
    Returns:
        tuple: (train_data, test_data) DataFrames containing split metadata
    """
    # Split data
    train_data, test_data = train_test_split(
        metadata, 
        test_size=test_size, 
        stratify=metadata['Abnormal']
    )
    
    return train_data, test_data

class DataGenerator(tf.keras.utils.Sequence):
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
        # Prefetch next batch indices
        self.prefetch_indexes = []
        self.prefetch_size = 2  # Number of batches to prefetch

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        # Initialize arrays with the correct batch size
        batch_size = self.batch_size
        batch_images = np.zeros((batch_size,) + self.img_size + (1,), dtype=np.float32)
        num_positions = len(self.metadata['ViewPosition'].unique())
        batch_positions = np.zeros((batch_size, num_positions), dtype=np.float32)
        batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

        valid_samples = 0
        current_idx = idx * self.batch_size
        max_attempts = len(self.metadata)  # Maximum number of attempts to fill the batch
        attempts = 0
        valid_indices = []

        while valid_samples < batch_size and attempts < max_attempts:
            if current_idx >= len(self.metadata):
                current_idx = 0  # Wrap around to the beginning if needed

            idx_in_metadata = self.indexes[current_idx]
            row = self.metadata.iloc[idx_in_metadata]
            relative_path = row['DicomPath']
            filename = os.path.basename(relative_path)
            file_path = os.path.join(self.data_dir, filename)
            
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
                        valid_indices.append(valid_samples - 1)
                except Exception as e:
                    pass  # Skip failed images silently

            current_idx += 1
            attempts += 1

        if valid_samples == 0:
            raise ValueError(f"Could not find any valid images after {max_attempts} attempts")

        # If we don't have enough samples, repeat the valid ones
        if valid_samples < batch_size:
            for i in range(valid_samples, batch_size):
                idx_to_repeat = valid_indices[i % len(valid_indices)]
                batch_images[i] = batch_images[idx_to_repeat]
                batch_positions[i] = batch_positions[idx_to_repeat]
                batch_labels[i] = batch_labels[idx_to_repeat]

        # Always return full batch size
        return {'image_input': tf.convert_to_tensor(batch_images), 
                'position_input': tf.convert_to_tensor(batch_positions)}, \
               tf.convert_to_tensor(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def load_dataset(data_dir, metadata, img_size=(224, 224), batch_size=16):
    """
    Create data generator for efficient loading of large datasets.
    
    Args:
        data_dir (str): Base directory containing DICOM files.
        metadata (pd.DataFrame): DataFrame containing image metadata with DicomPath column.
        img_size (tuple): Target size for resizing images.
        batch_size (int): Number of images per batch.
    
    Returns:
        DataGenerator: A generator that yields batches of (images, labels)
    """
    return DataGenerator(data_dir, metadata, batch_size=batch_size, img_size=img_size)