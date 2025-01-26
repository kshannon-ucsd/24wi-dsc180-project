import os
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_dcm(file_path, img_size=(224, 224)):
    """
    Load and preprocess a DICOM image.
    
    Args:
        file_path (str): Path to the DICOM file.
        img_size (tuple): Target size for resizing images (height, width).
    
    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(file_path)
        
        # Convert to float and normalize
        image = dicom.pixel_array.astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        
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

def prepare_dataset(metadata, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    
    Args:
        metadata (pd.DataFrame): DataFrame containing image metadata.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.
    
    Returns:
        tuple: (train_data, test_data) DataFrames containing split metadata
    """
    # Split data
    train_data, test_data = train_test_split(
        metadata, 
        test_size=test_size, 
        random_state=random_state,
        stratify=metadata['Abnormal']
    )
    
    return train_data, test_data

def load_dataset(data_dir, metadata, img_size=(224, 224)):
    """
    Load and preprocess images using metadata DataFrame.
    
    Args:
        data_dir (str): Base directory containing DICOM files.
        metadata (pd.DataFrame): DataFrame containing image metadata with DicomPath column.
        img_size (tuple): Target size for resizing images.
    
    Returns:
        tuple: (images array, labels array)
    """
    images, labels = [], []
    
    for _, row in metadata.iterrows():
        # Remove 'files/' prefix from DicomPath if present
        file_path = row['DicomPath'].replace('files/', '')
        file_path = os.path.join(data_dir, file_path)
        if os.path.exists(file_path):
            image = preprocess_dcm(file_path, img_size)
            if image is not None:
                images.append(image)
                labels.append(row['Abnormal'])
        else:
            print(f"File not found: {file_path}")
    
    # Convert lists to numpy arrays and ensure proper shapes
    images = np.array(images)
    if len(images) > 0:
        labels = np.array(labels).reshape(-1, 1)  # Reshape labels to 2D array
    else:
        print("No valid images found in the dataset")
        labels = np.array([]).reshape(-1, 1)
    
    return images, labels