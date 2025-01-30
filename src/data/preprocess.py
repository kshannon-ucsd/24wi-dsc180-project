"""Module for preprocessing DICOM images.

This module provides functionality for loading and preprocessing DICOM images
for use in the X-ray classification model.
"""

import os
import pydicom
import numpy as np
import tensorflow as tf

def preprocess_dcm(file_path, img_size=(224, 224)):
    """Load and preprocess a DICOM image for model input.
    
    This function performs several key preprocessing steps:
    1. Loads DICOM file and extracts pixel array
    2. Normalizes pixel values to [0,1] range
    3. Ensures correct dimensionality (adds channel dimension)
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