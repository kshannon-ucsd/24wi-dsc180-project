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
        
        # Verify pixel data exists
        if not hasattr(dicom, 'pixel_array'):
            print(f"No pixel data found in {os.path.basename(file_path)}")
            return None
        
        # Convert to float and normalize
        try:
            image = dicom.pixel_array.astype(float)
        except Exception as e:
            print(f"Error converting pixel array in {os.path.basename(file_path)}: {str(e)}")
            return None
        
        # Check for invalid pixel values
        if np.all(image == 0):
            print(f"Zero-valued image in {os.path.basename(file_path)}")
            return None
        if np.isnan(image).any():
            print(f"NaN values found in {os.path.basename(file_path)}")
            return None
        if np.isinf(image).any():
            print(f"Infinite values found in {os.path.basename(file_path)}")
            return None
            
        # Normalize to [0,1] range
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max <= image_min:
            print(f"No pixel value range in {os.path.basename(file_path)} (min={image_min}, max={image_max})")
            return None
        
        image = (image - image_min) / (image_max - image_min)
        
        # Ensure image has correct dimensions (H, W, 1) before resizing
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) != 3 or image.shape[-1] != 1:
            print(f"Invalid image dimensions in {os.path.basename(file_path)}: {image.shape}")
            return None
        
        # Resize with padding to maintain aspect ratio
        try:
            image = tf.image.resize_with_pad(
                image,
                target_height=img_size[0],
                target_width=img_size[1],
                method=tf.image.ResizeMethod.BICUBIC
            )   
            image = image.numpy()
        except Exception as e:
            print(f"Error resizing image {os.path.basename(file_path)}: {str(e)}")
            return None
        
        return image
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None