import os
import sys

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from datetime import datetime

from data.split_data import create_dataset_splits
from models.train_model import create_model, train_model
from models.evaluate_model import evaluate_model, save_evaluation_results
from visualization.plot_metrics import plot_training_history

def main():
    # Model and training configurations
    model_config = {
        'input_shape': (224, 224, 1),
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.0005,
        'patience': 8
    }
    
    # Define base paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    
    # Dataset configuration
    metadata_file = os.path.join(raw_data_dir, 'increased_toy.csv')
    raw_images_dir = os.path.join(raw_data_dir, 'xray_imgs')
    
    # Create preprocessed and output directories
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    output_dir = os.path.join(base_dir, 'output', 'xray_dataset')
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify raw data directory structure
    if not os.path.exists(raw_images_dir):
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")
    
    if len(os.listdir(raw_images_dir)) == 0:
        raise FileNotFoundError(f"No DICOM files found in {raw_images_dir}. Please ensure all image files are properly downloaded and extracted.")
    
    # Read and prepare metadata
    print("Loading and preparing dataset...")
    metadata = pd.read_csv(metadata_file)
    
    # Split the dataset
    train_metadata, test_metadata = create_dataset_splits(metadata_file)
    # Define image size for processing
    img_size = (224, 224)  # Set target size for image processing
    
    # Verify image files exist
    print("Verifying image files...")
    missing_files = []
    for df in [train_metadata, test_metadata]:
        for _, row in df.iterrows():
            filename = os.path.basename(row['DicomPath'])
            full_path = os.path.join(raw_images_dir, filename)
            if not os.path.exists(full_path):
                missing_files.append(filename)
    
    if missing_files:
        print(f"Error: {len(missing_files)} image files are missing.")
        print("First few missing files:")
        for path in missing_files[:5]:
            print(f" - {path}")
        raise FileNotFoundError("Required image files are missing. Please ensure all images are downloaded and extracted properly.")

    print(f"Found {len(train_metadata)} training images and {len(test_metadata)} test images.")
    
    # Convert labels to numeric format for binary classification
    for df in [train_metadata, test_metadata]:
        df['Abnormal'] = df['Abnormal'].map({1.0: 1.0, -1.0: 0.0})

    # Create data generators using the DataGenerator class from utilities
    from data.utilities import DataGenerator
    
    # Get unique view positions for one-hot encoding
    unique_positions = metadata['ViewPosition'].unique()
    num_positions = len(unique_positions)
    
    train_gen = DataGenerator(
        data_dir=raw_images_dir,
        metadata=train_metadata,
        batch_size=model_config['batch_size'],
        img_size=img_size,
        num_positions=num_positions
    )
    
    val_gen = DataGenerator(
        data_dir=raw_images_dir,
        metadata=test_metadata,
        batch_size=model_config['batch_size'],
        img_size=img_size,
        num_positions=num_positions
    )

    # Build model
    print("Building and training model...")
    model = create_model(input_shape=model_config['input_shape'])
    
    # Train the model
    history = train_model(model, train_gen, val_gen,
                         epochs=model_config['epochs'],
                         batch_size=model_config['batch_size'],
                         model_dir=output_dir)
    
    # Plot and save training history
    plot_training_history(history, 
                         output_path=os.path.join(output_dir, 'plots', 'training_history.png'))
    
    # Evaluate model on test data
    evaluation_metrics = evaluate_model(model, val_gen)
    
    # Save evaluation results
    save_evaluation_results(evaluation_metrics, output_dir)
    
    return model, history, evaluation_metrics

if __name__ == '__main__':
    main()