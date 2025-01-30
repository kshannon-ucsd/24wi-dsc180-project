import os
import pandas as pd
import tensorflow as tf
from datetime import datetime

from reload import prepare_dataset, load_dataset
from model import build_cnn_model
from training import train_model, plot_training_history

def main():
    # Model and training configurations
    model_config = {
        # Model architecture parameters - optimized for medical imaging
        'input_shape': (224, 224, 1),
        'conv_filters': [8, 16, 32],  # Simplified filter progression
        'dense_units': 32,  # Balanced dense layer capacity
        'dropout_rates': [0.3, 0.3, 0.3],  # Consistent moderate dropout
        'l2_reg': 0.01,  # Moderate L2 regularization
        
        # Training parameters - balanced for stability and learning
        'learning_rate': 0.0005,  # Moderate learning rate
        'batch_size': 16,  # Maintained for stable gradients
        'epochs': 50,  # Increased to allow proper convergence
        'validation_split': 0.2,
        
        # Early stopping parameters
        'patience': 8,  # Increased patience for finding optimal weights
        'min_delta': 0.001  # Refined improvement threshold
    }
    
    # Define base paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    
    # Dataset configurations
    dataset_configs = {
        'toy': {
            'metadata': os.path.join(raw_data_dir, 'toy.csv'),
            'images': os.path.join(raw_data_dir, 'images')
        },
        'increased': {
            'metadata': os.path.join(raw_data_dir, 'increased_toy.csv'),
            'images': os.path.join(raw_data_dir, 'xray_imgs')
        }
    }
    
    # Select dataset (can be modified to use command line arguments)
    selected_dataset = 'increased'  # Change to 'toy' for smaller dataset 
    metadata_file = dataset_configs[selected_dataset]['metadata']
    raw_images_dir = dataset_configs[selected_dataset]['images']
    
    # Create preprocessed and output directories
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    output_dir = os.path.join(base_dir, 'output', f'{selected_dataset}_dataset')
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and prepare metadata
    print("Loading and preparing dataset...")
    metadata = pd.read_csv(metadata_file)
    train_metadata, test_metadata = prepare_dataset(metadata)
    
    # Save split metadata
    train_metadata_path = os.path.join(preprocessed_dir, 'train_metadata.csv')
    test_metadata_path = os.path.join(preprocessed_dir, 'test_metadata.csv')
    train_metadata.to_csv(train_metadata_path, index=False)
    test_metadata.to_csv(test_metadata_path, index=False)
    
    # Create data generators
    print("Setting up data generators...")
    img_size = model_config['input_shape'][:2]
    train_generator = load_dataset(raw_images_dir, train_metadata, img_size, model_config['batch_size'])
    val_generator = load_dataset(raw_images_dir, test_metadata, img_size, model_config['batch_size'])
    
    # Print dataset information
    print(f"Training batches: {len(train_generator)}")
    print(f"Validation batches: {len(val_generator)}")
    
    # Build and train model
    print("Building and training model...")
    model = build_cnn_model(
        input_shape=model_config['input_shape'],
        learning_rate=model_config['learning_rate'],
        conv_filters=model_config['conv_filters'],
        dense_units=model_config['dense_units'],
        dropout_rates=model_config['dropout_rates'],
        l2_reg=model_config['l2_reg']
    )
    
    # Train with generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=model_config['epochs'],
        verbose=1,
        batch_size=model_config['batch_size']
    )
    
    # Save model and training history
    print("\nSaving model and results...")
    model.save(os.path.join(output_dir, 'model.h5'))
    plot_training_history(history, output_dir)
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\nFinal Training Metrics:")
    print(f"Loss: {final_train_loss:.4f}")
    print(f"Accuracy: {final_train_acc:.4f}")
    print("\nFinal Validation Metrics:")
    print(f"Loss: {final_val_loss:.4f}")
    print(f"Accuracy: {final_val_acc:.4f}")
    print(f"\nResults saved in: {output_dir}")

if __name__ == '__main__':
    main()