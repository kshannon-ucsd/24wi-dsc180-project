import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from model import build_cnn_model
from reload import prepare_dataset, load_dataset

def train_model(train_data_dir, val_data_dir, train_metadata, val_metadata,
                epochs=10, batch_size=16, img_size=(224, 224)):
    """
    Train the CNN model and save results.
    
    Args:
        train_data_dir (str): Directory containing training DICOM files
        val_data_dir (str): Directory containing validation DICOM files
        train_metadata (str): Path to training metadata CSV
        val_metadata (str): Path to validation metadata CSV
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (tuple): Image dimensions for model input
    
    Returns:
        tuple: (trained model, training history)
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    X_train, y_train = load_dataset(train_data_dir, train_metadata, img_size)
    X_val, y_val = load_dataset(val_data_dir, val_metadata, img_size)
    
    # Build and train model
    model = build_cnn_model(input_shape=(*img_size, 1))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save model
    model.save(os.path.join(output_dir, 'model.h5'))
    
    # Plot and save training history
    plot_training_history(history, output_dir)
    
    return model, history

def plot_training_history(history, output_dir):
    """
    Plot and save training history visualization.
    
    Args:
        history: Training history from model.fit()
        output_dir (str): Directory to save plots
    """
    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()