import os
import pandas as pd
from datetime import datetime

from reload import prepare_dataset, load_dataset
from model import build_cnn_model
from training import train_model, plot_training_history

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    raw_images_dir = os.path.join(raw_data_dir, 'images')
    metadata_file = os.path.join(raw_data_dir, 'toy.csv')
    
    # Create preprocessed and output directories
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    output_dir = os.path.join(base_dir, 'output', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
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
    
    # Load and preprocess image data
    print("Loading and preprocessing images...")
    img_size = (224, 224)
    # Fix parameter order in load_dataset calls
    X_train, y_train = load_dataset(raw_images_dir, train_metadata, img_size)
    X_val, y_val = load_dataset(raw_images_dir, test_metadata, img_size)
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Build and train model
    print("Building and training model...")
    model = build_cnn_model(input_shape=(*img_size, 1))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=10,
        verbose=1
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