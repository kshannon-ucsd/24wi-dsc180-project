import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from datetime import datetime

from data.split_data import create_dataset_splits
from models.train_model import create_model, train_model
from visualization.plot_metrics import plot_training_history

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

    def load_dicom_image(dcm_path, target_size=(224, 224)):
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        img = img / np.max(img)
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = tf.image.resize(img, target_size).numpy()
        return img

    def dicom_generator(df, batch_size=16, target_size=(224, 224)):
        while True:
            df_sample = df.sample(frac=1)
            for start in range(0, len(df_sample), batch_size):
                batch_df = df_sample.iloc[start:start+batch_size]
                batch_images = []
                batch_labels = []
                for _, row in batch_df.iterrows():
                    # Extract just the filename from the full path
                    filename = os.path.basename(row['DicomPath'])
                    # Construct the full path in the flat directory structure
                    dicom_path = os.path.join(raw_images_dir, filename)
                    label = row['Abnormal']

                    img = load_dicom_image(dicom_path, target_size)
                    batch_images.append(img)
                    batch_labels.append(label)

                yield np.array(batch_images), np.array(batch_labels)

    # Create data generators
    train_gen = dicom_generator(train_metadata, batch_size=model_config['batch_size'])
    val_gen = dicom_generator(test_metadata, batch_size=model_config['batch_size'])

    # Calculate steps per epoch
    steps_per_epoch = len(train_metadata) // model_config['batch_size']
    val_steps = len(test_metadata) // model_config['batch_size']

    print(f"Training steps per epoch: {steps_per_epoch}")
    print(f"Validation steps per epoch: {val_steps}")
    
    # Print dataset information
    print(f"Training batches: {steps_per_epoch}") # Using steps_per_epoch instead of len(train_gen)
    print(f"Validation batches: {val_steps}") # Using val_steps instead of len(val_gen)
    
    # Build and train model
    print("Building and training model...")
    model = create_model(input_shape=model_config['input_shape'])
    
    # Configure optimizer with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Configure callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=model_config['patience'],
            min_delta=model_config['min_delta'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train with generators
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=model_config['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks
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