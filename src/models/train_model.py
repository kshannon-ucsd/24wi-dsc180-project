"""Module for training the X-ray classification model.

This module implements the training pipeline for the CNN model,
including model configuration, training loop, and checkpoint management.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_model(input_shape=(224, 224, 1)):
    """Create and configure the CNN model.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)

    Returns:
        tf.keras.Model: Configured model ready for training
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_data, val_data, epochs=50, batch_size=32, model_dir='results/models'):
    """Train the model with early stopping and checkpointing.

    Args:
        model (tf.keras.Model): Model to train
        train_data (tf.data.Dataset): Training dataset
        val_data (tf.data.Dataset): Validation dataset
        epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        model_dir (str): Directory to save model checkpoints

    Returns:
        dict: Training history
    """
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history.history