import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_cnn_model
from reload import prepare_dataset, load_dataset

def train_model(train_data_dir, val_data_dir, train_metadata, val_metadata,
                epochs=50, batch_size=16, img_size=(224, 224)):
    # Create output directory if it doesn't exist
    output_dir = os.path.join('output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced data augmentation configuration optimized for small medical imaging dataset
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),  # Reduced rotation for medical context
        tf.keras.layers.RandomZoom(0.1),  # Moderate zoom
        tf.keras.layers.RandomTranslation(0.05, 0.05),  # Subtle translations
        tf.keras.layers.RandomFlip("horizontal"),  # Keep horizontal flip
        tf.keras.layers.RandomBrightness(0.1),  # Reduced brightness variation
        tf.keras.layers.RandomContrast(0.1),  # Reduced contrast variation
        tf.keras.layers.GaussianNoise(0.01)  # Slight noise for robustness
    ])
    
    # Create data generators with augmentation
    train_generator = load_dataset(train_data_dir, train_metadata, img_size, batch_size,
                                 augmentation=data_augmentation)
    val_generator = load_dataset(val_data_dir, val_metadata, img_size, batch_size)
    
    # Calculate class weights
    train_labels = train_metadata['Abnormal'].values
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    class_weights = {0: total/(2*class_counts[0]), 1: total/(2*class_counts[1])}
    
    # Enhanced callbacks with learning rate scheduling
    initial_learning_rate = 0.001
    decay_steps = len(train_generator) * 5  # Decay over 5 epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        decay_steps,
        t_mul=2.0,  # Double the period after each restart
        m_mul=0.9  # Slightly reduce max learning rate after each restart
    )
    
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            min_delta=0.001
        ),
        # Learning rate scheduler callback
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Build and train model with transfer learning
    model = build_pretrained_model(
        base_model_name='EfficientNetB0',
        input_shape=(224, 224, 3),
        learning_rate=1e-5,  # Using very small learning rate for transfer learning
        freeze_backbone=True
    )
    
    # Train the model with all optimizations
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Plot training history
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