"""Module for visualizing model metrics and training results.

This module provides functions for creating plots and visualizations of model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history, output_path=None):
    """Plot training and validation metrics over epochs.

    Args:
        history (dict): Training history from model.fit()
        output_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, output_path=None):
    """Plot confusion matrix as a heatmap.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Abnormal'],
        yticklabels=['Normal', 'Abnormal']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()