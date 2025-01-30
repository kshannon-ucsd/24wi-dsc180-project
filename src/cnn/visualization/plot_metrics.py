"""Module for visualizing model metrics and training results.

This module provides functions for creating plots and visualizations of model performance.
"""

import os
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
    ax1.plot(history['accuracy'], label='Training')
    ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, output_path=None):
    """Plot ROC curve with AUC score.

    Args:
        fpr (array-like): False positive rates
        tpr (array-like): True positive rates
        roc_auc (float): Area under the ROC curve
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()