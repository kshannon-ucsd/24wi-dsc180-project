"""Module for evaluating the trained X-ray classification model.

This module provides functionality for model evaluation, including metrics calculation
and performance assessment.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model, test_data):
    """Evaluate model performance on test dataset.

    Args:
        model (tf.keras.Model): Trained model to evaluate
        test_data (tf.data.Dataset): Test dataset

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    predictions = model.predict(test_data)
    y_pred = (predictions > 0.5).astype(int)
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    
    # Calculate metrics
    metrics = {
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Add model's built-in metrics
    model_metrics = model.evaluate(test_data)
    for name, value in zip(model.metrics_names, model_metrics):
        metrics[name] = value
    
    return metrics

def save_evaluation_results(metrics, output_dir):
    """Save evaluation results to files.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Directory to save evaluation results
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics that can be serialized to JSON
    serializable_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, list, dict))
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Save non-serializable metrics to text file
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            if metric_name not in serializable_metrics:
                f.write(f'\n{metric_name}:\n{value}\n')