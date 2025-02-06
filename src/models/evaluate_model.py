"""Module for evaluating the trained X-ray classification model.

This module provides functionality for model evaluation, including metrics calculation
and performance assessment.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf

def evaluate_model(model, test_data):
    """Evaluate model performance on test dataset.

    Args:
        model (tf.keras.Model): Trained model to evaluate
        test_data (tf.data.Dataset): Test dataset

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Collect predictions and true labels
    y_pred = []
    y_true = []
    
    for batch_idx in range(len(test_data)):
        try:
            batch = test_data[batch_idx]
            if batch is not None:
                x, y = batch
                y_pred.extend(model.predict(x).flatten())
                y_true.extend(y.numpy())
        except ValueError as e:
            print(f"Warning: Skipping invalid batch {batch_idx}: {str(e)}")
            continue
    
    if not y_true or not y_pred:
        raise ValueError("No valid samples found in the test dataset")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Convert predictions to binary values for classification report
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'classification_report': classification_report(y_true, y_pred_binary),
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'roc_auc': roc_auc
    }
    
    # Add model's built-in metrics
    model_metrics = model.evaluate(test_data)
    for name, value in zip(model.metrics_names, model_metrics):
        metrics[name] = value
    
    return metrics

def save_evaluation_results(metrics, output_dir):
    """Save evaluation results to files in an organized directory structure.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Base directory to save evaluation results
    """
    import os
    import json
    
    # Create directory structure
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    reports_dir = os.path.join(output_dir, 'reports')
    
    for dir_path in [metrics_dir, plots_dir, reports_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save metrics that can be serialized to JSON
    serializable_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, list, dict))
    }
    
    with open(os.path.join(metrics_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Save detailed evaluation report
    with open(os.path.join(reports_dir, 'evaluation_report.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            if metric_name not in serializable_metrics:
                f.write(f'\n{metric_name}:\n{value}\n')
    
    # Import visualization functions using direct import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization.plot_metrics import plot_confusion_matrix, plot_roc_curve
    
    # Generate and save plots
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            output_path=os.path.join(plots_dir, 'confusion_matrix.png')
        )
    
    if 'roc_curve' in metrics:
        plot_roc_curve(
            metrics['roc_curve']['fpr'],
            metrics['roc_curve']['tpr'],
            metrics['roc_auc'],
            output_path=os.path.join(plots_dir, 'roc_curve.png')
        )