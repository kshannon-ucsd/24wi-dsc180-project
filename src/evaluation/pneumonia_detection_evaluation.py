import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

def evaluate_model(model, datagen, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get true labels
    true_labels = datagen.classes
    
    # Make predictions
    y_pred = model.predict(datagen)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Threshold at 0.5 for binary classification
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, y_pred_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "model1_confusion_matrix.png"))
    plt.show()
    
    # Compute ROC curve and AUC
    fpr, tpr, threshold = roc_curve(true_labels, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.savefig(os.path.join(save_dir, "model1_roc_curve.png"))
    plt.show()
    plt.close()
