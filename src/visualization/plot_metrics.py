import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

def plot_confusion_matrix(y_true, y_pred, mdl_name='catboost'):
    """
    Plot confusion matrix as a heatmap.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
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
    plt.savefig(f'plots/{mdl_name}_confusion_matrix.png')
    plt.show()


def plot_roc_curves(model, X_train, y_train, X_test, y_test, mdl_name='catboost'):
    """
    Plot ROC curves for both training and test sets
    """
    # Get probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

    # Calculate AUC
    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, label=f'Train AUC: {auc_train:.3f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC: {auc_test:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{mdl_name}_roc_curves.png')
    plt.show()

def plot_precision_recall(model, X_train, y_train, X_test, y_test, mdl_name='catboost'):
    """
    Plot Precision-Recall curves for both training and test sets
    """
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall_train, precision_train, label='Train')
    plt.plot(recall_test, precision_test, label='Test')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{mdl_name}_precision_recall_curve.png')
    plt.show()