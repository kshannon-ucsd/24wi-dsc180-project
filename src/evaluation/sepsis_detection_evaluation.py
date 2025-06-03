"""
This module provides evaluation metrics and reporting functionality for
sepsis detection models. It includes functions to generate classification
reports and accuracy scores for model evaluation.
"""

from sklearn.metrics import classification_report, accuracy_score


def catboost_classification_report(mdl, X_train, X_test, y_train, y_test):
    """
    Generate classification reports and accuracy scores for CatBoost model
    predictions.

    Args:
        mdl: Trained CatBoost model
        X_train: Training feature data
        X_test: Testing feature data
        y_train: Training target labels
        y_test: Testing target labels

    Returns:
        tuple: (train_report, test_report) containing classification metrics
        for both training and test sets as dictionaries
    """

    y_train_pred = mdl.predict(X_train)
    y_test_pred = mdl.predict(X_test)

    train_report = classification_report(
        y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    print(classification_report(y_train, y_train_pred))
    print(accuracy_score(y_train, y_train_pred))

    print("==================", end='\n')

    print(classification_report(y_test, y_test_pred))
    print(accuracy_score(y_test, y_test_pred))

    return train_report, test_report
