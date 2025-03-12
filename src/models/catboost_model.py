"""
This module implements a CatBoost classifier model for binary classification
tasks. It provides functionality to train a CatBoost model with specified
hyperparameters.
"""
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def catboost_model(X_train, y_train):
    """
    Train a CatBoost classifier model for binary classification.

    Args:
        X_train (array-like): Training data features of shape
            (n_samples, n_features)
        y_train (array-like): Target values of shape (n_samples,)

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline containing the CatBoost
        classifier
    """

    params = {
        'iterations': 3000,             # Similar to n_estimators in XGBoost
        'learning_rate': 0.4,           # Step size shrinkage
        'depth': 2,                     # Equivalent to max_depth
        'min_data_in_leaf': 4,          # Similar to min_child_weight
        'subsample': 0.8,               # Random subsampling of data
        'colsample_bylevel': 0.9,       # Similar to colsample_bytree
        'l2_leaf_reg': 0.3,             # L2 regularization
        'random_seed': 42,              # Random state for reproducibility
        'loss_function': 'Logloss',     # For binary classification
        'eval_metric': 'AUC',           # Area under curve metric
        'bootstrap_type': 'Bernoulli',  # Type of bootstrap to perform
        'verbose': False,               # Suppress verbose output
        'early_stopping_rounds': 20
    }

    # Pipeline for classification
    mdl = Pipeline([
        ('classifier', CatBoostClassifier(**params))
    ])

    mdl.fit(X_train, y_train)

    return mdl
