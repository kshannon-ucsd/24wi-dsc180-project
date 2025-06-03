"""
This module provides visualization and interpretation tools for
CatBoost model predictions. It includes functions for plotting feature
importance, permutation importance, partial dependence, SHAP values,
calibration curves, and learning curves to help understand model
behavior and performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.inspection import permutation_importance


# 1. Feature Importance
def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from a CatBoost model
    """
    # Extract the CatBoost model from the pipeline
    catboost_model = model.named_steps['classifier']

    # Get feature importance
    importances = catboost_model.get_feature_importance()

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title('CatBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance_plot.png')
    plt.show()

    return importance_df


# 2. Permutation Importance (alternative measure that shows feature impact
# on performance)
def plot_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    top_n=20,
    n_repeats=10
):
    """
    Calculate and plot permutation importance for features
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42
    )

    # Create DataFrame for plotting
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=perm_importance_df.head(top_n)
    )
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('catboost_permutation_importance_plot.png')
    plt.show()

    return perm_importance_df


# 3. Partial Dependence Plots (shows how a feature affects predictions)
def plot_partial_dependence(
    model,
    X,
    feature_names,
    feature_idx,
    grid_resolution=50
):
    """
    Create partial dependence plot for a specific feature
    """
    # Extract CatBoost model from pipeline
    catboost_model = model.named_steps['classifier']

    # Get feature values for the selected feature
    feature_values = np.linspace(
        np.min(X[:, feature_idx]),
        np.max(X[:, feature_idx]),
        num=grid_resolution
    )

    # Create a grid for prediction
    X_pd = np.tile(X.mean(axis=0), (grid_resolution, 1))

    # Replace the feature values with our grid
    X_pd[:, feature_idx] = feature_values

    # Make predictions
    predictions = catboost_model.predict_proba(X_pd)[:, 1]

    # Plot the partial dependence
    plt.figure(figsize=(8, 6))
    plt.plot(feature_values, predictions)
    plt.xlabel(feature_names[feature_idx])
    plt.ylabel('Predicted probability')
    plt.title(f'Partial Dependence Plot: {feature_names[feature_idx]}')
    plt.grid(True)
    plt.savefig('catboost_partial_dependence_plot.png')
    plt.show()


# 4. SHAP Values (shows contribution of each feature to each prediction)
def plot_shap_values(model, X, feature_names, sample_size=100):
    """
    Calculate and plot SHAP values for CatBoost model
    """
    # Extract CatBoost model from pipeline
    catboost_model = model.named_steps['classifier']

    # Sample data for SHAP analysis if dataset is large
    if X.shape[0] > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # Create explainer
    explainer = shap.TreeExplainer(catboost_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    plt.savefig('catboost_shap_summary_plot.png')

    # Dependence plots for top features
    importance = np.abs(shap_values).mean(0)
    indices = np.argsort(importance)

    for idx in indices:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx,
            shap_values,
            X_sample,
            feature_names=feature_names
        )
        plt.savefig(
            f'catboost_{feature_names[idx]}_shap_dependence_plot.png'
        )

    return explainer, shap_values


# 6. Model calibration plot
def plot_calibration_curve(model, X_train, y_train, X_test, y_test, n_bins=10):
    """
    Plot calibration curve to check if predicted probabilities match observed
    frequencies
    """
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    def calculate_calibration(y_true, y_prob, n_bins):
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        return prob_true, prob_pred

    prob_true_train, prob_pred_train = calculate_calibration(
        y_train, y_train_proba, n_bins
    )
    prob_true_test, prob_pred_test = calculate_calibration(
        y_test,
        y_test_proba,
        n_bins
    )

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(prob_pred_train, prob_true_train, 's-', label='Train')
    plt.plot(prob_pred_test, prob_true_test, 's-', label='Test')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('catboost_calibration_curve.png')
    plt.show()


# 7. Learning curves to check for overfitting
def plot_learning_curves(model, X_train, y_train, X_test, y_test, cv=5):
    """
    Plot learning curves to check for overfitting/underfitting
    """
    from sklearn.model_selection import learning_curve

    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1
    )
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1
    )
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('catboost_learning_curve.png')
    plt.show()


def run_full_analysis(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Run a comprehensive model interpretation and analysis
    """
    print("=== Model Analysis ===")
    print("1. Feature Importance")
    importance_df = plot_feature_importance(model, feature_names)
    print(importance_df.head(10))

    print("\n2. Permutation Importance")
    perm_importance_df = plot_permutation_importance(
        model, X_test, y_test, feature_names
    )
    print(perm_importance_df.head(10))

    print("\n3. Model Calibration")
    plot_calibration_curve(model, X_train, y_train, X_test, y_test)

    print("\n4. Learning Curves")
    plot_learning_curves(model, X_train, y_train, X_test, y_test)

    print("\n5. Partial Dependence Plots for top features")
    for idx in importance_df.head(5).index:
        feature = importance_df.iloc[idx]['Feature']
        feature_idx = list(feature_names).index(feature)
        plot_partial_dependence(
            model,
            X_train.values,
            feature_names,
            feature_idx
        )

    print("\n6. SHAP Analysis")
    plot_shap_values(model, X_train, feature_names)

    return {
        'feature_importance': importance_df,
        'permutation_importance': perm_importance_df,
    }
