#imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import classification_report

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier


def probabilistic_imputation(df, seed=None):
    """
    Impute missing values probabilistically based on the sepsis feature.
    First cleans unrealistic values, then imputes by sampling from observed distributions.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with missing values
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    pandas DataFrame
        The dataframe with imputed values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clean unrealistic values (treats them as missing)
    # df_cleaned = clean_and_normalize_values(df)
    
    df_cleaned = df.copy()
    # Create a copy for imputation
    imputed_df = df_cleaned = df.copy().copy()
    
    # Get sepsis and non-sepsis indices
    sepsis_idx = df_cleaned['sepsis'] == 1
    nonsepsis_idx = df_cleaned['sepsis'] == 0
    
    # Define realistic ranges (useful for sampling in case no clean values exist)
    # For each feature
    for col in df_cleaned.columns:
        if col == 'sepsis':
            continue  # Skip the target column
        
        # Check if column has missing values
        if df_cleaned[col].isna().any():
            print(f"Imputing missing values for {col}...")
            
            # For binary features (like pneumonia)
            if col == 'pneumonia':
                # For sepsis patients
                sepsis_pneumonia_values = df_cleaned.loc[sepsis_idx, col].dropna()
                if len(sepsis_pneumonia_values) > 0:
                    sepsis_pneumonia_prob = sepsis_pneumonia_values.mean()  # Probability of pneumonia=1
                else:
                    sepsis_pneumonia_prob = 0.5  # Default if no data
                
                # For non-sepsis patients
                nonsepsis_pneumonia_values = df_cleaned.loc[nonsepsis_idx, col].dropna()
                if len(nonsepsis_pneumonia_values) > 0:
                    nonsepsis_pneumonia_prob = nonsepsis_pneumonia_values.mean()
                else:
                    nonsepsis_pneumonia_prob = 0.3  # Default if no data
                
                # Impute for sepsis patients
                missing_sepsis = sepsis_idx & df_cleaned[col].isna()
                if missing_sepsis.any():
                    n_missing = missing_sepsis.sum()
                    imputed_df.loc[missing_sepsis, col] = np.random.binomial(1, sepsis_pneumonia_prob, size=n_missing)
                
                # Impute for non-sepsis patients
                missing_nonsepsis = nonsepsis_idx & df_cleaned[col].isna()
                if missing_nonsepsis.any():
                    n_missing = missing_nonsepsis.sum()
                    imputed_df.loc[missing_nonsepsis, col] = np.random.binomial(1, nonsepsis_pneumonia_prob, size=n_missing)
            
            # For continuous features
            else:
                # For sepsis group - sample from observed clean values
                sepsis_values = df_cleaned.loc[sepsis_idx, col].dropna()
                
                # Impute for sepsis patients
                missing_sepsis = sepsis_idx & df_cleaned[col].isna()
                if missing_sepsis.any():
                    n_missing = missing_sepsis.sum()
                    imputed_df.loc[missing_sepsis, col] = np.random.choice(sepsis_values, size=n_missing, replace=True)
                
                # For non-sepsis group
                nonsepsis_values = df_cleaned.loc[nonsepsis_idx, col].dropna()
                
                # Impute for non-sepsis patients
                missing_nonsepsis = nonsepsis_idx & df_cleaned[col].isna()
                if missing_nonsepsis.any():
                    n_missing = missing_nonsepsis.sum()
                    imputed_df.loc[missing_nonsepsis, col] = np.random.choice(nonsepsis_values, size=n_missing, replace=True)
    # Post-processing to ensure values make sense
    # Round integer-like values
    for col in ['sbp', 'resp_rate', 'heart_rate', 'mbp']:
        if col in imputed_df.columns:
            imputed_df[col] = imputed_df[col].round(0)
            
    # For bilirubin, less precision is needed
    if 'bilirubin' in imputed_df.columns:
        imputed_df['bilirubin'] = imputed_df['bilirubin'].round(2)
    
    # Ensure pneumonia is binary (0 or 1)
    if 'pneumonia' in imputed_df.columns:
        imputed_df['pneumonia'] = imputed_df['pneumonia'].round().astype(int)
    
    return imputed_df

# To examine the distributions:
def compare_distributions(original_df, imputed_df, feature):
    """
    Compare distributions of original vs imputed data, separated by sepsis status
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sepsis patients - original data
    sns.histplot(original_df.loc[original_df['sepsis']==1, feature].dropna(), 
                 kde=True, ax=axes[0,0])
    axes[0,0].set_title(f'Original {feature} - Sepsis Patients')
    
    # Non-sepsis patients - original data
    sns.histplot(original_df.loc[original_df['sepsis']==0, feature].dropna(), 
                 kde=True, ax=axes[0,1])
    axes[0,1].set_title(f'Original {feature} - Non-sepsis Patients')
    
    # Sepsis patients - imputed data
    sns.histplot(imputed_df.loc[imputed_df['sepsis']==1, feature], 
                 kde=True, ax=axes[1,0])
    axes[1,0].set_title(f'Imputed {feature} - Sepsis Patients')
    
    # Non-sepsis patients - imputed data
    sns.histplot(imputed_df.loc[imputed_df['sepsis']==0, feature], 
                 kde=True, ax=axes[1,1])
    axes[1,1].set_title(f'Imputed {feature} - Non-sepsis Patients')
    
    plt.tight_layout()
    return fig

# Usage:
# fig = compare_distributions(trial, imputed_trial, 'sbp')
# plt.show()

def second_model():
    full_data = pd.read_csv("../data/processed/full_data.csv")

    trial = full_data[['heart_rate', 'sbp', 'mbp', 'resp_rate', 'temperature', 'platelet',
       'wbc', 'bands', 'lactate', 'inr', 'ptt', 'creatinine', 'bilirubin','pneumonia']]
    trial['sepsis'] = np.where(full_data.days == -1, 0, 1)

    imputed_trial = probabilistic_imputation(trial, seed=42)

    X_abs = imputed_trial.dropna().drop(columns=["sepsis"])
    y_abs = imputed_trial.dropna().sepsis

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_abs, y_abs, test_size= 0.2, stratify=y_abs, random_state = 42)


    X_abs = trial.dropna().drop(columns="sepsis")
    y_abs = trial.dropna().sepsis

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_abs, y_abs, test_size= 0.3, stratify=y_abs, random_state = 42)

    

    
    # Pipeline for classification
    positive_pipeline = Pipeline([
        ('classifier', CatBoostClassifier(**params))
    ])

    positive_pipeline.fit(X_a_train, y_a_train)
    y_train_pred = positive_pipeline.predict(X_a_train)
    y_test_pred = positive_pipeline.predict(X_a_test)
    train_report = classification_report(y_a_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_a_test, y_test_pred, output_dict=True)

    print(classification_report(y_a_train, y_train_pred))
    print(accuracy_score(y_a_train, y_train_pred))
    print("==================", end='\n')
    print(classification_report(y_a_test, y_test_pred))
    accuracy_score(y_a_test, y_test_pred)

    return positive_pipeline

    


