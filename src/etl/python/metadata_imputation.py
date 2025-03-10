import numpy as np

def probabilistic_imputation(df, seed=42):
    """
    Impute missing values probabilistically based on the sepsis feature.
    Imputes by sampling from observed distributions.
    -----------
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with missing values
    seed : int, optional
        Random seed for reproducibility
    --------
    Returns:
    --------
    pandas DataFrame
        The dataframe with imputed values
    """
    if seed is not None:
        np.random.seed(seed)

    # Create a copy for imputation
    imputed_df = df_cleaned = df.copy().copy()

    # Get sepsis and non-sepsis indices
    sepsis_idx = df_cleaned['sepsis'] == 1
    nonsepsis_idx = df_cleaned['sepsis'] == 0

    for col in df_cleaned.columns:
        if col == 'sepsis':
            continue  # Skip the target column

        if df_cleaned[col].isna().any():
            print(f"Imputing missing values for {col}...")
            # For binary features (like pneumonia)
            if col == 'pneumonia':
                # For sepsis patients
                sepsis_pneumonia_values = (df_cleaned.loc[sepsis_idx, col]
                                           .dropna())
                if len(sepsis_pneumonia_values) > 0:
                    # Probability of pneumonia=1
                    sepsis_pneumonia_prob = sepsis_pneumonia_values.mean()
                else:
                    sepsis_pneumonia_prob = 0.5  # Default if no data

                # For non-sepsis patients
                nonsepsis_pneumonia_values = (df_cleaned.loc[nonsepsis_idx, col]
                                              .dropna())
                if len(nonsepsis_pneumonia_values) > 0:
                    nonsepsis_pneumonia_prob = nonsepsis_pneumonia_values.mean()
                else:
                    nonsepsis_pneumonia_prob = 0.3  # Default if no data

                # Impute for sepsis patients
                missing_sepsis = sepsis_idx & df_cleaned[col].isna()
                if missing_sepsis.any():
                    n_missing = missing_sepsis.sum()
                    imputed_df.loc[missing_sepsis, col] = np.random.binomial(
                        1,
                        sepsis_pneumonia_prob,
                        size=n_missing)

                # Impute for non-sepsis patients
                missing_nonsepsis = nonsepsis_idx & df_cleaned[col].isna()
                if missing_nonsepsis.any():
                    n_missing = missing_nonsepsis.sum()
                    imputed_df.loc[missing_nonsepsis, col] = np.random.binomial(
                        1,
                        nonsepsis_pneumonia_prob,
                        size=n_missing)

            # For continuous features
            else:
                # For sepsis group - sample from observed clean values
                sepsis_values = df_cleaned.loc[sepsis_idx, col].dropna()

                # Impute for sepsis patients
                missing_sepsis = sepsis_idx & df_cleaned[col].isna()
                if missing_sepsis.any():
                    n_missing = missing_sepsis.sum()
                    imputed_df.loc[missing_sepsis, col] = np.random.choice(
                        sepsis_values,
                        size=n_missing,
                        replace=True)

                # For non-sepsis group
                nonsepsis_values = df_cleaned.loc[nonsepsis_idx, col].dropna()

                # Impute for non-sepsis patients
                missing_nonsepsis = nonsepsis_idx & df_cleaned[col].isna()
                if missing_nonsepsis.any():
                    n_missing = missing_nonsepsis.sum()
                    imputed_df.loc[missing_nonsepsis, col] = np.random.choice(
                        nonsepsis_values,
                        size=n_missing,
                        replace=True)
                    
    print("====================================")
    print(f'Total number of imputed columns: {len(df.columns) - 1}')
    
    return imputed_df