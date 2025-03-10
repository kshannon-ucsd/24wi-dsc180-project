import matplotlib.pyplot as plt
import seaborn as sns

def compare_distributions(original_df, imputed_df, feature):
    """
    Compare distributions of original vs imputed data, separated by sepsis status
    """

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
    plt.savefig(f"imputed_{feature}_distribution.png")
    
    return fig