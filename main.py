"""
This module implements a medical diagnosis pipeline for detecting pneumonia
and sepsis. It uses machine learning models to analyze patient data and
medical images for diagnosis.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# pneumonia detection model packages
from src.etl.python.dataloader import download_and_preprocess_data
from src.models.resnet_model import pneumonia_detection_model
import tensorflow as tf

# sepsis detection model packages
from src.etl.python.meta_data_preprocessing import metadata_preprocessing
from src.etl.python.metadata_imputation import probabilistic_imputation
from src.models.catboost_model import catboost_model
from src.evaluation.sepsis_detection_evaluation import (
    catboost_classification_report
)
from src.evaluation.pneumonia_detection_evaluation import evaluate_model
from src.visualization.plot_metrics import plot_roc_curves
from src.visualization.catboost_prediction_interpretation import (
    plot_shap_values
)


def main():
    """
    Main function that executes the medical diagnosis pipeline.

    This function orchestrates the following processes:
    1. Loads and preprocesses patient metadata and sepsis data
    2. Performs data imputation on missing values
    3. Trains a sepsis detection model using CatBoost
    4. Evaluates model performance and generates visualization
    5. Generates SHAP value plots for model interpretation

    Returns:
        None
    """
    print("\n\n\n" + "=" * 44 +
          "\nStarting to build the pneumonia detection model" +
          "\n" + "=" * 44 + "\n\n\n")
    print("Loading and Preprocessing Data...")
    datasets = download_and_preprocess_data("data/raw", "data/processed")
    train_df = pd.DataFrame(datasets['train'])
    val_df = pd.DataFrame(datasets['val'])
    test_df = pd.DataFrame(datasets['test'])
    
    print('Data Loaded and Preprocessed Successfully', end='\n\n\n')

    # Train Model
    print("Training Pneumonia Detection Model...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=f'data/processed/train',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=f'data/processed/val',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df, directory=f'data/processed/test',
        x_col='filename', 
        y_col='label', 
        target_size=(224, 224), 
        batch_size=32, 
        class_mode='binary', 
        validate_filenames=False,
        shuffle=False)
    
    pneumonia_detector = pneumonia_detection_model(train_generator,
                                                 val_generator)
    
    print('Model Trained Successfully', end='\n\n\n')

    # Evaluate Model
    print('Starting Pneumonia Detection Model Evaluation', end='\n\n\n')
    evaluate_model(pneumonia_detector, test_generator)

    print('Pneumonia Detector Successfully Built')

    # Sepsis Detection Model
    print("\n\n\n" + "=" * 44 +
          "\nStarting to build the sepsis detection model" +
          "\n" + "=" * 44 + "\n\n\n")
    print("Loading Data...")
    metadata = pd.read_csv('data/sql-data/patients_metadata.csv')
    sepsis = pd.read_csv('data/sql-data/sepsis.csv', low_memory=False)
    print('Data Loaded Successfully', end='\n\n\n')

    # Preprocess and Combine Data
    print("Preprocessing Data...")
    full_data = metadata_preprocessing(metadata=metadata, sepsis=sepsis)
    print('Data Preprocessed Successfully', end='\n\n\n')

    # Impute Missing Values
    print("Imputing Data...")
    imputed_metadata = probabilistic_imputation(full_data, seed=42)
    print('Data Imputed Successfully', end='\n\n\n')

    # Train Model
    print("Training Sepsis Detection Model...")
    X = imputed_metadata.drop(columns=["sepsis"])
    y = imputed_metadata.sepsis

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    sepsis_detector = catboost_model(X_train=X_train, y_train=y_train)
    print('Model Trained Successfully', end='\n\n\n')

    # Evaluate Model
    print('Starting Sepsis Detection Model Evaluation', end='\n\n\n')
    train_report, test_report = catboost_classification_report(sepsis_detector,
                                                               X_train=X_train,
                                                               X_test=X_test,
                                                               y_train=y_train,
                                                               y_test=y_test)

    print(pd.DataFrame(train_report).T)
    # Plot ROC Curves
    plot_roc_curves(model=sepsis_detector,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    mdl_name='catboost')

    print('Pipeline Successfully Executed')
    # Add more evaluations and plots as desired (refer to visualization and
    # evaluation directories under src).
    plot_shap_values(sepsis_detector, X, X.columns, sample_size=100)
    # Save the model to a file

    # with open("best_sepsis_detection_mdl.pkl", "wb") as file:
    #     pickle.dump(catboost_mdl, file)


if __name__ == "__main__":
    main()
