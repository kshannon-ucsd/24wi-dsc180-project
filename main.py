# from src.etl.python.dataloader import download_and_preprocess_data
# from src.models.resnet_model import pneumonia_detection_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# sepsis detection model packages
from src.etl.python.meta_data_preprocessing import metadata_preprocessing
from src.etl.python.metadata_imputation import probabilistic_imputation
from src.models.catboost_model import catboost_model
from src.evaluation.sepsis_detection_evaluation import catboost_classification_report
from src.visualization.plot_metrics import plot_roc_curves
def main():
    # local_raw_path = "./data/raw"
    # local_processed_path = "./data/processed"
    # datasets = download_and_preprocess_data(local_raw_path, local_processed_path)
    # train_df = pd.DataFrame(datasets['train'], columns=['filename', 'binary_label'])
    # val_df = pd.DataFrame(datasets['val'], columns=['filename', 'binary_label'])
    # train_datagen = ImageDataGenerator()
    # val_datagen = ImageDataGenerator()
    # train_generator = train_datagen.flow_from_dataframe(train_df, directory=f'{local_processed_path}/train', x_col='filename', y_col='binary_label', target_size=(224, 224), batch_size=16, class_mode='binary')
    # val_generator = val_datagen.flow_from_dataframe(val_df, directory=f'{local_processed_path}/val', x_col='filename', y_col='binary_label', target_size=(224, 224), batch_size=16, class_mode='binary')
    
    # pneumonia_detector = pneumonia_detection_model(train_generator, val_generator)

    # Sepsis Detection Model
    print("Loading Data...")
    metadata = pd.read_csv('data/sql-data/patients_metadata.csv')
    sepsis = pd.read_csv('data/sql-data/sepsis.csv')
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
    
    catboost_mdl = catboost_model(X_train=X_train, y_train=y_train)
    print('Model Trained Successfully', end='\n\n\n')
    
    # Evaluate Model
    print('Starting Sepsis Detection Model Evaluation', end='\n\n\n')
    train_report, test_report = catboost_classification_report(catboost_mdl,
                                                               X_train=X_train,
                                                               X_test=X_test,
                                                               y_train=y_train,
                                                               y_test=y_test)
    
    # Plot ROC Curves
    plot_roc_curves(model=catboost_mdl,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    mdl_name='catboost')
    
    print('Pipeline Successfully Executed')
    # Add more evaluations and plots as desired (refer to visualization and evaluation directories under src).
    
    # Save the model to a file
    
    # with open("best_sepsis_detection_mdl.pkl", "wb") as file:
    #     pickle.dump(catboost_mdl, file)
    
if __name__ == "__main__":
    main()
