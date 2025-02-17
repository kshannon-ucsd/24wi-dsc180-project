import os
import pandas as pd
import pydicom as dicom
from skimage.transform import resize
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json

def load_and_preprocess_data(data_path):
    """Load and preprocess the X-ray data."""
    if os.path.isfile(data_path):
        # If data_path is a CSV file
        df = pd.read_csv(data_path)
        if len(df) == 0:
            raise ValueError(f"No data found in {data_path}. Please ensure the data files are properly populated.")
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Extract filenames from the paths and construct new paths in xray_imgs directory
        def get_dicom_path(relative_path):
            filename = os.path.basename(relative_path)
            return os.path.join(base_dir, 'data', 'raw', 'xray_imgs', filename)
            
        pixels = df['DicomPath'].apply(lambda x: resize(dicom.dcmread(get_dicom_path(x)).pixel_array, (256, 256)))
        X = np.stack(pixels)
        y = df['Abnormal'].astype(int)
    else:
        # If data_path is a DICOM file
        try:
            img = dicom.dcmread(data_path).pixel_array
            X = np.expand_dims(resize(img, (256, 256)), axis=0)
            y = None  # No label available for single image
        except Exception as e:
            raise ValueError(f"Error loading DICOM file {data_path}: {str(e)}")
    
    return X, y

def train_knn_model(X_train, y_train, X_test, y_test, n_neighbors=3):
    """Train KNN model and make predictions."""
    # Reshape the data
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Train model
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train_2d, y_train)
    
    # Make predictions
    y_pred = neigh.predict(X_test_2d)
    
    return y_pred

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, conf_matrix

def main():
    import sys
    
    # Check if a DICOM file path is provided as an argument
    if len(sys.argv) > 1:
        dicom_path = sys.argv[1]
        try:
            # Process single DICOM file
            X, _ = load_and_preprocess_data(dicom_path)
            print(f"Successfully loaded and preprocessed DICOM file: {dicom_path}")
            print(f"Image shape after preprocessing: {X.shape}")
        except Exception as e:
            print(f"Error processing DICOM file: {str(e)}")
            sys.exit(1)
    else:
        # Original training and evaluation workflow
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        train_data_path = os.path.join(base_dir, 'data', 'preprocessed', 'train_metadata.csv')
        test_data_path = os.path.join(base_dir, 'data', 'preprocessed', 'test_metadata.csv')
        output_dir = os.path.join(base_dir, 'output', 'xray_dataset', 'baseline')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess data
        X_train, y_train = load_and_preprocess_data(train_data_path)
        X_test, y_test = load_and_preprocess_data(test_data_path)
        
        # Train model and make predictions
        y_pred = train_knn_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        accuracy, conf_matrix = evaluate_model(y_test, y_pred)
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        with open(os.path.join(output_dir, 'knn_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print results
        print(f'Accuracy: {accuracy}')
        print('Confusion Matrix:')
        print(conf_matrix)

if __name__ == '__main__':
    main()










