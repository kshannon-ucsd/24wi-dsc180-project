# X-Ray Image Classification Project

This project implements a deep learning pipeline for classifying X-ray images using a Convolutional Neural Network (CNN) architecture. The system is designed to process DICOM format medical images and provide binary classification results.

## Project Structure

```
src/
├── data/
│   ├── preprocess.py      # Image preprocessing and normalization
│   ├── split_data.py      # Dataset splitting utilities
│   └── utilities.py       # Common data handling functions
├── models/
│   ├── train_model.py     # Model training pipeline
│   └── evaluate_model.py  # Model evaluation and metrics
├── visualization/
│   └── plot_metrics.py    # Training history and results visualization
└── __init__.py
```

## Module Descriptions

### Data Processing (`src/data/`)

The data processing modules handle all aspects of data preparation and management:

- `preprocess.py`: Implements image preprocessing pipeline including:
  - DICOM file loading and parsing
  - Image normalization and standardization
  - Resizing to model input dimensions
  
- `split_data.py`: Manages dataset organization:
  - Train/validation/test set splitting
  - Data augmentation configuration
  - Metadata management

- `utilities.py`: Provides common utility functions for:
  - File handling and path management
  - Data validation and error checking
  - Batch processing helpers

### Model Management (`src/models/`)

The model modules implement the core machine learning functionality:

- `train_model.py`: Defines and trains the CNN model:
  - Model architecture configuration
  - Training pipeline implementation
  - Hyperparameter management
  - Checkpoint handling

- `evaluate_model.py`: Handles model evaluation:
  - Performance metric calculation
  - Results analysis and reporting
  - Model validation procedures

### Visualization (`src/visualization/`)

- `plot_metrics.py`: Creates visualizations for:
  - Training history plots
  - Confusion matrices
  - Performance metric graphs
  - ROC curves and AUC scores

## Dependencies

The project requires the following Python packages:

- TensorFlow (2.x) - Deep learning framework
- NumPy - Numerical computing
- Pandas - Data manipulation
- Matplotlib - Plotting and visualization
- scikit-learn - Machine learning utilities
- pydicom - DICOM file handling
- seaborn - Statistical visualization

## Data Organization

The project data is organized into the following structure:

```
data/
├── raw/                  # Original DICOM X-ray images
│   ├── patient1/
│   │   ├── study1.dcm
│   │   └── study2.dcm
│   └── patient2/
│       └── study1.dcm
├── preprocessed/         # Normalized and resized images
│   ├── train/
│   ├── validation/
│   └── test/
└── metadata.csv         # Image and patient information
```

### Raw Data

- DICOM files (.dcm) containing original X-ray images
- Organized by patient ID for easy access
- Maintains original image quality and metadata

### Preprocessed Data

- Normalized images in NumPy array format
- Split into train/validation/test sets
- Standardized dimensions: 224x224x1
- Pixel values normalized to [0,1] range

### Metadata Structure

The metadata.csv file contains essential information about each image:

| Column        | Description                           | Type    |
|---------------|---------------------------------------|----------|
| PatientID     | Unique patient identifier             | string  |
| StudyID       | Unique study identifier               | string  |
| DicomPath     | Path to DICOM file                    | string  |
| ViewPosition  | X-ray view position (AP/PA/LATERAL)   | string  |
| Abnormal      | Binary classification label           | boolean |
| Age           | Patient age                           | integer |
| Sex           | Patient sex                           | string  |


## Training Process

The model training pipeline includes several key features:

1. Early stopping to prevent overfitting
2. Model checkpointing to save best weights
3. Training history visualization
4. Comprehensive evaluation metrics

## Results Storage

All outputs are organized in the `results/` directory:

- Trained model weights
- Evaluation metrics in JSON format
- Performance plots and visualizations
- Detailed analysis reports