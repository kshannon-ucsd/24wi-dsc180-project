# X-Ray Image Classification Project

This project implements a deep learning pipeline for classifying X-ray images using a Convolutional Neural Network (CNN) architecture. The system is designed to process DICOM format medical images and provide binary classification results.

## Project Structure

```
src/cnn/
├── data/
│   ├── preprocess.py      # Image preprocessing and normalization
│   ├── split_data.py      # Dataset splitting utilities
│   └── utilities.py       # Common data handling functions
├── models/
│   ├── baseline/         # Baseline model implementations
│   ├── train_model.py     # Model training pipeline
│   └── evaluate_model.py  # Model evaluation and metrics
├── visualization/
│   └── plot_metrics.py    # Training history and results visualization
└── main.py               # Main execution script
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
│   ├── increased_toy.csv # Metadata file containing image information
│   └── xray_imgs/       # Directory containing DICOM images
│       └── *.dcm        # DICOM image files
└── preprocessed/        # Normalized and resized images
    ├── train/           # Training set images
    ├── validation/      # Validation set images
    ├── test/            # Test set images
    ├── train_metadata.csv # Training set metadata
    └── test_metadata.csv  # Test set metadata
```

### Raw Data

Raw data should be stored in the `data/raw/` directory at the project root. This includes:
- The metadata file (`increased_toy.csv`) containing image information
- DICOM image files in the `xray_imgs/` subdirectory

### Preprocessed Data

Preprocessed data is automatically saved in the `data/preprocessed/` directory, including:
- Normalized and resized images split into train/validation/test sets
- Generated metadata files for training and test sets

Note: Make sure to place your raw DICOM images in the `data/raw/xray_imgs/` directory and the metadata file as `data/raw/increased_toy.csv` before running the pipeline.

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

All outputs are organized in the `output/cnn/` directory:

- Trained model weights (`best_model.h5`)
- Evaluation metrics in JSON format (in `metrics/` subdirectory)
- Performance plots and visualizations (in `plots/` subdirectory)
- Detailed analysis reports (in `reports/` subdirectory)