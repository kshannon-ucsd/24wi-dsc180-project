# X-Ray Image Classification Project

This project implements a Convolutional Neural Network (CNN) for classifying X-ray images as normal or abnormal.

## Project Structure

- `main.py`: Entry point of the application that orchestrates the entire training pipeline
- `model.py`: Contains the CNN architecture definition and model building logic
- `reload.py`: Handles data loading, preprocessing, and dataset preparation
- `training.py`: Implements model training procedures and visualization utilities

## Data Organization

The project expects data to be organized in the following structure (excluded from version control):

```
data/
├── raw/
│   ├── images/        # Directory containing toy dataset DICOM files
│   ├── xray_imgs/     # Directory containing increased dataset DICOM files
│   ├── toy.csv        # Metadata for toy dataset
│   └── increased_toy.csv  # Metadata for increased dataset
├── preprocessed/      # Directory for storing preprocessed data
│   ├── train_metadata.csv
│   └── test_metadata.csv
└── output/           # Directory for storing model outputs and results
    └── {dataset_name}_dataset/
        ├── model.h5
        └── training_history.png
```

### Data Setup Instructions

1. Create a `data` directory in the project root
2. Inside `data`, create the following subdirectories:
   - `raw/`: Store your original DICOM files and metadata
   - `preprocessed/`: Will contain processed data (automatically generated)
3. Place your DICOM files in the appropriate directory under `raw/`
4. Ensure your metadata CSV files follow the required format:
   - Column for image file paths
   - Column for binary labels (normal/abnormal)

## Dependencies

- TensorFlow (2.x)
- NumPy
- Pandas
- Matplotlib
- pydicom

## Module Details

### main.py
- Manages the overall training pipeline
- Configures dataset selection (toy/increased/full dataset)
- Handles directory creation and file organization
- Coordinates data preparation, model training, and result saving

### model.py
- Implements a CNN architecture with three convolutional blocks
- Features:
  - L2 regularization for preventing overfitting
  - Batch normalization for stable training
  - Dropout layers for regularization
  - Binary classification output

### reload.py
- Handles DICOM image preprocessing:
  - Pixel value normalization
  - Image resizing
  - Channel dimension handling
- Implements dataset splitting functionality
- Provides data loading utilities

### training.py
- Manages model training process
- Implements training history visualization
- Handles model checkpointing and result saving

## Usage

1. Set up the data directory structure as described above
2. Place your DICOM files and metadata in the appropriate directories
3. Configure dataset selection in `main.py` (toy/increased)
4. Run the training pipeline:
   ```python
   python main.py
   ```

## Model Architecture

The CNN model consists of:
- 3 Convolutional blocks with increasing filter sizes (32, 64, 128)
- Batch normalization and dropout after each block
- Dense layers with L2 regularization
- Binary classification output with sigmoid activation

## Data Preprocessing

Images are preprocessed by:
1. Loading DICOM files
2. Normalizing pixel values to [0,1] range
3. Resizing to 224x224 pixels
4. Adding channel dimension for model input

## Note on Data Storage

The `data/` directory is excluded from version control via `.gitignore` to avoid storing large files in the repository. When sharing this project, users should:
1. Share the data separately through appropriate channels (e.g., secure file transfer, cloud storage)
2. Document the data source and acquisition process
3. Ensure all team members follow the same data organization structure