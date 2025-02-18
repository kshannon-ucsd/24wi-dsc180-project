# dsc180-central-code

This repository contains code for the Winter 2025 DSC 180 capstone project, mentored by Kyle Shannon. The project focuses on building a machine learning tool with a front-facing website that processes X-ray data and patient biomarkers to predict sepsis development.

## Project Structure

### `data/`
Contains all datasets used in the project.
- **`raw/`** – Holds unprocessed X-ray images.
- **`processed/`** – Holds preprocessed images:
  - Training images are resized to 224x224 pixels, slightly tilted, contrast-adjusted, and some are flipped horizontally.
  - Validation and test images are also stored here.
- **`sql-data/`** – Contains `.csv` results from SQL queries.

### `src/`
Contains all source code related to data processing, modeling, and evaluation.
- **`etl/`** – Scripts for data preprocessing:
  - Python scripts for processing raw X-ray and SQL data.
  - SQL queries for structuring processed datasets.
- **`notebooks/`** – Jupyter notebooks for:
  - Loading and preprocessing data.
  - Building and training models.
  - Evaluating model performance with metrics.
- **`visualization/`** – Scripts for generating plots:
  - Classification metrics.
  - Loss and accuracy curves.
- **`evaluation/`** – Scripts for model evaluation:
  - Calculates precision, recall, AUC, and other performance metrics.

### `main.py`
Entry point for running the full pipeline, including:
- Data preprocessing.
- Model training and evaluation.
- Generating and storing results.

## Usage
To run the full pipeline:
```bash
python main.py
```

## Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install -r requirements.txt
```

## Authors & Mentorshipu
- **Mentor:** Kyle Shannon
- **Capstone Project:** DSC 180 Winter 2025
