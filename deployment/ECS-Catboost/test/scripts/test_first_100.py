import requests
import numpy as np
import json
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Define directory structure
SCRIPT_DIR = Path(__file__).resolve().parent  
ROOT_DIR = SCRIPT_DIR.parent.parent  
DATA_DIR = ROOT_DIR / "test" / "data"  

# Load .env file from config
env_path = ROOT_DIR / "config" / ".env"
print(f"📂 Loading environment from: {env_path}")
load_dotenv(env_path)

# Load test data paths from .env
X_TEST_PATH = os.getenv("X_TEST_PATH")
Y_TEST_PATH = os.getenv("Y_TEST_PATH")

# Convert them to absolute paths
X_TEST_PATH = (DATA_DIR / X_TEST_PATH).resolve()
Y_TEST_PATH = (DATA_DIR / Y_TEST_PATH).resolve()

# Determine deployment environment
DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV") 

# Define API URL based on environment
if DEPLOYMENT_ENV == "local":
    API_URL = "http://127.0.0.1:8080/predict"
    print("📍 Running in LOCAL environment")
else:
    API_URL = os.environ.get("API_URL")
    print(f"☁️ Running in CLOUD environment: {API_URL}")

# Load feature data (X_test.csv) and true labels (Y_test.csv)
print(f"📊 Loading test data from: {X_TEST_PATH}")
X_test = pd.read_csv(X_TEST_PATH, index_col=0)
Y_test = pd.read_csv(Y_TEST_PATH, index_col=0)

# Ensure both datasets match in size
num_samples = min(100, len(X_test), len(Y_test))  # Test only the first 100 samples
X_test_subset = X_test.iloc[:num_samples]
Y_test_subset = Y_test.iloc[:num_samples]

# Store predictions and true labels
predictions = []
true_labels = Y_test_subset.values.flatten()  # Convert to 1D array

headers = {"Content-Type": "application/json"}

# Loop through first 100 samples and send them to API
for i in range(num_samples):
    sample_data = X_test_subset.iloc[i].to_dict()
    formatted_data = {feature: sample_data[feature] for feature in X_test_subset.columns}

    response = requests.post(API_URL, data=json.dumps(formatted_data), headers=headers)

    if response.status_code == 200:
        prediction = response.json().get("prediction", None)
        predictions.append(prediction)
    else:
        predictions.append(None)  # If request fails, store None

# Convert lists to numpy arrays
predictions = np.array(predictions, dtype=np.float32)
true_labels = np.array(true_labels, dtype=np.float32)

# Compute accuracy (ignoring None values if any API calls failed)
valid_indices = ~np.isnan(predictions)
accuracy = np.mean(predictions[valid_indices] == true_labels[valid_indices])

# Print results
print(f"✅ Accuracy on first {num_samples} samples: {accuracy:.4f}")

# Save results as a DataFrame for debugging
results_df = pd.DataFrame({"True Label": true_labels, "Prediction": predictions})
print(results_df.head(20))  # Show first 20 rows
