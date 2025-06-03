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

# Load test data
print(f"📊 Loading test data from: {X_TEST_PATH}")
X_test = pd.read_csv(X_TEST_PATH)

print(f"📊 Loading y_test from: {Y_TEST_PATH}")
y_test = pd.read_csv(Y_TEST_PATH, index_col=0)

# Ensure y_test has the correct format
if "sepsis" not in y_test.columns:
    raise ValueError("❌ y_test.csv must have a 'sepsis' column indicating class (0 or 1).")

# Convert sepsis column to integers (handle string issues)
y_test["sepsis"] = y_test["sepsis"].astype(str).str.strip().astype(int)

# Reset indices to ensure alignment
print("⚠️ Resetting indices for both X_test and y_test...")
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Debugging: Print available indices
print("🔍 X_test first 10 indices:", X_test.index.tolist()[:10])
print("🔍 y_test first 10 indices:", y_test.index.tolist()[:10])

# Find indices where sepsis == 0 (negative cases)
negative_indices = y_test[y_test["sepsis"] == 0].index.tolist()

if not negative_indices:
    raise ValueError("❌ No negative cases found in y_test!")

negative_index = negative_indices[0]  # Select the first available negative sample
print(f"⚠️ Using negative test case at index: {negative_index}")

# Define required feature names (ensure they match API expectation)
REQUIRED_FEATURES = [
    "bilirubin", "creatinine", "heart_rate", "inr", "mbp",
    "platelet", "ptt", "resp_rate", "sbp", "wbc", "pneumonia"
]

# Check if CSV has all required columns
missing_features = [feature for feature in REQUIRED_FEATURES if feature not in X_test.columns]
if missing_features:
    raise ValueError(f"❌ CSV is missing required features: {missing_features}")

# Extract the negative test sample using .iloc[] since we reset the index
sample_data = X_test.iloc[negative_index].to_dict()

# Send POST request
headers = {"Content-Type": "application/json"}
print(f"🚀 Sending negative test request to: {API_URL}")
response = requests.post(API_URL, data=json.dumps(sample_data), headers=headers)

# Print response
print(f"✅ Response Code: {response.status_code}")
print(f"📝 Response JSON: {response.json()}")
