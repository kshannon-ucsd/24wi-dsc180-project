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

# Load feature data (X_test.csv)
print(f"📊 Loading test data from: {X_TEST_PATH}")
X_test = pd.read_csv(X_TEST_PATH,index_col=0)


# Convert the first row of test data into JSON with explicit feature names
sample_data = X_test.iloc[0].to_dict()

# Ensure feature names are explicitly passed in JSON
formatted_data = {feature: sample_data[feature] for feature in X_test.columns}

# Send POST request
headers = {"Content-Type": "application/json"}
response = requests.post(API_URL, data=json.dumps(formatted_data), headers=headers)

# Print response
print(f"✅ Response Code: {response.status_code}")
print(f"📝 Response JSON: {response.json()}")
