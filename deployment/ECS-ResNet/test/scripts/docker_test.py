import requests
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent  
ROOT_DIR = SCRIPT_DIR.parent.parent  
DATA_DIR = ROOT_DIR / "test" / "data"  
IMAGE_DIR = DATA_DIR / "image"

# Load .env file
env_path = ROOT_DIR / "config" / ".env"
load_dotenv(env_path)

# Ensure the container's API URL
API_URL = "http://localhost:8080/predict"  # Target the running Docker container

# Configure logger
LOG_FILE = ROOT_DIR / "logs" / "api_predictions.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Ensure IMAGE_DIR exists
if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Image directory not found at {IMAGE_DIR}")

# Process and send each image
for image_file in IMAGE_DIR.glob("*.jpeg"):
    with open(image_file, "rb") as img_file:
        files = {"image": (image_file.name, img_file, "image/jpeg")}
        response = requests.post(API_URL, files=files)
    
    # Handle response
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction", "Unknown")
        logger.info(f"Image: {image_file.name}, Prediction: {prediction}")
        print(f"Image: {image_file.name}, Prediction: {prediction}")
    else:
        logger.error(f"Failed to get a response for {image_file.name}. Status Code: {response.status_code}")
        print(f"Error for {image_file.name}: {response.status_code}")
