import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  
ROOT_DIR = SCRIPT_DIR.parent.parent  
DATA_DIR = ROOT_DIR / "test" / "data"  

# Load .env file from config
env_path = ROOT_DIR / "config" / ".env"
print("Loading environment from:", env_path)
load_dotenv(env_path)

# Load test data paths from .env
X_TEST_PATH = os.getenv("X_TEST_PATH")
Y_TEST_PATH = os.getenv("Y_TEST_PATH")

# Convert them to absolute paths
X_TEST_PATH = (DATA_DIR / X_TEST_PATH).resolve()
Y_TEST_PATH = (DATA_DIR / Y_TEST_PATH).resolve()

DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV") 
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "PNG").upper()

if IMAGE_FORMAT not in ["PNG", "JPEG"]:
    raise ValueError("Unsupported IMAGE_FORMAT. Use 'PNG' or 'JPEG'.")

MIME_TYPE = f"image/{IMAGE_FORMAT.lower()}"

if DEPLOYMENT_ENV == "local":
    API_URL = "http://127.0.0.1:8080/predict"
    print("this is from local")
else:
    API_URL = os.environ.get("API_URL")
    print("this is from cloud")

# Load test data
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("y_test first result:", y_test[0])

# Convert first test image to a PIL image
image_array = (X_test[0] * 255).astype(np.uint8)  # Scale if needed
image = Image.fromarray(image_array)

# Save as a temporary in-memory file
image_buffer = BytesIO()
image.save(image_buffer, format=IMAGE_FORMAT)
image_buffer.seek(0)

# Send the image to the API and Cloud Deployment Test
files = {"image": (f"image.{IMAGE_FORMAT.lower()}", image_buffer, MIME_TYPE)}
response = requests.post(API_URL, files=files)

print(response.json())
