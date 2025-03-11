import os
import pickle
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV")

if DEPLOYMENT_ENV is None:  # Local environment
    env_path = Path(__file__).parent.parent / "config" / ".env"
    print(f"Loading environment from: {env_path}")
    load_dotenv(env_path)

DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_FILE_NAME = os.environ.get("MODEL_FILE_NAME")  

# Adjust the S3 key for cloud deployment
S3_MODEL_PATH = f"production/{MODEL_FILE_NAME}" if DEPLOYMENT_ENV == "cloud" else MODEL_FILE_NAME

# Define model path based on environment
if DEPLOYMENT_ENV == "local":
    MODEL_PATH = f"../models/{MODEL_FILE_NAME}"
    print(f"Using local model path: {MODEL_PATH}")
else:
    MODEL_PATH = f"/tmp/{MODEL_FILE_NAME}"  # Temporary storage for S3 download

# Initialize S3 client
s3_client = boto3.client("s3")


def check_s3_model_exists():
    """Check if the model file exists in S3 before downloading."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=S3_MODEL_PATH)
        print(f"✅ Model file '{S3_MODEL_PATH}' exists in S3 bucket '{S3_BUCKET_NAME}'.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"❌ Model file '{S3_MODEL_PATH}' does NOT exist in S3 bucket '{S3_BUCKET_NAME}'.")
        else:
            print(f"❌ Error checking S3: {str(e)}")
        return False
    except NoCredentialsError:
        print("❌ AWS credentials not found. Make sure IAM permissions are set.")
        return False


def load_or_download_model():
    """Load the model from local storage or download from S3 if needed."""
    
    # Check if model already exists locally
    if os.path.exists(MODEL_PATH):
        print(f"📂 Loading model from {MODEL_PATH}...")
    else:
        if not S3_BUCKET_NAME:
            raise ValueError("❌ No S3 bucket specified for model storage.")

        if not check_s3_model_exists():
            raise FileNotFoundError(f"❌ Model file '{S3_MODEL_PATH}' not found in S3. Check your bucket.")

        print(f"🌍 Downloading model from S3: {S3_BUCKET_NAME}/{S3_MODEL_PATH} -> {MODEL_PATH}")
        try:
            with open(MODEL_PATH, "wb") as model_file:
                s3_client.download_fileobj(S3_BUCKET_NAME, S3_MODEL_PATH, model_file)
            print("✅ Model successfully downloaded from S3!")
        except (NoCredentialsError, ClientError) as e:
            raise RuntimeError(f"❌ Failed to download model from S3: {str(e)}")

    # Load the Random Forest model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model successfully loaded!")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {str(e)}")


# Load the model at import time
model = load_or_download_model()
