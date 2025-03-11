import os
import boto3
import tensorflow as tf
from tensorflow.keras.models import load_model
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from pathlib import Path

DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV")
# If DEPLOYMENT_ENV is none, then it is local, since I set up DEPLOYMENT_ENV as cloud on AWS 
if DEPLOYMENT_ENV is None:
    env_path = Path(__file__).parent.parent / "config" / ".env"
    print(env_path)
    load_dotenv(env_path)

DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV") 
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_FILE_NAME = os.environ.get("MODEL_FILE_NAME")

if DEPLOYMENT_ENV == "local":
    MODEL_PATH = f"../models/{MODEL_FILE_NAME}"
    print(MODEL_PATH)
else:
    MODEL_PATH = f"/tmp/{MODEL_FILE_NAME}"

s3_client = boto3.client("s3")

def configure_tensorflow():
    """Automatically selects GPU if available, otherwise uses CPU."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for GPUs to prevent full allocation
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detected. Using: {gpus}")
        except RuntimeError as e:
            print(f"⚠️ Error setting GPU memory growth: {e}")
    else:
        # Force TensorFlow to use only CPU if no GPU is available
        tf.config.set_visible_devices([], 'GPU')
        print("❌ No GPU detected. Running on CPU only.")

def check_s3_model_exists():
    """Check if the model file exists in S3 before downloading."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=MODEL_FILE_NAME)
        print(f"✅ Model file '{MODEL_FILE_NAME}' exists in S3 bucket '{S3_BUCKET_NAME}'.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"❌ Model file '{MODEL_FILE_NAME}' does NOT exist in S3 bucket '{S3_BUCKET_NAME}'.")
        else:
            print(f"❌ Error checking S3: {str(e)}")
        return False
    except NoCredentialsError:
        print("❌ AWS credentials not found. Make sure IAM permissions are set.")
        return False

def load_or_download_model():
    """Load the model from local storage or download from S3 if needed."""

    if os.path.exists(MODEL_PATH):

        print(f"📂 Loading model from {MODEL_PATH}...")
    else:

        if not S3_BUCKET_NAME:
            raise ValueError("❌ No S3 bucket specified for model storage.")

        if not check_s3_model_exists():
            raise FileNotFoundError(f"❌ Model file '{MODEL_FILE_NAME}' not found in S3. Check your bucket.")

        print(f"🌍 Downloading model from S3: {S3_BUCKET_NAME}/{MODEL_FILE_NAME} -> {MODEL_PATH}")
        with open(MODEL_PATH, "wb") as model_file:
            s3_client.download_fileobj(S3_BUCKET_NAME, MODEL_FILE_NAME, model_file)

    # Load the model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model successfully loaded!")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {str(e)}")

# Apply TensorFlow configuration before loading the model
configure_tensorflow()
# Load model at import time
model = load_or_download_model()