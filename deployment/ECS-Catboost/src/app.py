from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging
import os
from model_loader import model  



# Create logs directory if it doesn't exist
LOG_DIR = "../logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging (file + console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for frontend integration
CORS(app, resources={r"/*": {"origins": ["https://kshannon-ucsd.github.io",
                                         "https://kshannon-ucsd.github.io/24wi-dsc180-profile/",
                                         "https://kshannon-ucsd.github.io/24wi-dsc180-profile",
                                         "http://localhost:3000"]}})

# Expected feature order (explicitly defined)
EXPECTED_FEATURE_ORDER = [
            "heart_rate", "sbp", "mbp", "resp_rate", "temperature",
            "platelet", "wbc", "bands", "lactate", "inr",
            "ptt", "creatinine", "bilirubin", "pneumonia"
        ]

@app.route("/", methods=["GET"])
def home():
    """Home endpoint to verify API is running."""
    return jsonify({"message": "ECS Model API with CatBoost is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to get predictions from the CatBoost model."""
    try:
        data = request.get_json()
        logging.info(f"Received JSON Data: {data}")

        if not data:
            return jsonify({"error": "Empty request payload"}), 400

        # Ensure input is a dictionary (single row of features)
        if not isinstance(data, dict):
            return jsonify({"error": "Input should be a dictionary"}), 400


        # Check for missing features
        missing_features = [feature for feature in EXPECTED_FEATURE_ORDER if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert input JSON to DataFrame, enforcing column order
        features_df = pd.DataFrame([data])[EXPECTED_FEATURE_ORDER]  


        features_df = features_df.loc[0:0] 

        # Log for debugging
        logging.info(f"Final input DataFrame before prediction:\n{features_df}")

        # **Predict using CatBoost**
        prediction = int(model.predict(features_df)[0])

        logging.info(f"Generated prediction: {prediction}")
        return jsonify({"prediction": prediction})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=8080)
