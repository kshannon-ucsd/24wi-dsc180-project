import io
import json
import warnings
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from model_loader import model
from flask_cors import CORS

# Suppress warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)
# Configure CORS properly - specify the exact origin
CORS(app, resources={r"/*": {"origins": ["https://kshannon-ucsd.github.io",
                                          "https://kshannon-ucsd.github.io/24wi-dsc180-profile/",
                                            "https://kshannon-ucsd.github.io/24wi-dsc180-profile",
                                              "http://localhost:3000"]}})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ECS Model API with CNN model is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    # Load image from request properly
    file = request.files["image"]
    # Convert file to in-memory format before loading
    img = image.load_img(io.BytesIO(file.read()), target_size=(256, 256))
    img = tf.image.central_crop(img, 224 / 256)


    img_array = image.img_to_array(img)

    img = preprocess_input(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    
    # Debugging: Print Image Shape Before Prediction
    print("Image shape after processing:", img_array.shape)
    # Predict the result
    predictions = model.predict(img_array)
    predicted_class = (predictions > 0.5).astype(int)
    # Use Flask's jsonify which properly handles CORS when Flask-CORS is configured
    return jsonify({
        "prediction": int(predicted_class[0][0])
    })



if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=8080)