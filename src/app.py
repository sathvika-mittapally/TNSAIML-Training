"""
Flask API for AQI prediction.
Exposes a /predict endpoint that accepts JSON with the same feature columns used in training.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simple health check endpoint
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "aqi_rf.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Wind_Speed",
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # Validate request
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400
    missing = [c for c in FEATURE_COLS if c not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400
    # Build DataFrame for model prediction
    input_df = pd.DataFrame([data], columns=FEATURE_COLS)
    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"predicted_AQI": round(float(pred), 2)})

if __name__ == "__main__":
    # Enable Flask's reloader when run from VS Code debugger
    app.run(host="127.0.0.1", port=5000, debug=True)
