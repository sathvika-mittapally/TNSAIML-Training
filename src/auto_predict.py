"""
auto_predict.py
Runs in the background (via Antigravity schedule) and writes a prediction
to logs/auto_predictions.csv every time it is invoked.
It re‑uses the trained model and can either fetch live weather data
(using the same helper from live_predict.py) or generate a random
environmental sample if no API key is supplied.
"""

import os
import json
import csv
import datetime
import joblib
import pandas as pd
import numpy as np
import requests

# -------------------------------------------------------------------
# Configuration ------------------------------------------------------
# -------------------------------------------------------------------
# Path to the trained model (relative to this file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "aqi_rf.pkl")

# OpenWeatherMap settings – leave KEY empty to skip live fetch
OWM_API_KEY = "YOUR_API_KEY"  # <-- replace with a real key if you want live weather
OWM_CITY = "New York"
OWM_UNITS = "metric"

# Output CSV file (will be created under the project "logs" folder)
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "auto_predictions.csv")

# -------------------------------------------------------------------
# Helper – generate a synthetic sample (same logic as generate_dataset.py)
# -------------------------------------------------------------------
def random_sample():
    """Create a random environmental record – used when no API key is set.
    The distribution mirrors the synthetic generator used for training.
    """
    # Weather variables
    temperature = np.clip(np.random.normal(26, 7.5), 2, 48)
    humidity = np.clip(np.random.normal(55, 18), 15, 98)
    wind = np.clip(np.random.lognormal(2.2, 0.55), 1, 45)

    # Pollutants – use the same stochastic formulas as in generate_dataset.py
    stagnation = (1 / (wind + 1)) * 15 + (humidity / 100) * 1.5
    pm25 = np.clip(np.random.exponential(35) * (stagnation * 0.4) + np.random.normal(5, 3), 3, 450)
    pm10 = np.clip(pm25 * np.random.uniform(1.3, 2.4) + np.random.normal(8, 4), 8, 600)
    no2 = np.clip(np.random.gamma(3, 12) * (1 + (35 - temperature) * 0.015), 2, 250)
    so2 = np.clip(np.random.gamma(2.5, 8), 1, 180)
    co = np.clip(np.random.gamma(2, 2.5) * (stagnation * 0.25), 0.2, 38)
    o3 = np.clip(temperature * 1.8 + np.random.beta(2, 2) * 40 + no2 * 0.15 + np.random.normal(0, 6), 5, 280)

    return {
        "Temperature": round(float(temperature), 1),
        "Humidity": round(float(humidity), 1),
        "Wind_Speed": round(float(wind), 1),
        "PM2.5": round(float(pm25), 2),
        "PM10": round(float(pm10), 2),
        "NO2": round(float(no2), 2),
        "SO2": round(float(so2), 2),
        "CO": round(float(co), 2),
        "O3": round(float(o3), 2),
    }

# -------------------------------------------------------------------
# Helper – fetch live weather (same as live_predict.py)
# -------------------------------------------------------------------
def fetch_weather(api_key, city, units="metric"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return {
        "Temperature": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Wind_Speed": data["wind"]["speed"],
        "PM2.5": 0,  # placeholders – replace with real pollutant source if available
        "PM10": 0,
        "NO2": 0,
        "SO2": 0,
        "CO": 0,
        "O3": 0,
    }

# -------------------------------------------------------------------
# Core prediction routine
# -------------------------------------------------------------------
def predict(features, model_path=MODEL_PATH):
    model = joblib.load(model_path)
    df = pd.DataFrame([features], columns=[
        "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
        "Temperature", "Humidity", "Wind_Speed",
    ])
    return round(float(model.predict(df)[0]), 2)

# -------------------------------------------------------------------
# Main entry point – invoked by Antigravity scheduler
# -------------------------------------------------------------------
import time

# Configuration – interval in seconds between predictions (default 3600s = 1 hour)
INTERVAL_SECONDS = int(os.getenv('AUTO_PREDICT_INTERVAL', '3600'))

def main_loop():
    while True:
        try:
            # Choose data source
            if OWM_API_KEY and OWM_API_KEY != "YOUR_API_KEY":
                env = fetch_weather(OWM_API_KEY, OWM_CITY, OWM_UNITS)
            else:
                env = random_sample()

            aqi = predict(env)
            timestamp = datetime.datetime.now().isoformat()
            print(f"[{timestamp}] Auto‑prediction: AQI={aqi} | data={env}")
        except Exception as e:
            print(f"[{datetime.datetime.now().isoformat()}] Error during auto prediction: {e}")
        # Wait before next run
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()
