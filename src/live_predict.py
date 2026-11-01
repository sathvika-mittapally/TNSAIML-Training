"""
live_predict.py
Utility script to fetch current environmental measurements from OpenWeatherMap (or another API) and obtain an AQI prediction using the trained model.
Replace `YOUR_API_KEY` with a valid key from https://openweathermap.org/api.
"""

import os
import json
import joblib
import requests
import pandas as pd

# ==== CONFIGURATION ====
API_KEY = "YOUR_API_KEY"  # <-- put your OpenWeatherMap API key here
CITY = "New York"        # city name or latitude/longitude coordinates
UNITS = "metric"         # 'metric' for Celsius, 'imperial' for Fahrenheit

# Model path (relative to this script)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "aqi_rf.pkl")

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

def fetch_weather(api_key, city, units="metric"):
    """Fetch current weather data from OpenWeatherMap.
    Returns a dict with temperature, humidity, wind speed, and a placeholder for pollutants.
    The free tier does NOT provide pollutant concentrations, so you will need to supply them
    (or use a different API such as AirNow that returns AQI components)."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return {
        "Temperature": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Wind_Speed": data["wind"]["speed"],
        # The following pollutant fields are set to 0 as placeholders – replace with real values if you have them
        "PM2.5": 0,
        "PM10": 0,
        "NO2": 0,
        "SO2": 0,
        "CO": 0,
        "O3": 0,
    }

def predict_aqi(features, model_path=MODEL_PATH):
    model = joblib.load(model_path)
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    pred = model.predict(df)[0]
    return round(float(pred), 2)

def main():
    if API_KEY == "YOUR_API_KEY":
        raise ValueError("Please replace API_KEY with your OpenWeatherMap API key.")
    print(f"Fetching current weather for {CITY} …")
    env = fetch_weather(API_KEY, CITY, UNITS)
    print("Environmental data:")
    print(json.dumps(env, indent=2))
    aqi = predict_aqi(env)
    print(f"\nPredicted AQI (based on current weather & placeholder pollutants): {aqi}")

if __name__ == "__main__":
    main()
