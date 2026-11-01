"""
train_model.py
Train a simple RandomForestRegressor to predict AQI from environmental features.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "air_quality_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_rf.pkl")

def load_data(path):
    df = pd.read_csv(path)
    # Features used for prediction (exclude target and categorical label)
    feature_cols = [
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
    X = df[feature_cols]
    y = df["AQI"]
    return X, y

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model trained. MAE: {mae:.2f}, R²: {r2:.3f}")
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
