"""
AQI Prediction Model Training Script
Trains a Random Forest model on synthetic environmental data to predict AQI.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

np.random.seed(42)

def generate_synthetic_data(n_samples=5000):
    """
    Generate realistic synthetic environmental data for AQI prediction.
    Features: PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed
    """
    # Generate pollutant concentrations (realistic ranges)
    pm25 = np.random.exponential(scale=35, size=n_samples).clip(0, 500)
    pm10 = pm25 * np.random.uniform(1.2, 2.5, size=n_samples) + np.random.normal(0, 10, size=n_samples)
    pm10 = pm10.clip(0, 600)
    
    no2 = np.random.exponential(scale=40, size=n_samples).clip(0, 400)
    so2 = np.random.exponential(scale=20, size=n_samples).clip(0, 300)
    co = np.random.exponential(scale=2, size=n_samples).clip(0, 30)
    o3 = np.random.exponential(scale=50, size=n_samples).clip(0, 300)
    
    # Meteorological features
    temperature = np.random.normal(28, 10, size=n_samples).clip(-10, 50)
    humidity = np.random.normal(60, 20, size=n_samples).clip(10, 100)
    wind_speed = np.random.exponential(scale=8, size=n_samples).clip(0, 50)
    
    # Calculate AQI based on EPA breakpoints (simplified but realistic)
    # AQI is primarily driven by PM2.5, but other pollutants contribute
    aqi_pm25 = calculate_pm25_aqi(pm25)
    aqi_pm10 = calculate_pm10_aqi(pm10)
    aqi_no2 = no2 * 0.8
    aqi_so2 = so2 * 0.7
    aqi_co = co * 15
    aqi_o3 = o3 * 0.6
    
    # AQI is the maximum of individual sub-indices (EPA standard)
    aqi = np.maximum.reduce([aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co, aqi_o3])
    
    # Add weather influence (temperature inversions trap pollutants)
    weather_factor = 1 + 0.005 * (humidity - 50) - 0.01 * wind_speed + 0.003 * np.abs(temperature - 25)
    aqi = (aqi * weather_factor).clip(0, 500)
    
    # Add some noise
    aqi += np.random.normal(0, 5, size=n_samples)
    aqi = aqi.clip(0, 500).astype(int)
    
    data = pd.DataFrame({
        'PM2.5': np.round(pm25, 2),
        'PM10': np.round(pm10, 2),
        'NO2': np.round(no2, 2),
        'SO2': np.round(so2, 2),
        'CO': np.round(co, 2),
        'O3': np.round(o3, 2),
        'Temperature': np.round(temperature, 1),
        'Humidity': np.round(humidity, 1),
        'Wind_Speed': np.round(wind_speed, 1),
        'AQI': aqi
    })
    
    return data


def calculate_pm25_aqi(pm25):
    """Calculate AQI sub-index from PM2.5 concentration using EPA breakpoints."""
    aqi = np.zeros_like(pm25)
    
    # Good: 0-12.0 -> 0-50
    mask = pm25 <= 12.0
    aqi[mask] = pm25[mask] * (50 / 12.0)
    
    # Moderate: 12.1-35.4 -> 51-100
    mask = (pm25 > 12.0) & (pm25 <= 35.4)
    aqi[mask] = 51 + (pm25[mask] - 12.1) * (49 / 23.3)
    
    # Unhealthy for Sensitive: 35.5-55.4 -> 101-150
    mask = (pm25 > 35.4) & (pm25 <= 55.4)
    aqi[mask] = 101 + (pm25[mask] - 35.5) * (49 / 19.9)
    
    # Unhealthy: 55.5-150.4 -> 151-200
    mask = (pm25 > 55.4) & (pm25 <= 150.4)
    aqi[mask] = 151 + (pm25[mask] - 55.5) * (49 / 94.9)
    
    # Very Unhealthy: 150.5-250.4 -> 201-300
    mask = (pm25 > 150.4) & (pm25 <= 250.4)
    aqi[mask] = 201 + (pm25[mask] - 150.5) * (99 / 99.9)
    
    # Hazardous: 250.5+ -> 301-500
    mask = pm25 > 250.4
    aqi[mask] = 301 + (pm25[mask] - 250.5) * (199 / 249.5)
    
    return aqi.clip(0, 500)


def calculate_pm10_aqi(pm10):
    """Calculate AQI sub-index from PM10 concentration."""
    aqi = np.zeros_like(pm10)
    
    mask = pm10 <= 54
    aqi[mask] = pm10[mask] * (50 / 54)
    
    mask = (pm10 > 54) & (pm10 <= 154)
    aqi[mask] = 51 + (pm10[mask] - 55) * (49 / 99)
    
    mask = (pm10 > 154) & (pm10 <= 254)
    aqi[mask] = 101 + (pm10[mask] - 155) * (49 / 99)
    
    mask = (pm10 > 254) & (pm10 <= 354)
    aqi[mask] = 151 + (pm10[mask] - 255) * (49 / 99)
    
    mask = pm10 > 354
    aqi[mask] = 201 + (pm10[mask] - 355) * (99 / 99)
    
    return aqi.clip(0, 500)


def train_and_save_model():
    """Train the AQI prediction model and save artifacts."""
    print("=" * 60)
    print("  AQI Prediction Model Training")
    print("=" * 60)
    
    # Generate data
    print("\n[1/5] Generating synthetic environmental data...")
    data = generate_synthetic_data(n_samples=8000)
    print(f"  → Generated {len(data)} samples")
    print(f"  → AQI Range: {data['AQI'].min()} - {data['AQI'].max()}")
    print(f"  → AQI Mean: {data['AQI'].mean():.1f}")
    
    # Save sample data for reference
    data.to_csv(os.path.join(os.path.dirname(__file__), 'sample_data.csv'), index=False)
    
    # Prepare features and target
    feature_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']
    X = data[feature_columns].values
    y = data['AQI'].values
    
    # Split data
    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  → Training set: {len(X_train)} samples")
    print(f"  → Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n[4/5] Training models...")
    
    # Random Forest
    print("  → Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Gradient Boosting
    print("  → Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Evaluate models
    print("\n[5/5] Evaluating models...")
    
    rf_metrics = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    }
    
    gb_metrics = {
        'MAE': mean_absolute_error(y_test, gb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'R2': r2_score(y_test, gb_pred)
    }
    
    print(f"\n  Random Forest:      MAE={rf_metrics['MAE']:.2f}  RMSE={rf_metrics['RMSE']:.2f}  R²={rf_metrics['R2']:.4f}")
    print(f"  Gradient Boosting:  MAE={gb_metrics['MAE']:.2f}  RMSE={gb_metrics['RMSE']:.2f}  R²={gb_metrics['R2']:.4f}")
    
    # Select best model
    best_model_name = 'Random Forest' if rf_metrics['R2'] >= gb_metrics['R2'] else 'Gradient Boosting'
    best_model = rf_model if rf_metrics['R2'] >= gb_metrics['R2'] else gb_model
    best_metrics = rf_metrics if rf_metrics['R2'] >= gb_metrics['R2'] else gb_metrics
    
    print(f"\n  ✓ Best Model: {best_model_name}")
    
    # Feature importance
    importances = best_model.feature_importances_
    feature_importance = dict(zip(feature_columns, importances.tolist()))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  Feature Importance:")
    for feat, imp in sorted_features:
        bar = '█' * int(imp * 50)
        print(f"    {feat:15s} {imp:.4f} {bar}")
    
    # Save model artifacts
    model_dir = os.path.dirname(__file__)
    
    joblib.dump(best_model, os.path.join(model_dir, 'aqi_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    model_info = {
        'model_name': best_model_name,
        'features': feature_columns,
        'feature_importance': feature_importance,
        'metrics': best_metrics,
        'aqi_categories': {
            'Good': [0, 50],
            'Moderate': [51, 100],
            'Unhealthy for Sensitive Groups': [101, 150],
            'Unhealthy': [151, 200],
            'Very Unhealthy': [201, 300],
            'Hazardous': [301, 500]
        },
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"  Model saved successfully!")
    print(f"  → aqi_model.pkl")
    print(f"  → scaler.pkl")
    print(f"  → model_info.json")
    print(f"{'=' * 60}")
    
    return best_model, scaler, model_info


if __name__ == '__main__':
    train_and_save_model()
