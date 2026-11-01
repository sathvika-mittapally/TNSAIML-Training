"""
AQI Prediction Flask API
Serves predictions from the trained ML model via REST endpoints.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import json
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Load model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
model = None
scaler = None
model_info = None


def load_model():
    """Load the trained model and scaler."""
    global model, scaler, model_info
    model_path = os.path.join(MODEL_DIR, 'aqi_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    
    if not os.path.exists(model_path):
        print("Model not found. Training a new model...")
        from model.train_model import train_and_save_model
        train_and_save_model()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    print(f"✓ Model loaded: {model_info['model_name']}")
    print(f"  R² Score: {model_info['metrics']['R2']:.4f}")


def get_aqi_category(aqi_value):
    """Get AQI category and health recommendations based on value."""
    categories = {
        'Good': {
            'range': [0, 50],
            'color': '#00e400',
            'emoji': '🟢',
            'description': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'health_advice': 'Enjoy outdoor activities! Air quality is excellent.',
            'sensitive_advice': 'No precautions needed.'
        },
        'Moderate': {
            'range': [51, 100],
            'color': '#ffff00',
            'emoji': '🟡',
            'description': 'Air quality is acceptable. However, there may be a risk for some people.',
            'health_advice': 'Unusually sensitive people should consider reducing prolonged outdoor exertion.',
            'sensitive_advice': 'Consider reducing prolonged or heavy outdoor exertion if you experience symptoms.'
        },
        'Unhealthy for Sensitive Groups': {
            'range': [101, 150],
            'color': '#ff7e00',
            'emoji': '🟠',
            'description': 'Members of sensitive groups may experience health effects.',
            'health_advice': 'People with respiratory or heart conditions, children, and older adults should limit prolonged outdoor exertion.',
            'sensitive_advice': 'Reduce prolonged or heavy outdoor exertion. Take more breaks during outdoor activities.'
        },
        'Unhealthy': {
            'range': [151, 200],
            'color': '#ff0000',
            'emoji': '🔴',
            'description': 'Some members of the general public may experience health effects.',
            'health_advice': 'Everyone should reduce prolonged or heavy outdoor exertion. Sensitive groups should avoid all outdoor exertion.',
            'sensitive_advice': 'Avoid prolonged or heavy outdoor exertion. Move activities indoors or reschedule.'
        },
        'Very Unhealthy': {
            'range': [201, 300],
            'color': '#8f3f97',
            'emoji': '🟣',
            'description': 'Health alert: The risk of health effects is increased for everyone.',
            'health_advice': 'Everyone should avoid prolonged or heavy outdoor exertion. Sensitive groups should remain indoors.',
            'sensitive_advice': 'Remain indoors and keep activity levels low. Keep windows and doors closed.'
        },
        'Hazardous': {
            'range': [301, 500],
            'color': '#7e0023',
            'emoji': '⛔',
            'description': 'Health warning of emergency conditions: everyone is more likely to be affected.',
            'health_advice': 'Everyone should avoid all outdoor physical activities. Stay indoors with air purification if possible.',
            'sensitive_advice': 'Remain indoors with doors and windows sealed. Use air purifiers. Seek medical attention if experiencing symptoms.'
        }
    }
    
    aqi_int = int(round(aqi_value))
    for category, info in categories.items():
        if info['range'][0] <= aqi_int <= info['range'][1]:
            return {
                'category': category,
                'color': info['color'],
                'emoji': info['emoji'],
                'description': info['description'],
                'health_advice': info['health_advice'],
                'sensitive_advice': info['sensitive_advice']
            }
    
    # If AQI > 500, return hazardous
    info = categories['Hazardous']
    return {
        'category': 'Hazardous',
        'color': info['color'],
        'emoji': info['emoji'],
        'description': info['description'],
        'health_advice': info['health_advice'],
        'sensitive_advice': info['sensitive_advice']
    }


@app.route('/')
def serve_frontend():
    """Serve the frontend application."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict_aqi():
    """
    Predict AQI from environmental parameters.
    
    Expected JSON body:
    {
        "pm25": float,      # PM2.5 concentration (µg/m³)
        "pm10": float,      # PM10 concentration (µg/m³)
        "no2": float,       # NO2 concentration (ppb)
        "so2": float,       # SO2 concentration (ppb)
        "co": float,        # CO concentration (ppm)
        "o3": float,        # O3 concentration (ppb)
        "temperature": float, # Temperature (°C)
        "humidity": float,   # Relative Humidity (%)
        "wind_speed": float  # Wind Speed (km/h)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features in correct order
        required_fields = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity', 'wind_speed']
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Build feature array
        features = np.array([[
            float(data['pm25']),
            float(data['pm10']),
            float(data['no2']),
            float(data['so2']),
            float(data['co']),
            float(data['o3']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['wind_speed'])
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        aqi_prediction = model.predict(features_scaled)[0]
        aqi_prediction = max(0, min(500, round(aqi_prediction)))
        
        # Get category info
        category_info = get_aqi_category(aqi_prediction)
        
        # Get feature contributions (approximate with feature importance)
        feature_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
        importances = model_info['feature_importance']
        
        contributions = []
        for i, name in enumerate(feature_names):
            key = name.replace(' ', '_')
            contributions.append({
                'feature': name,
                'value': float(features[0][i]),
                'importance': importances.get(key, 0)
            })
        
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        response = {
            'success': True,
            'prediction': {
                'aqi': int(aqi_prediction),
                'category': category_info['category'],
                'color': category_info['color'],
                'emoji': category_info['emoji'],
                'description': category_info['description'],
                'health_advice': category_info['health_advice'],
                'sensitive_advice': category_info['sensitive_advice']
            },
            'contributions': contributions,
            'model_info': {
                'name': model_info['model_name'],
                'r2_score': model_info['metrics']['R2'],
                'mae': model_info['metrics']['MAE']
            }
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Return model metadata and performance metrics."""
    return jsonify({
        'success': True,
        'model': model_info
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    load_model()
    print("\n🌍 AQI Prediction API running at http://localhost:5000")
    print("📊 Frontend available at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
