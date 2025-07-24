"""
Offline Forecast Module for VayuDrishti Dashboard
Provides local forecasting when API server is unavailable
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import sys
from pathlib import Path

# Unicode print fix
def safe_print(text):
    """Safe print that handles Unicode encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii') if isinstance(text, str) else str(text))

class OfflineForecast:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the XGBoost model from disk"""
        try:
            # Try multiple possible model locations
            possible_paths = [
                Path(__file__).parent.parent / "best_model.pkl",  # Project root
                Path(__file__).parent.parent / "models" / "best_model.pkl",  # Models directory
                Path("best_model.pkl"),  # Current directory
                Path("models/best_model.pkl"),  # Models subdirectory
                Path("../best_model.pkl"),  # Parent directory
                Path("../models/best_model.pkl")  # Parent models directory
            ]
            
            for model_path in possible_paths:
                if model_path.exists():
                    self.model = joblib.load(model_path)
                    self.model_loaded = True
                    safe_print(f"✅ Model loaded from {model_path}")
                    return
            
            safe_print("❌ Model file 'best_model.pkl' not found in any expected location")
            safe_print("Expected locations:")
            for path in possible_paths:
                safe_print(f"  - {path}")
                
        except Exception as e:
            safe_print(f"❌ Error loading model: {e}")
    
    def pm25_to_cpcb_aqi(self, pm25):
        """Convert PM2.5 to CPCB AQI"""
        if pm25 <= 30:
            return pm25 * 50 / 30, "Good"
        elif pm25 <= 60:
            return 50 + (pm25 - 30) * 50 / 30, "Satisfactory"
        elif pm25 <= 90:
            return 100 + (pm25 - 60) * 100 / 30, "Moderate"
        elif pm25 <= 120:
            return 200 + (pm25 - 90) * 100 / 30, "Poor"
        elif pm25 <= 250:
            return 300 + (pm25 - 120) * 100 / 130, "Very Poor"
        else:
            return min(500, 400 + (pm25 - 250) * 100 / 130), "Severe"
    
    def generate_baseline_features(self, lat, lon, date):
        """Generate baseline features for a location and date"""
        # Seasonal patterns
        month = date.month
        hour = date.hour
        
        if month in [12, 1, 2]:
            season = 1  # Winter
            temp_base = 15
            humidity_base = 70
        elif month in [3, 4, 5]:
            season = 2  # Spring
            temp_base = 25
            humidity_base = 60
        elif month in [6, 7, 8, 9]:
            season = 3  # Monsoon
            temp_base = 30
            humidity_base = 85
        else:
            season = 4  # Post-monsoon
            temp_base = 20
            humidity_base = 65
        
        # Location-based adjustments
        # Delhi/NCR area (high pollution)
        if 28.0 <= lat <= 29.0 and 76.5 <= lon <= 77.5:
            aod_base = 0.8
            wind_base = 3.0
            blh_base = 600
        # Mumbai area (coastal, moderate)
        elif 18.8 <= lat <= 19.3 and 72.7 <= lon <= 73.2:
            aod_base = 0.6
            wind_base = 5.0
            blh_base = 900
        # Bangalore area (elevated, better)
        elif 12.8 <= lat <= 13.2 and 77.4 <= lon <= 77.8:
            aod_base = 0.5
            wind_base = 4.0
            blh_base = 1000
        # Kolkata area (high humidity, moderate pollution)
        elif 22.3 <= lat <= 22.8 and 88.2 <= lon <= 88.5:
            aod_base = 0.7
            wind_base = 3.5
            blh_base = 700
        else:
            # Default for other locations
            aod_base = 0.6
            wind_base = 4.0
            blh_base = 800
        
        # Add some randomness for realistic variation
        np.random.seed(int(lat * lon * 1000) + date.timetuple().tm_yday)
        
        # Generate all 12 features required by the model
        features = {
            'aod_550': max(0.1, min(2.0, aod_base + np.random.normal(0, 0.1))),
            't2m_celsius': temp_base + np.random.normal(0, 3),
            'wind_speed_10m': max(0.5, wind_base + np.random.normal(0, 1)),
            'r2m': max(30, min(95, humidity_base + np.random.normal(0, 10))),
            'blh': max(200, blh_base + np.random.normal(0, 200)),
            'lat_cos': np.cos(np.radians(lat)),
            'lat_sin': np.sin(np.radians(lat)),
            'lon_cos': np.cos(np.radians(lon)),
            'lon_sin': np.sin(np.radians(lon)),
            'hour': hour,
            'month': month,
            'season': season
        }
        
        return features
    
    def generate_forecast(self, latitude, longitude, start_date, forecast_days):
        """Generate offline forecast"""
        if not self.model_loaded or self.model is None:
            raise Exception("Model not loaded. Cannot generate offline forecast.")
        
        forecasts = []
        
        # Convert date to datetime if necessary
        if hasattr(start_date, 'hour'):
            current_date = start_date
        else:
            # If it's a date object, convert to datetime with noon as default hour
            current_date = datetime.combine(start_date, datetime.min.time().replace(hour=12))
        
        for day in range(forecast_days):
            # Generate features for this day
            features = self.generate_baseline_features(latitude, longitude, current_date)
            
            # Add day-to-day variation
            if day > 0:
                # Pollution tends to accumulate over consecutive days
                pollution_trend = min(0.3, day * 0.1)
                features['aod_550'] += pollution_trend
                features['aod_550'] = min(2.0, features['aod_550'])
            
            # Create feature array for prediction (12 features in correct order)
            feature_array = np.array([[
                features['aod_550'],
                features['t2m_celsius'], 
                features['wind_speed_10m'],
                features['r2m'],
                features['blh'],
                features['lat_cos'],
                features['lat_sin'],
                features['lon_cos'],
                features['lon_sin'],
                features['hour'],
                features['month'],
                features['season']
            ]])
            
            # Make prediction
            pm25_prediction = self.model.predict(feature_array)[0]
            
            # Ensure realistic bounds
            pm25_prediction = max(5, min(500, pm25_prediction))
            
            # Convert to AQI
            aqi, category = self.pm25_to_cpcb_aqi(pm25_prediction)
            
            forecast_item = {
                'date': current_date.isoformat(),
                'pm2_5': round(pm25_prediction, 1),
                'aqi': int(aqi),
                'category': category,
                'temperature': round(features['t2m_celsius'], 1),
                'humidity': round(features['r2m'], 1),
                'wind_speed': round(features['wind_speed_10m'], 1)
            }
            
            forecasts.append(forecast_item)
            current_date += timedelta(days=1)
        
        return {
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'forecast': forecasts,
            'model_info': {
                'type': 'offline_fallback',
                'accuracy': 'approximate',
                'note': 'Generated using local model with baseline meteorological assumptions'
            }
        }
    
    def predict_single(self, feature_array):
        """Make a single prediction using the loaded model"""
        if not self.model_loaded or self.model is None:
            raise Exception("Model not loaded. Cannot make prediction.")
        
        prediction = self.model.predict(feature_array)[0]
        return max(5, min(500, prediction))  # Ensure realistic bounds
    
    def predict_pm25_offline(self, input_features: dict) -> dict:
        """
        Simplified PM2.5 prediction function for hackathon demo
        
        Args:
            input_features (dict): Features like {'aod_550': 0.6, 't2m_celsius': 25, ...}
            
        Returns:
            dict: {'pm25': float, 'aqi': int, 'health_category': str, 'health_message': str}
        """
        if not self.model_loaded or self.model is None:
            return {
                'pm25': 0.0,
                'aqi': 0,
                'health_category': 'Error',
                'health_message': 'Model not available'
            }
        
        try:
            # Extract features in the correct order for the model (12 features)
            lat = input_features.get('latitude', 28.6)
            lon = input_features.get('longitude', 77.2)
            hour = input_features.get('hour', 12)
            month = input_features.get('month', 6)
            
            feature_array = np.array([[
                input_features.get('aod_550', 0.6),
                input_features.get('t2m_celsius', 25.0),
                input_features.get('wind_speed_10m', 4.0),
                input_features.get('r2m', 65.0),
                input_features.get('blh', 800.0),
                np.cos(np.radians(lat)),  # lat_cos
                np.sin(np.radians(lat)),  # lat_sin
                np.cos(np.radians(lon)),  # lon_cos
                np.sin(np.radians(lon)),  # lon_sin
                hour,
                month,
                input_features.get('season', 2)
            ]])
            
            # Make prediction
            pm25_prediction = self.model.predict(feature_array)[0]
            pm25_prediction = max(5, min(500, pm25_prediction))  # Realistic bounds
            
            # Convert to AQI and health category
            aqi_val, health_cat = self.pm25_to_cpcb_aqi(pm25_prediction)
            
            # Generate health message
            health_messages = {
                "Good": "🟢 Excellent air quality! Perfect for outdoor activities.",
                "Satisfactory": "🟡 Good air quality with minor concern for sensitive individuals.",
                "Moderate": "🟠 Moderate air quality. Sensitive individuals may experience symptoms.",
                "Poor": "🔴 Poor air quality. Health effects may be experienced by everyone.",
                "Very Poor": "🟣 Very poor air quality. Serious health effects for everyone.",
                "Severe": "🔴 Severe air quality emergency! Stay indoors."
            }
            
            return {
                'pm25': round(pm25_prediction, 1),
                'aqi': int(aqi_val),
                'health_category': health_cat,
                'health_message': health_messages.get(health_cat, "Unknown air quality status")
            }
            
        except Exception as e:
            safe_print(f"Prediction error: {e}")
            return {
                'pm25': 0.0,
                'aqi': 0,
                'health_category': 'Error',
                'health_message': f'Prediction failed: {str(e)}'
            }

# Create global instance
offline_forecast = OfflineForecast()
