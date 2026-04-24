"""
Offline forecast module for the Streamlit dashboard.

This module now delegates model loading and prediction to the canonical
`src.predict` bundle interface so dashboard inference matches training.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_bundle, predict_from_mapping  # noqa: E402


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode("ascii"))


class OfflineForecast:
    def __init__(self) -> None:
        self.bundle = None
        self.model_loaded = False
        self.load_model()

    def load_model(self) -> None:
        try:
            self.bundle = load_bundle()
            self.model_loaded = True
            safe_print("Loaded canonical PM2.5 model bundle")
        except Exception as exc:
            self.bundle = None
            self.model_loaded = False
            safe_print(f"Unable to load canonical model bundle: {exc}")

    def pm25_to_cpcb_aqi(self, pm25: float):
        if pm25 <= 30:
            return pm25 * 50 / 30, "Good"
        if pm25 <= 60:
            return 50 + (pm25 - 30) * 50 / 30, "Satisfactory"
        if pm25 <= 90:
            return 100 + (pm25 - 60) * 100 / 30, "Moderate"
        if pm25 <= 120:
            return 200 + (pm25 - 90) * 100 / 30, "Poor"
        if pm25 <= 250:
            return 300 + (pm25 - 120) * 100 / 130, "Very Poor"
        return min(500, 400 + (pm25 - 250) * 100 / 130), "Severe"

    def generate_baseline_features(self, lat: float, lon: float, date: datetime) -> dict:
        month = date.month
        hour = date.hour

        if month in [12, 1, 2]:
            season, temp_base, humidity_base = 1, 15, 70
        elif month in [3, 4, 5]:
            season, temp_base, humidity_base = 2, 25, 60
        elif month in [6, 7, 8, 9]:
            season, temp_base, humidity_base = 3, 30, 85
        else:
            season, temp_base, humidity_base = 4, 20, 65

        if 28.0 <= lat <= 29.0 and 76.5 <= lon <= 77.5:
            aod_base, wind_base, blh_base = 0.8, 3.0, 600
        elif 18.8 <= lat <= 19.3 and 72.7 <= lon <= 73.2:
            aod_base, wind_base, blh_base = 0.6, 5.0, 900
        elif 12.8 <= lat <= 13.2 and 77.4 <= lon <= 77.8:
            aod_base, wind_base, blh_base = 0.5, 4.0, 1000
        elif 22.3 <= lat <= 22.8 and 88.2 <= lon <= 88.5:
            aod_base, wind_base, blh_base = 0.7, 3.5, 700
        else:
            aod_base, wind_base, blh_base = 0.6, 4.0, 800

        np.random.seed(int(lat * lon * 1000) + date.timetuple().tm_yday)
        aod = max(0.1, min(2.0, aod_base + np.random.normal(0, 0.1)))
        temp = temp_base + np.random.normal(0, 3)
        wind = max(0.5, wind_base + np.random.normal(0, 1))
        humidity = max(30, min(95, humidity_base + np.random.normal(0, 10)))

        return {
            "latitude": lat,
            "longitude": lon,
            "aod_550": aod,
            "t2m_celsius": temp,
            "wind_speed_10m": wind,
            "r2m": humidity,
            "blh": max(200, blh_base + np.random.normal(0, 200)),
            "hour": hour,
            "month": month,
            "season": season,
        }

    def _predict(self, features: dict) -> float:
        if not self.model_loaded or self.bundle is None:
            raise RuntimeError("Model bundle not loaded")
        prediction = predict_from_mapping(features, self.bundle)
        return max(5.0, min(500.0, prediction))

    def generate_forecast(self, latitude, longitude, start_date, forecast_days):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate offline forecast.")

        current_date = start_date if hasattr(start_date, "hour") else datetime.combine(
            start_date, datetime.min.time().replace(hour=12)
        )
        forecasts = []

        for day in range(forecast_days):
            features = self.generate_baseline_features(latitude, longitude, current_date)
            if day > 0:
                features["aod_550"] = min(2.0, features["aod_550"] + min(0.3, day * 0.1))

            pm25_prediction = self._predict(features)
            aqi, category = self.pm25_to_cpcb_aqi(pm25_prediction)
            forecasts.append(
                {
                    "date": current_date.isoformat(),
                    "pm2_5": round(pm25_prediction, 1),
                    "aqi": int(aqi),
                    "category": category,
                    "temperature": round(features["t2m_celsius"], 1),
                    "humidity": round(features["r2m"], 1),
                    "wind_speed": round(features["wind_speed_10m"], 1),
                }
            )
            current_date += timedelta(days=1)

        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "forecast": forecasts,
            "model_info": {"type": "canonical_bundle", "accuracy": "see bundle metrics"},
        }

    def predict_single(self, feature_array):
        try:
            row = {
                "aod_550": feature_array[0][0],
                "t2m_celsius": feature_array[0][1],
                "wind_speed_10m": feature_array[0][2],
                "r2m": feature_array[0][3],
                "blh": feature_array[0][4],
                "latitude": np.degrees(np.arcsin(feature_array[0][6])),
                "longitude": np.degrees(np.arcsin(feature_array[0][8])),
                "hour": feature_array[0][9],
                "month": feature_array[0][10],
                "season": feature_array[0][11],
            }
            return self._predict(row)
        except Exception as exc:
            safe_print(f"Prediction error: {exc}")
            aod = feature_array[0][0] if len(feature_array) > 0 and len(feature_array[0]) > 0 else 0.6
            return min(500, max(5, aod * 120))

    def predict_pm25_offline(self, input_features: dict) -> dict:
        if not self.model_loaded:
            return {
                "pm25": 0.0,
                "aqi": 0,
                "health_category": "Error",
                "health_message": "Model bundle not available",
            }

        try:
            pm25_prediction = self._predict(input_features)
            aqi_val, health_cat = self.pm25_to_cpcb_aqi(pm25_prediction)
            health_messages = {
                "Good": "🟢 Excellent air quality! Perfect for outdoor activities.",
                "Satisfactory": "🟡 Good air quality with minor concern for sensitive individuals.",
                "Moderate": "🟠 Moderate air quality. Sensitive individuals may experience symptoms.",
                "Poor": "🔴 Poor air quality. Health effects may be experienced by everyone.",
                "Very Poor": "🟣 Very poor air quality. Serious health effects for everyone.",
                "Severe": "🔴 Severe air quality emergency! Stay indoors.",
            }
            return {
                "pm25": round(pm25_prediction, 1),
                "aqi": int(aqi_val),
                "health_category": health_cat,
                "health_message": health_messages.get(health_cat, "Unknown air quality status"),
            }
        except Exception as exc:
            safe_print(f"Prediction error: {exc}")
            return {
                "pm25": 0.0,
                "aqi": 0,
                "health_category": "Error",
                "health_message": f"Prediction failed: {exc}",
            }


offline_forecast = OfflineForecast()
