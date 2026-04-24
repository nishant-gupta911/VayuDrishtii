import pandas as pd
import pytest

from src.predict import predict_from_mapping


class DummyPreprocessor:
    def transform(self, frame):
        return frame


class DummyModel:
    def predict(self, frame):
        return [42.0 for _ in range(len(frame))]


def test_prediction_returns_stable_value():
    bundle = {
        "model": DummyModel(),
        "preprocessor": {"object": DummyPreprocessor()},
        "feature_schema": {"features": [
            {"name": "aod_550", "dtype": "float64", "required": True},
            {"name": "t2m_celsius", "dtype": "float64", "required": True},
            {"name": "wind_speed_10m", "dtype": "float64", "required": True},
            {"name": "r2m", "dtype": "float64", "required": True},
            {"name": "blh", "dtype": "float64", "required": True},
            {"name": "lat_cos", "dtype": "float64", "required": True},
            {"name": "lat_sin", "dtype": "float64", "required": True},
            {"name": "lon_cos", "dtype": "float64", "required": True},
            {"name": "lon_sin", "dtype": "float64", "required": True},
            {"name": "hour", "dtype": "float64", "required": True},
            {"name": "month", "dtype": "float64", "required": True},
            {"name": "season", "dtype": "float64", "required": True},
            {"name": "aod_temp_interaction", "dtype": "float64", "required": True},
            {"name": "aod_wind_interaction", "dtype": "float64", "required": True},
            {"name": "temp_humidity_interaction", "dtype": "float64", "required": True},
        ]},
        "feature_names": [],
    }
    payload = {
        "latitude": 28.6,
        "longitude": 77.2,
        "aod_550": 0.6,
        "t2m_celsius": 25.0,
        "wind_speed_10m": 4.0,
        "r2m": 65.0,
        "blh": 800.0,
        "hour": 12,
        "month": 6,
        "season": 2,
    }
    assert predict_from_mapping(payload, bundle) == 42.0


def test_prediction_rejects_invalid_input():
    bundle = {
        "model": DummyModel(),
        "preprocessor": {"object": DummyPreprocessor()},
        "feature_schema": {"features": [
            {"name": "aod_550", "dtype": "float64", "required": True},
            {"name": "t2m_celsius", "dtype": "float64", "required": True},
            {"name": "wind_speed_10m", "dtype": "float64", "required": True},
            {"name": "r2m", "dtype": "float64", "required": True},
            {"name": "blh", "dtype": "float64", "required": True},
            {"name": "lat_cos", "dtype": "float64", "required": True},
            {"name": "lat_sin", "dtype": "float64", "required": True},
            {"name": "lon_cos", "dtype": "float64", "required": True},
            {"name": "lon_sin", "dtype": "float64", "required": True},
            {"name": "hour", "dtype": "float64", "required": True},
            {"name": "month", "dtype": "float64", "required": True},
            {"name": "season", "dtype": "float64", "required": True},
            {"name": "aod_temp_interaction", "dtype": "float64", "required": True},
            {"name": "aod_wind_interaction", "dtype": "float64", "required": True},
            {"name": "temp_humidity_interaction", "dtype": "float64", "required": True},
        ]},
        "feature_names": [],
    }
    with pytest.raises(ValueError):
        predict_from_mapping({"latitude": 28.6, "longitude": 77.2, "aod_550": 0.6}, bundle)
