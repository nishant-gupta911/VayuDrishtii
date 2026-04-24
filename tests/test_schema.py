import pandas as pd
import pytest

from src.config import ModelConfig
from src.pipeline import validate_feature_frame
from src.schema import FeatureSchema


def test_schema_rejects_missing_feature():
    schema = FeatureSchema.from_feature_names(ModelConfig().feature_columns)
    frame = pd.DataFrame([{"aod_550": 0.5}])
    with pytest.raises(ValueError):
        validate_feature_frame(frame, schema)


def test_schema_rejects_invalid_range():
    schema = FeatureSchema.from_feature_names(ModelConfig().feature_columns)
    frame = pd.DataFrame([{
        "aod_550": 0.5,
        "t2m_celsius": 25,
        "wind_speed_10m": 4,
        "r2m": 150,
        "blh": 700,
        "lat_cos": 0.9,
        "lat_sin": 0.3,
        "lon_cos": 0.2,
        "lon_sin": 0.8,
        "hour": 12,
        "month": 5,
        "season": 2,
        "aod_temp_interaction": 0.1,
        "aod_wind_interaction": 1.0,
        "temp_humidity_interaction": 3.0,
    }])
    with pytest.raises(ValueError):
        validate_feature_frame(frame, schema)
