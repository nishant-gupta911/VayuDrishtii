import pandas as pd

from src.config import ModelConfig
from src.pipeline import chronological_split


def test_pipeline_chronological_split_orders_time():
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=10, freq="D"),
        "latitude": [28.6] * 10,
        "longitude": [77.2] * 10,
        "aod_550": [0.5] * 10,
        "t2m_celsius": [25.0] * 10,
        "wind_speed_10m": [4.0] * 10,
        "r2m": [60.0] * 10,
        "blh": [800.0] * 10,
        "hour": [12] * 10,
        "month": [1] * 10,
        "season": [1] * 10,
        "pm2_5": [40.0 + i for i in range(10)],
        "station_name": ["station"] * 10,
    })
    split = chronological_split(frame, ModelConfig())
    assert len(split.train_X) == 8
    assert len(split.valid_X) == 2
    assert split.train_meta["datetime"].max() < split.valid_meta["datetime"].min()
