from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    canonical_bundle_path: Path = models_dir / "pm25_clean_bundle.joblib"

    def latest_preprocessed_csv(self) -> Path:
        candidates = sorted(self.processed_dir.glob("preprocessed_data_*.csv"))
        if candidates:
            return candidates[-1]

        fallback = self.processed_dir / "train_sample_200k.csv"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            "No preprocessed dataset found in data/processed. "
            "Expected preprocessed_data_*.csv or train_sample_200k.csv."
        )


@dataclass(frozen=True)
class ModelConfig:
    target_column: str = "pm2_5"
    timestamp_column: str = "datetime"
    station_column: str = "station_name"
    train_fraction: float = 0.8
    random_state: int = 42
    model_version: str = "1.0.0"
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    feature_columns: Sequence[str] = field(
        default_factory=lambda: (
            "aod_550",
            "t2m_celsius",
            "wind_speed_10m",
            "r2m",
            "blh",
            "lat_cos",
            "lat_sin",
            "lon_cos",
            "lon_sin",
            "hour",
            "month",
            "season",
            "aod_temp_interaction",
            "aod_wind_interaction",
            "temp_humidity_interaction",
        )
    )
    leakage_columns: Sequence[str] = field(
        default_factory=lambda: (
            "pm25",
            "pm2_5",
            "aqi",
            "aqi_value",
            "aqi_category",
            "category",
            "has_ground_truth",
        )
    )

