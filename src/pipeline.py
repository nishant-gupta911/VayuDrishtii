from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import ModelConfig, PathConfig
from .schema import FeatureSchema, validate_no_nulls, validate_ranges, validate_statistical_sanity

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    train_X: pd.DataFrame
    train_y: pd.Series
    valid_X: pd.DataFrame
    valid_y: pd.Series
    train_meta: pd.DataFrame
    valid_meta: pd.DataFrame


def load_training_frame(path_config: PathConfig, data_path: str | None = None) -> pd.DataFrame:
    csv_path = path_config.latest_preprocessed_csv() if data_path is None else path_config.project_root / data_path
    logger.info("Loading training data from %s", csv_path)
    frame = pd.read_csv(csv_path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Dataset at {csv_path} is empty")
    return frame


def build_feature_frame(frame: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    data = frame.copy()
    required_base = ["aod_550", "t2m_celsius", "wind_speed_10m", "r2m", "blh", "hour", "month"]
    missing_base = [column for column in required_base if column not in data.columns]
    if missing_base:
        raise ValueError(f"Dataset is missing required base columns: {missing_base}")

    if "latitude" not in data.columns or "longitude" not in data.columns:
        raise ValueError("Dataset must contain latitude and longitude columns")

    if "season" not in data.columns:
        data["season"] = ((pd.to_numeric(data["month"], errors="coerce") % 12) // 3).astype("Int64")

    data["lat_cos"] = np.cos(np.radians(pd.to_numeric(data["latitude"], errors="coerce")))
    data["lat_sin"] = np.sin(np.radians(pd.to_numeric(data["latitude"], errors="coerce")))
    data["lon_cos"] = np.cos(np.radians(pd.to_numeric(data["longitude"], errors="coerce")))
    data["lon_sin"] = np.sin(np.radians(pd.to_numeric(data["longitude"], errors="coerce")))
    data["aod_temp_interaction"] = pd.to_numeric(data["aod_550"], errors="coerce") * pd.to_numeric(
        data["t2m_celsius"], errors="coerce"
    ) / 100.0
    data["aod_wind_interaction"] = pd.to_numeric(data["aod_550"], errors="coerce") * (
        10.0 - np.minimum(pd.to_numeric(data["wind_speed_10m"], errors="coerce"), 10.0)
    )
    data["temp_humidity_interaction"] = (
        pd.to_numeric(data["t2m_celsius"], errors="coerce") * pd.to_numeric(data["r2m"], errors="coerce") / 1000.0
    )

    leakage_present = [column for column in config.leakage_columns if column in config.feature_columns]
    if leakage_present:
        raise ValueError(f"Canonical feature config contains leakage columns: {leakage_present}")

    return data


def validate_feature_frame(frame: pd.DataFrame, schema: FeatureSchema, *, allow_extra: bool = False) -> pd.DataFrame:
    validated = schema.validate_dataframe(frame, allow_extra=allow_extra)
    validate_no_nulls(validated, schema.feature_names)
    validate_ranges(frame)
    validate_statistical_sanity(frame)
    return validated


def build_schema(config: ModelConfig) -> FeatureSchema:
    return FeatureSchema.from_feature_names(config.feature_columns)


def chronological_split(frame: pd.DataFrame, config: ModelConfig) -> DatasetSplit:
    if config.target_column not in frame.columns:
        raise ValueError(f"Target column '{config.target_column}' missing from dataset")
    if config.timestamp_column not in frame.columns:
        raise ValueError(f"Timestamp column '{config.timestamp_column}' missing from dataset")

    prepared = build_feature_frame(frame, config)
    prepared[config.timestamp_column] = pd.to_datetime(prepared[config.timestamp_column], errors="coerce")
    prepared = prepared.dropna(subset=[config.timestamp_column, config.target_column]).sort_values(config.timestamp_column)
    if prepared.empty:
        raise ValueError("No valid rows remain after timestamp/target filtering")

    schema = build_schema(config)
    features = validate_feature_frame(prepared.loc[:, list(config.feature_columns)], schema, allow_extra=True)
    target = pd.to_numeric(prepared[config.target_column], errors="coerce")

    split_index = int(len(prepared) * config.train_fraction)
    if split_index <= 0 or split_index >= len(prepared):
        raise ValueError("Chronological split produced an empty train or validation partition")

    meta_columns = [column for column in [config.timestamp_column, config.station_column] if column in prepared.columns]
    meta = prepared.loc[:, meta_columns].reset_index(drop=True)
    return DatasetSplit(
        train_X=features.iloc[:split_index].reset_index(drop=True),
        train_y=target.iloc[:split_index].reset_index(drop=True),
        valid_X=features.iloc[split_index:].reset_index(drop=True),
        valid_y=target.iloc[split_index:].reset_index(drop=True),
        train_meta=meta.iloc[:split_index].reset_index(drop=True),
        valid_meta=meta.iloc[split_index:].reset_index(drop=True),
    )


def train_model(split: DatasetSplit, config: ModelConfig) -> Tuple[SimpleImputer, Any]:
    logger.info("Training canonical XGBoost regressor on %s rows", len(split.train_X))
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for training. Install dependencies from requirements.txt."
        ) from exc
    imputer = SimpleImputer(strategy="median")
    train_matrix = imputer.fit_transform(split.train_X)
    model = XGBRegressor(**config.xgb_params)
    model.fit(train_matrix, split.train_y)
    return imputer, model


def evaluate_model(imputer: SimpleImputer, model: Any, split: DatasetSplit) -> Dict[str, float]:
    valid_matrix = imputer.transform(split.valid_X)
    predictions = model.predict(valid_matrix)
    rmse = mean_squared_error(split.valid_y, predictions) ** 0.5
    mae = mean_absolute_error(split.valid_y, predictions)
    r2 = r2_score(split.valid_y, predictions)
    tolerance_accuracy = float(np.mean(np.abs(predictions - split.valid_y) / np.maximum(np.abs(split.valid_y), 1.0) <= 0.10))
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "tolerance_accuracy": tolerance_accuracy}


def vectorized_merge_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    time_column: str,
    latitude_column: str,
    longitude_column: str,
    tolerance: pd.Timedelta,
    suffix: str,
) -> pd.DataFrame:
    """Nearest timestamp merge plus vectorized geographic distance filter."""
    left_sorted = left.sort_values(time_column).copy()
    right_sorted = right.sort_values(time_column).copy()

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on=time_column,
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", suffix),
    )
    right_lat = f"{latitude_column}{suffix}"
    right_lon = f"{longitude_column}{suffix}"
    if right_lat in merged.columns and right_lon in merged.columns:
        merged["merge_distance_deg"] = np.sqrt(
            (merged[latitude_column] - merged[right_lat]) ** 2 + (merged[longitude_column] - merged[right_lon]) ** 2
        )
    return merged
