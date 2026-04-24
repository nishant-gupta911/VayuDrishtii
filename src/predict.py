from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .config import ModelConfig, PathConfig
from .pipeline import build_feature_frame, validate_feature_frame
from .schema import FeatureSchema, FeatureSpec


def load_bundle(bundle_path: str | Path | None = None) -> dict[str, Any]:
    path_config = PathConfig()
    path = path_config.canonical_bundle_path if bundle_path is None else Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}")
    bundle = joblib.load(path)
    required_keys = {"model", "preprocessor", "feature_schema", "feature_names"}
    missing = sorted(required_keys - set(bundle.keys()))
    if missing:
        raise ValueError(f"Invalid model bundle. Missing keys: {missing}")
    return bundle


def _schema_from_bundle(bundle: dict[str, Any]) -> FeatureSchema:
    return FeatureSchema(
        features=[FeatureSpec(**feature) for feature in bundle["feature_schema"]["features"]]
    )


def predict_dataframe(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    config = ModelConfig()
    schema = _schema_from_bundle(bundle)
    engineered = build_feature_frame(frame, config)
    validated = validate_feature_frame(engineered, schema, allow_extra=True)
    matrix = bundle["preprocessor"]["object"].transform(validated)
    return pd.Series(bundle["model"].predict(matrix), index=frame.index, name="predicted_pm2_5")


def predict_from_mapping(payload: Mapping[str, Any], bundle: dict[str, Any]) -> float:
    frame = pd.DataFrame([dict(payload)])
    return float(predict_dataframe(frame, bundle).iloc[0])
