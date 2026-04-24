from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib

from .config import ModelConfig, PathConfig
from .pipeline import chronological_split, evaluate_model, load_training_frame, train_model
from .schema import FeatureSchema


def build_bundle(
    *,
    model: Any,
    imputer: Any,
    schema: FeatureSchema,
    model_config: ModelConfig,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "bundle_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_config.model_version,
        "target_column": model_config.target_column,
        "timestamp_column": model_config.timestamp_column,
        "feature_schema": schema.to_dict(),
        "feature_names": schema.feature_names,
        "preprocessor": {"type": "SimpleImputer", "strategy": "median", "object": imputer},
        "model": model,
        "metrics": metrics,
        "config": {
            "train_fraction": model_config.train_fraction,
            "xgb_params": model_config.xgb_params,
        },
    }


def run_training(data_path: str | None = None, output_path: str | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path_config = PathConfig()
    model_config = ModelConfig()

    frame = load_training_frame(path_config, data_path)
    split = chronological_split(frame, model_config)
    imputer, model = train_model(split, model_config)
    metrics = evaluate_model(imputer, model, split)
    schema = FeatureSchema.from_feature_names(model_config.feature_columns)
    bundle = build_bundle(model=model, imputer=imputer, schema=schema, model_config=model_config, metrics=metrics)

    path_config.models_dir.mkdir(parents=True, exist_ok=True)
    destination = path_config.canonical_bundle_path if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)

    metrics_path = path_config.models_dir / "pm25_model_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    logging.getLogger(__name__).info("Saved canonical model bundle to %s", destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the canonical VayuDrishti PM2.5 model")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV path relative to project root")
    parser.add_argument("--output-path", type=str, default=None, help="Optional output bundle path")
    args = parser.parse_args(argv)
    run_training(data_path=args.data_path, output_path=args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

