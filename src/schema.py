from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    required: bool = True


@dataclass(frozen=True)
class FeatureSchema:
    features: Sequence[FeatureSpec]

    @property
    def feature_names(self) -> List[str]:
        return [feature.name for feature in self.features]

    def to_dict(self) -> Dict[str, object]:
        return {
            "features": [
                {"name": feature.name, "dtype": feature.dtype, "required": feature.required}
                for feature in self.features
            ]
        }

    @classmethod
    def from_feature_names(cls, feature_names: Iterable[str]) -> "FeatureSchema":
        return cls(features=[FeatureSpec(name=name, dtype="float64") for name in feature_names])

    def validate_dataframe(self, frame: pd.DataFrame, *, allow_extra: bool = False) -> pd.DataFrame:
        expected = set(self.feature_names)
        provided = set(frame.columns)

        missing = sorted(expected - provided)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        extras = sorted(provided - expected)
        if extras and not allow_extra:
            raise ValueError(f"Unexpected extra features: {extras}")

        validated = frame.loc[:, self.feature_names].copy()
        for spec in self.features:
            validated[spec.name] = pd.to_numeric(validated[spec.name], errors="raise")

        return validated


def validate_no_nulls(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    null_columns = [column for column in required_columns if frame[column].isna().any()]
    if null_columns:
        raise ValueError(f"Null values detected in required features: {null_columns}")


def validate_ranges(frame: pd.DataFrame) -> None:
    range_checks = {
        "aod_550": (0.0, 5.0),
        "t2m_celsius": (-50.0, 65.0),
        "wind_speed_10m": (0.0, 100.0),
        "r2m": (0.0, 100.0),
        "blh": (0.0, 10000.0),
        "hour": (0.0, 23.0),
        "month": (1.0, 12.0),
        "season": (0.0, 4.0),
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
    }
    violations = []
    for column, (minimum, maximum) in range_checks.items():
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="raise")
            if ((series < minimum) | (series > maximum)).any():
                violations.append(f"{column} outside [{minimum}, {maximum}]")
    if violations:
        raise ValueError(f"Range validation failed: {violations}")


def validate_statistical_sanity(frame: pd.DataFrame) -> None:
    if len(frame) < 20:
        return
    if "aod_550" in frame.columns and "blh" in frame.columns:
        if frame["aod_550"].nunique(dropna=True) == 1 and frame["blh"].nunique(dropna=True) == 1:
            raise ValueError("Statistical sanity check failed: inputs appear constant across key features")
