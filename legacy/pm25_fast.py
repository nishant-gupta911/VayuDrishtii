#!/usr/bin/env python3
"""Aggressive PM2.5 Optimization: Sample-Train + GPU + 97%+ Target"""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANGE_BINS = [0.0, 35.0, 75.0, 150.0, np.inf]
RANGE_LABELS = ["0-35", "35-75", "75-150", "150+"]


class FastPM25Optimizer:
    def __init__(self, sample_train_size=None, val_size=0.15):
        self.sample_train_size = sample_train_size  # None = use full data
        self.val_size = val_size

    def load_data(self):
        """Load full training data."""
        print("[DATA] Loading complete train CSV...")
        chunks = []
        for chunk in pd.read_csv(
            "data/processed/train_2gb_engineered_8694071rows_20260408_102015.csv",
            chunksize=500_000
        ):
            chunks.append(chunk)
        
        train_df = pd.concat(chunks, ignore_index=True)
        print(f"  ✓ Train: {len(train_df):,} (FULL DATASET)")
        
        print("[DATA] Loading test CSV...")
        test_df = pd.read_csv("data/processed/test_2gb_engineered_2173518rows_20260408_102015.csv")
        print(f"  ✓ Test: {len(test_df):,}")
        
        print("[DATA] Loading features...")
        with open("data/processed/feature_list_20260408_102015.json") as f:
            features = json.load(f).get("features", [])
        print(f"  ✓ Features: {len(features)}")
        
        return train_df, test_df, features

    def prepare_data(self, train_df, test_df, features):
        """Prepare X, y matrices."""
        # Filter features
        feature_cols = [f for f in features if f in train_df.columns]
        
        # Split train/val
        train_base, val_df = train_test_split(
            train_df, test_size=self.val_size, random_state=42
        )
        
        # Extract X, y with imputation
        def transform(df, feature_cols):
            y = pd.to_numeric(df['pm25'], errors='coerce').dropna()
            X = df.loc[y.index, feature_cols].copy()
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            return X, y
        
        X_train, y_train = transform(train_base, feature_cols)
        X_val, y_val = transform(val_df, feature_cols)
        X_test, y_test = transform(test_df, feature_cols)
        
        print(f"\n[PREP] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def _tol_acc(y_true, y_pred):
        denom = np.maximum(np.abs(y_true), 1.0)
        return np.mean(np.abs(y_pred - y_true) / denom <= 0.10)

    def train_global_model(self, X_train, y_train_log):
        """Train XGBoost global model with optimized settings."""
        print("\n[MODEL] Training global XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method='hist',
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        model.fit(X_train, y_train_log)
        return model

    def train_range_models(self, X_train, y_train):
        """Train range-specific XGBoost models."""
        print("\n[RANGES] Training range-specific models...")
        y_train_log = np.log1p(y_train.values)
        y_bins = pd.cut(y_train_log, bins=np.log1p(RANGE_BINS), 
                       labels=RANGE_LABELS, include_lowest=True, right=False)
        range_models = {}
        
        range_params = {
            "0-35": {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 1000, 
                     "subsample": 0.8, "colsample_bytree": 0.8},
            "35-75": {"max_depth": 10, "learning_rate": 0.02, "n_estimators": 1000,
                      "subsample": 0.85, "colsample_bytree": 0.85},
            "75-150": {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 800,
                       "subsample": 0.9, "colsample_bytree": 0.9},
            "150+": {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 600,
                     "subsample": 0.9, "colsample_bytree": 0.9},
        }
        
        for label in RANGE_LABELS:
            mask = y_bins.astype(str) == label
            if mask.sum() < 500:
                range_models[label] = None
                continue
            
            X_slice = X_train[mask]
            y_slice = y_train_log[mask]
            
            params = range_params[label].copy()
            n_est = params.pop("n_estimators")
            
            print(f"  [{label}] {len(X_slice):,} samples")
            
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_slice, y_slice, test_size=0.1, random_state=42
            )
            
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr)
            deval = xgb.DMatrix(X_es.values, label=y_es)
            
            params['tree_method'] = 'hist'
            
            model = xgb.train(
                params, dtrain, num_boost_round=n_est,
                evals=[(deval, 'eval')],
                early_stopping_rounds=40,
                verbose_eval=False
            )
            range_models[label] = model
        
        return range_models

    def train_router(self, X_train, y_train):
        """Train range router classifier."""
        print("[ROUTER] Training range router...")
        y_bins = pd.cut(y_train, bins=RANGE_BINS, labels=RANGE_LABELS,
                       include_lowest=True, right=False).astype(str)
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=12,
                                         random_state=42, n_jobs=-1))
        ])
        clf.fit(X_train, y_bins)
        return clf

    def train_residual_model(self, X_train, y_pred_train, y_train_orig):
        """Train residual correction with LightGBM."""
        residuals = y_train_orig.values - y_pred_train
        if np.abs(residuals).mean() < 0.3:
            return None
        
        print("[RESIDUAL] Training residual correction...")
        model = LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, residuals)
        return model

    def predict_ensemble(self, bundle, X_data):
        """Make predictions with full ensemble."""
        global_pred_log = bundle['global'].predict(X_data)
        pred_log = global_pred_log.copy()
        
        # Residual correction
        if bundle.get('residual') is not None:
            pred_orig = np.expm1(pred_log)
            correction = bundle['residual'].predict(X_data)
            pred_orig = pred_orig + 0.1 * correction
            pred_log = np.log1p(np.clip(pred_orig, 5.0, 500.0))
        
        # Calibration
        if bundle.get('calibrator') is not None:
            pred_log = bundle['calibrator'].predict(pred_log)
        
        return np.clip(np.expm1(pred_log), 5.0, 500.0)

    def run(self):
        """Main optimization pipeline."""
        print("="*80)
        print("AGGRESSIVE PM2.5 OPTIMIZER: LOG + RANGES + GPU + 97%+ TARGET")
        print("="*80)
        
        # Load data
        train_df, test_df, features = self.load_data()
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
            train_df, test_df, features
        )
        
        # Log transform
        print("\n[LOG] Applying log1p transformation...")
        y_train_log = np.log1p(y_train.values)
        y_val_log = np.log1p(y_val.values)
        y_test_log = np.log1p(y_test.values)
        
        # Train global model
        global_model = self.train_global_model(X_train, y_train_log)
        global_val_log = global_model.predict(X_val)
        global_val_orig = np.expm1(global_val_log)
        print(f"  Global ±10% acc (val): {self._tol_acc(y_val.values, global_val_orig)*100:.2f}%")
        
        # Train range models
        range_models = self.train_range_models(X_train, y_train)
        
        # Router
        router = self.train_router(X_train, y_train)
        
        # Residual model
        residual_model = self.train_residual_model(
            X_train, global_model.predict(X_train), y_train
        )
        
        # Calibrator
        print("[CALIB] Fitting isotonic calibration...")
        calib_input = global_model.predict(X_val)
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(calib_input, y_val_log)
        
        # Bundle (simplified - routing not used in predict_ensemble now)
        bundle = {
            'global': global_model,
            'residual': residual_model,
            'calibrator': calibrator,
        }
        
        # Test set evaluation
        print("\n[TEST] Evaluating on full test set...")
        test_pred = self.predict_ensemble(bundle, X_test)
        
        test_tol_std = self._tol_acc(y_test.values, test_pred)
        test_mae = mean_absolute_error(y_test.values, test_pred)
        test_rmse = math.sqrt(mean_squared_error(y_test.values, test_pred))
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"±10% Accuracy: {test_tol_std*100:.2f}%")
        print(f"MAE:          {test_mae:.4f}")
        print(f"RMSE:         {test_rmse:.4f}")
        
        # Per-range breakdown
        y_test_bins = pd.cut(y_test.values, bins=RANGE_BINS, labels=RANGE_LABELS,
                            include_lowest=True, right=False)
        print(f"\nRANGE BREAKDOWN")
        print(f"{'─'*60}")
        for label in RANGE_LABELS:
            mask = y_test_bins.astype(str) == label
            if mask.sum() == 0:
                continue
            y_range = y_test.values[mask]
            pred_range = test_pred[mask]
            acc = self._tol_acc(y_range, pred_range)
            mae = mean_absolute_error(y_range, pred_range)
            rmse = math.sqrt(mean_squared_error(y_range, pred_range))
            print(f"{label:8s} | {mask.sum():8,} | {acc*100:7.2f}% | MAE {mae:6.2f} | RMSE {rmse:6.2f}")
        
        # Save
        model_path = Path("models/pm25_clean_bundle.joblib")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, model_path)
        print(f"\n[SAVED] {model_path}")
        
        # Report
        metadata = {
            "accuracy_std": test_tol_std,
            "mae": test_mae,
            "rmse": test_rmse,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        
        report_path = Path("reports/pipeline_report.json")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(metadata, indent=2))
        print(f"[SAVED] {report_path}")
        
        print(f"{'='*80}\n")
        
        return metadata


if __name__ == "__main__":
    optimizer = FastPM25Optimizer()  # Use full data
    optimizer.run()
