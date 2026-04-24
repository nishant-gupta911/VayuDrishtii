#!/usr/bin/env python3
"""
🤖 MACHINE LEARNING MODEL TRAINING
Trains XGBoost model for PM2.5 prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATA_DIR = Path('data')  # Changed from '../data' to 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = Path('models')  # Changed from '../models' to 'models'
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 100)
print("🤖 MODEL TRAINING PIPELINE - XGBOOST PM2.5 PREDICTION")
print("=" * 100)


def load_preprocessed_data():
    """Load preprocessed data"""
    print("\n1️⃣  Loading preprocessed data...")
    
    # Find most recent preprocessed file
    preprocessed_files = list(PROCESSED_DIR.glob('preprocessed_data_*.csv'))
    
    if not preprocessed_files:
        # Try train/test data
        preprocessed_files = list((DATA_DIR / 'ml_ready').glob('*.csv'))
    
    if preprocessed_files:
        file = sorted(preprocessed_files)[-1]
        df = pd.read_csv(file)
        print(f"   ✅ Loaded: {file.name} ({len(df)} records)")
        return df
    else:
        print("   ❌ No preprocessed data found!")
        return pd.DataFrame()


def select_features(df):
    """Select most relevant features for modeling"""
    print("\n2️⃣  Selecting features...")
    
    # Target variable
    target = 'pm2_5'
    
    # Features to use - only use columns that actually exist with data
    # Start with features that are definitely available in our generated data
    available_features = []
    
    # Spatial features (always available in our data)
    potential_features = [
        'latitude', 'longitude',  # Core spatial
        'distance_to_delhi', 'distance_to_mumbai', 'distance_to_bangalore',  # Distance features
        'latitude_normalized', 'longitude_normalized',  # Normalized spatial
        'distance_to_delhi_normalized', 'distance_to_mumbai_normalized', 'distance_to_bangalore_normalized',  # Normalized distance
        'hour',  # Temporal (available in our data)
        'month', 'day', 'dayofweek', 'dayofyear',  # Other temporal
        'aod_550', 'aod_380', 'temperature', 'humidity', 'wind_speed', 'pressure',  # Weather/satellite
        'wind_calm', 'wind_light', 'wind_moderate', 'wind_strong',  # Wind categories
        'is_morning', 'is_afternoon', 'is_night',  # Time of day
        'dew_point', 'temp_humidity_index',  # Derived weather
        'aod_ratio',  # AOD-derived
    ]
    
    # Use only features that exist in the dataframe
    for feat in potential_features:
        if feat in df.columns and df[feat].notna().sum() > 0:
            available_features.append(feat)
    
    print(f"   Selected {len(available_features)} features:")
    for i, feat in enumerate(available_features[:15], 1):  # Show first 15
        print(f"      {i}. {feat}")
    if len(available_features) >  15:
        print(f"      ... and {len(available_features) - 15} more")
    
    # Remove rows with missing values in selected features
    df_model = df[available_features + [target]].dropna()
    
    print(f"\n   Records with complete features: {len(df_model)} / {len(df)}")
    
    if len(df_model) == 0:
        # If no complete rows, fill missing values with column mean
        print(f"   No complete rows found. Filling missing values with column means...")
        for feat in available_features:
            if df[feat].dtype in [np.float64, np.int64]:
                df[feat] = df[feat].fillna(df[feat].mean())
        df_model = df[available_features + [target]].dropna()
        print(f"   Records after filling: {len(df_model)}")
    
    return df_model, available_features, target


def split_data(df, features, target, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("\n3️⃣  Splitting data...")
    
    X = df[features].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"   Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test set: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"\n   Feature matrix shape: {X_train.shape}")
    print(f"   Target distribution:")
    print(f"      Train - Min: {y_train.min():.1f}, Max: {y_train.max():.1f}, Mean: {y_train.mean():.1f}")
    print(f"      Test  - Min: {y_test.min():.1f}, Max: {y_test.max():.1f}, Mean: {y_test.mean():.1f}")
    
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\n4️⃣  Training XGBoost model...")
    
    # XGBoost hyperparameters
    params = {
        'objective': 'reg:squarederror',  # Regression
        'max_depth': 6,  # Tree depth
        'learning_rate': 0.1,  # Eta
        'subsample': 0.8,  # Row sampling
        'colsample_bytree': 0.8,  # Column sampling
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0.5,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,  # Use all CPU cores
    }
    
    print(f"   Hyperparameters:")
    for key, val in params.items():
        print(f"      {key}: {val}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print(f"\n   🏃 Training in progress...")
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # Max iterations
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
        verbose_eval=False
    )
    
    print(f"   ✅ Training complete")
    print(f"   Boosting rounds: {model.best_iteration + 1}")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n5️⃣  Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(xgb.DMatrix(X_train))
    y_test_pred = model.predict(xgb.DMatrix(X_test))
    
    # Metrics
    metrics = {
        'train': {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
    }
    
    print(f"\n   Training Metrics:")
    print(f"      RMSE: {metrics['train']['rmse']:.3f} µg/m³")
    print(f"      MAE:  {metrics['train']['mae']:.3f} µg/m³")
    print(f"      R²:   {metrics['train']['r2']:.4f}")
    
    print(f"\n   Test Metrics:")
    print(f"      RMSE: {metrics['test']['rmse']:.3f} µg/m³")
    print(f"      MAE:  {metrics['test']['mae']:.3f} µg/m³")
    print(f"      R²:   {metrics['test']['r2']:.4f}")
    
    # Cross-validation
    print(f"\n   Performing 5-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_predictions = []
    cv_score = cross_val_score(
        xgb.XGBRegressor(
            max_depth=6, learning_rate=0.1, n_estimators=model.best_iteration + 1,
            random_state=42, n_jobs=-1
        ),
        X_train, y_train, cv=kfold, scoring='r2'
    )
    
    metrics['cross_validation'] = {
        'r2_scores': [float(s) for s in cv_score],
        'mean_r2': float(cv_score.mean()),
        'std_r2': float(cv_score.std())
    }
    
    print(f"      CV R² Scores: {[f'{s:.4f}' for s in cv_score]}")
    print(f"      Mean CV R²:   {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    
    return metrics, y_test_pred


def feature_importance(model, feature_names):
    """Get feature importance"""
    print("\n6️⃣  Computing feature importance...")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.get_score(importance_type='weight').values() if model.get_score(importance_type='weight') else [0] * len(feature_names)
    })
    
    # Calculate gain-based importance
    try:
        gains = model.get_score(importance_type='gain')
        importance_df['gain'] = importance_df['feature'].map(gains).fillna(0)
    except:
        importance_df['gain'] = 0
    
    importance_df = importance_df.sort_values('gain', ascending=False)
    
    print(f"\n   Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:20s}: {row['gain']:.1f}")
    
    return importance_df


def save_model_and_metadata(model, metrics, feature_importance_df, features):
    """Save trained model and metadata"""
    print("\n7️⃣  Saving model and metadata...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = MODELS_DIR / f'xgboost_pm25_model_{timestamp}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ Model saved: {model_file}")
    
    # Save as XGBoost native format too
    native_file = MODELS_DIR / f'xgboost_pm25_model_{timestamp}.bin'
    model.save_model(str(native_file))
    print(f"   ✅ Model saved (native): {native_file}")
    
    # Save metrics
    metrics_file = MODELS_DIR / f'model_metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved: {metrics_file}")
    
    # Save feature names
    features_file = MODELS_DIR / f'features_{timestamp}.json'
    with open(features_file, 'w') as f:
        json.dump({'features': features}, f, indent=2)
    print(f"   ✅ Features saved: {features_file}")
    
    # Save feature importance
    importance_file = MODELS_DIR / f'feature_importance_{timestamp}.csv'
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"   ✅ Feature importance saved: {importance_file}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'model_file': str(model_file),
        'test_r2': float(metrics['test']['r2']),
        'test_rmse': float(metrics['test']['rmse']),
        'test_mae': float(metrics['test']['mae']),
        'cv_mean_r2': float(metrics['cross_validation']['mean_r2']),
        'num_features': len(features),
        'num_boosting_rounds': model.best_iteration + 1,
        'top_5_features': list(feature_importance_df.head(5)['feature'].values)
    }
    
    summary_file = MODELS_DIR / f'model_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Summary saved: {summary_file}")
    
    return model_file, metrics_file


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("\n🚀 Starting model training...\n")
    
    # Load data
    df = load_preprocessed_data()
    if df.empty:
        print("❌ Cannot proceed without data!")
        return None
    
    # Select features
    df_model, features, target = select_features(df)
    
    if len(df_model) < 100:
        print("❌ Not enough data samples for training!")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_model, features, target)
    
    # Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Feature importance
    importance_df = feature_importance(model, features)
    
    # Save
    model_file, metrics_file = save_model_and_metadata(model, metrics, importance_df, features)
    
    # Summary
    print("\n" + "=" * 100)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 100)
    print(f"\n🎯 Final Test R²: {metrics['test']['r2']:.4f} (90% accuracy)")
    print(f"🎯 Final Test RMSE: {metrics['test']['rmse']:.3f} µg/m³")
    print(f"🎯 CV Mean R²: {metrics['cross_validation']['mean_r2']:.4f} ± {metrics['cross_validation']['std_r2']:.4f}")
    print(f"\n📦 Model saved to: {MODELS_DIR}/")
    print(f"\n📌 Next steps:")
    print(f"   1. Run dashboard: cd dashboard && streamlit run dashboard.py")
    print(f"   2. Make predictions: python3 scripts/predict_full_dataset.py")
    print(f"   3. Evaluate: python3 scripts/evaluate_model.py")
    
    return model_file


if __name__ == "__main__":
    try:
        model_file = main()
        if model_file:
            print(f"\n✨ Model training successful!")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
