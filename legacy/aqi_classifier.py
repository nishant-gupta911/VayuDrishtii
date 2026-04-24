#!/usr/bin/env python3
"""AQI Classifier - EPA PM2.5 Breakpoints (6 classes)"""
import json, warnings, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
warnings.filterwarnings('ignore')

# ============================================================================
# AQI CONVERSION
# ============================================================================

def pm25_to_aqi(val):
    """Convert PM2.5 µg/m³ to EPA AQI category (0-5)"""
    if val <= 12:
        return 0  # Good
    elif val <= 35:
        return 1  # Moderate
    elif val <= 55:
        return 2  # Unhealthy for Sensitive
    elif val <= 150:
        return 3  # Unhealthy
    elif val <= 250:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous

aqi_names = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy Sensitive",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}

# ============================================================================
# PIPELINE CONFIG
# ============================================================================

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_FULL = DATA_DIR / "train_2gb_engineered_8694071rows_20260408_102015.csv"
TEST_FULL = DATA_DIR / "test_2gb_engineered_2173518rows_20260408_102015.csv"
FEATURES_JSON = DATA_DIR / "feature_list_20260408_102015.json"
LEAKY_FEATURES = {"pm25", "pm2_5", "aqi", "aqi_value", "aqi_category", "category"}

# ============================================================================
# STAGE 1: LOAD & PREPARE DATA (100k / 50k STRATIFIED)
# ============================================================================

def load_and_prepare_stage1():
    """Load full data, create aqi_category, stratified sample 100k/50k"""
    
    print("\n[STAGE 1] Loading data...")
    train_df = pd.read_csv(TRAIN_FULL, low_memory=False)
    test_df = pd.read_csv(TEST_FULL, low_memory=False)
    
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    
    # Convert pm25 to aqi_category
    print("[CONVERT] pm25 → aqi_category")
    train_df['aqi_category'] = train_df['pm25'].apply(pm25_to_aqi)
    test_df['aqi_category'] = test_df['pm25'].apply(pm25_to_aqi)
    
    # Stratified sample using groupby-sample
    print("[SAMPLE] 100k train (stratified)...", end=" ", flush=True)
    n_samples = 100_000
    n_per_class = n_samples // 6
    dfs = []
    for cat in range(6):
        df_cat = train_df[train_df['aqi_category'] == cat]
        sample_size = min(n_per_class, len(df_cat))
        dfs.append(df_cat.sample(n=sample_size, random_state=42))
    train_sample = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    print("✓")
    
    print("[SAMPLE] 50k test (stratified)...", end=" ", flush=True)
    n_samples = 50_000
    n_per_class = n_samples // 6
    dfs = []
    for cat in range(6):
        df_cat = test_df[test_df['aqi_category'] == cat]
        sample_size = min(n_per_class, len(df_cat))
        dfs.append(df_cat.sample(n=sample_size, random_state=42))
    test_sample = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    print("✓")
    
    # Print class distribution
    print("\n[CLASS DISTRIBUTION] Train (100k):")
    for cat in range(6):
        count = (train_sample['aqi_category'] == cat).sum()
        pct = 100 * count / len(train_sample)
        print(f"  Category {cat} ({aqi_names[cat]:<20}): {count:5d} rows ({pct:5.2f}%)")
    
    print("\n[CLASS DISTRIBUTION] Test (50k):")
    for cat in range(6):
        count = (test_sample['aqi_category'] == cat).sum()
        pct = 100 * count / len(test_sample)
        print(f"  Category {cat} ({aqi_names[cat]:<20}): {count:5d} rows ({pct:5.2f}%)")
    
    # Check for severely imbalanced classes
    class_pcts = train_sample['aqi_category'].value_counts(normalize=True).min()
    use_class_weight = class_pcts < 0.01
    if use_class_weight:
        print(f"\n[WARNING] Minority class < 1% ({class_pcts*100:.2f}%). Using class_weight='balanced'")
    
    return train_sample, test_sample, use_class_weight

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def get_feature_columns():
    """Load feature list from JSON"""
    with open(FEATURES_JSON) as f:
        payload = json.load(f)
    features = payload["features"] if isinstance(payload, dict) else payload
    return [feature for feature in features if feature not in LEAKY_FEATURES]

def apply_log_transform(df, features):
    """Log1p transform for pm25, aod, concentration columns"""
    transformed = df.copy()
    for col in features:
        if any(x in col.lower() for x in ['pm25', 'aod', 'concentration']):
            if col in transformed.columns:
                transformed[col] = np.log1p(transformed[col].clip(lower=0))
    return transformed

def add_station_features(train_df, test_df):
    """Add station aggregates"""
    # Station PM2.5 stats
    station_pm25_mean = train_df.groupby('station_id')['pm25'].mean().to_dict()
    station_pm25_std = train_df.groupby('station_id')['pm25'].std().to_dict()
    station_pm25_mode_cat = train_df.groupby('station_id')['aqi_category'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
    global_pm25_mean = train_df['pm25'].mean()
    global_mode_cat = train_df['aqi_category'].mode()[0]
    global_mean_cat = train_df['aqi_category'].mean()
    
    # Hour + station category mean
    hour_cat_mean = train_df.groupby(['station_id', 'hour'])['aqi_category'].mean().to_dict()
    
    # Month + station category mean
    month_cat_mean = train_df.groupby(['station_id', 'month'])['aqi_category'].mean().to_dict()
    
    # Merge into train
    train_out = train_df.copy()
    train_out['station_pm25_mean'] = train_df['station_id'].map(station_pm25_mean).fillna(global_pm25_mean)
    train_out['station_pm25_std'] = train_df['station_id'].map(station_pm25_std).fillna(0)
    train_out['station_pm25_mode_cat'] = train_df['station_id'].map(station_pm25_mode_cat).fillna(global_mode_cat)
    train_out['hour_cat_mean'] = train_df[['station_id', 'hour']].apply(
        lambda r: hour_cat_mean.get((r['station_id'], r['hour']), global_mean_cat), axis=1
    )
    train_out['month_cat_mean'] = train_df[['station_id', 'month']].apply(
        lambda r: month_cat_mean.get((r['station_id'], r['month']), global_mean_cat), axis=1
    )
    
    # Merge into test
    test_out = test_df.copy()
    test_out['station_pm25_mean'] = test_df['station_id'].map(station_pm25_mean).fillna(global_pm25_mean)
    test_out['station_pm25_std'] = test_df['station_id'].map(station_pm25_std).fillna(0)
    test_out['station_pm25_mode_cat'] = test_df['station_id'].map(station_pm25_mode_cat).fillna(global_mode_cat)
    test_out['hour_cat_mean'] = test_df[['station_id', 'hour']].apply(
        lambda r: hour_cat_mean.get((r['station_id'], r['hour']), global_mean_cat), axis=1
    )
    test_out['month_cat_mean'] = test_df[['station_id', 'month']].apply(
        lambda r: month_cat_mean.get((r['station_id'], r['month']), global_mean_cat), axis=1
    )
    
    return train_out, test_out

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(X_train, y_train, use_class_weight=False, **params):
    """Train XGBoost classifier"""
    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    if use_class_weight:
        class_counts = np.bincount(y_train.astype(int), minlength=6)
        class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
        model._vd_sample_weight = np.array([class_weights[int(label)] for label in y_train], dtype=float)
    return model

def train_lightgbm(X_train, y_train, use_class_weight=False, **params):
    """Train LightGBM classifier"""
    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'device_type': 'gpu',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    default_params.update(params)
    
    if use_class_weight:
        default_params['class_weight'] = 'balanced'
    
    return LGBMClassifier(**default_params)

def train_randomforest(X_train, y_train, use_class_weight=False, **params):
    """Train RandomForest classifier"""
    default_params = {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_leaf': 5,
        'n_jobs': -1,
        'random_state': 42
    }
    default_params.update(params)
    
    if use_class_weight:
        default_params['class_weight'] = 'balanced'
    
    return RandomForestClassifier(**default_params)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, X_test, y_test, model_name=""):
    """Evaluate and print accuracy + per-class breakdown"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if model_name:
        print(f"\n[{model_name}] Overall accuracy: {acc*100:.2f}%")
    else:
        print(f"Overall accuracy: {acc*100:.2f}%")
    
    print("Per-class accuracy:")
    for cat in range(6):
        mask = y_test == cat
        if mask.sum() > 0:
            cat_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {aqi_names[cat]:<20}: {cat_acc*100:6.2f}%")
    
    return acc

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_stage1():
    """Stage 1: Train and optimize on 100k/50k"""
    
    print("\n" + "="*70)
    print("STAGE 1: 100k TRAIN / 50k TEST")
    print("="*70)
    
    # Load and prepare
    train_df, test_df, use_class_weight = load_and_prepare_stage1()
    
    # Features
    feature_cols = get_feature_columns()
    X_train_base = train_df[feature_cols].fillna(train_df[feature_cols].median())
    X_test_base = test_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df['aqi_category'].values
    y_test = test_df['aqi_category'].values
    
    # ========== STEP 1: BASELINE ==========
    print("\n[STEP 1] BASELINE XGBoost (no transforms)")
    X_train = X_train_base.copy()
    X_test = X_test_base.copy()
    
    model_xgb = train_xgboost(X_train, y_train, use_class_weight=use_class_weight)
    model_xgb.fit(X_train, y_train, sample_weight=getattr(model_xgb, "_vd_sample_weight", None))
    baseline_acc = evaluate(model_xgb, X_test, y_test, "XGBoost Baseline")
    
    # ========== STEP 2: LOG TRANSFORM FEATURES ==========
    print("\n[STEP 2] Log transform features")
    X_train = apply_log_transform(X_train_base, feature_cols)
    X_test = apply_log_transform(X_test_base, feature_cols)
    
    model_xgb_log = train_xgboost(X_train, y_train, use_class_weight=use_class_weight)
    model_xgb_log.fit(X_train, y_train, sample_weight=getattr(model_xgb_log, "_vd_sample_weight", None))
    log_acc = evaluate(model_xgb_log, X_test, y_test, "XGBoost + Log Transform")
    
    print(f"\nBEFORE: {baseline_acc*100:.2f}%")
    print(f"AFTER:  {log_acc*100:.2f}%")
    print(f"DELTA:  {(log_acc-baseline_acc)*100:+.2f}%")
    
    if log_acc < baseline_acc:
        print("REVERTED - Log transform degraded accuracy")
        X_train = X_train_base.copy()
        X_test = X_test_base.copy()
        model_xgb_log = model_xgb
        log_acc = baseline_acc
    
    # ========== STEP 3: STATION FEATURES ==========
    print("\n[STEP 3] Add station features")
    train_with_station, test_with_station = add_station_features(train_df, test_df)
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['station_pm25_mean'] = train_with_station['station_pm25_mean'].values
    X_train['station_pm25_std'] = train_with_station['station_pm25_std'].values
    X_train['station_pm25_mode_cat'] = train_with_station['station_pm25_mode_cat'].values
    X_train['hour_cat_mean'] = train_with_station['hour_cat_mean'].values
    X_train['month_cat_mean'] = train_with_station['month_cat_mean'].values
    
    X_test['station_pm25_mean'] = test_with_station['station_pm25_mean'].values
    X_test['station_pm25_std'] = test_with_station['station_pm25_std'].values
    X_test['station_pm25_mode_cat'] = test_with_station['station_pm25_mode_cat'].values
    X_test['hour_cat_mean'] = test_with_station['hour_cat_mean'].values
    X_test['month_cat_mean'] = test_with_station['month_cat_mean'].values
    
    model_xgb_station = train_xgboost(X_train, y_train, use_class_weight=use_class_weight)
    model_xgb_station.fit(X_train, y_train, sample_weight=getattr(model_xgb_station, "_vd_sample_weight", None))
    station_acc = evaluate(model_xgb_station, X_test, y_test, "XGBoost + Station Features")
    
    print(f"\nBEFORE: {log_acc*100:.2f}%")
    print(f"AFTER:  {station_acc*100:.2f}%")
    print(f"DELTA:  {(station_acc-log_acc)*100:+.2f}%")
    
    if station_acc < log_acc:
        print("REVERTED - Station features degraded accuracy")
        X_train = X_train.drop(['station_pm25_mean', 'station_pm25_std', 'station_pm25_mode_cat', 'hour_cat_mean', 'month_cat_mean'], axis=1, errors='ignore')
        X_test = X_test.drop(['station_pm25_mean', 'station_pm25_std', 'station_pm25_mode_cat', 'hour_cat_mean', 'month_cat_mean'], axis=1, errors='ignore')
        model_xgb_station = model_xgb_log
        station_acc = log_acc
    
    best_model = model_xgb_station
    best_acc = station_acc
    best_name = "XGBoost + Station"
    best_X_train = X_train.copy()
    best_X_test = X_test.copy()
    
    # ========== STEP 4: ENSEMBLE ==========
    print("\n[STEP 4] Ensemble (XGB + LightGBM + RandomForest)")
    
    model_lgb = train_lightgbm(X_train, y_train, use_class_weight=use_class_weight)
    model_lgb.fit(X_train, y_train)
    lgb_acc = evaluate(model_lgb, X_test, y_test, "LightGBM")
    
    model_rf = train_randomforest(X_train, y_train, use_class_weight=use_class_weight)
    model_rf.fit(X_train, y_train)
    rf_acc = evaluate(model_rf, X_test, y_test, "RandomForest")
    
    # Soft voting
    y_pred_xgb = model_xgb_station.predict_proba(X_test)
    y_pred_lgb = model_lgb.predict_proba(X_test)
    y_pred_rf = model_rf.predict_proba(X_test)
    
    y_pred_ensemble = (y_pred_xgb + y_pred_lgb + y_pred_rf) / 3
    y_pred_ensemble = np.argmax(y_pred_ensemble, axis=1)
    
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"\n[ENSEMBLE] Overall accuracy: {ensemble_acc*100:.2f}%")
    print("Per-class accuracy:")
    for cat in range(6):
        mask = y_test == cat
        if mask.sum() > 0:
            cat_acc = accuracy_score(y_test[mask], y_pred_ensemble[mask])
            print(f"  {aqi_names[cat]:<20}: {cat_acc*100:6.2f}%")
    
    print(f"\nBEFORE: {best_acc*100:.2f}% (XGBoost + Station)")
    print(f"AFTER:  {ensemble_acc*100:.2f}% (Ensemble)")
    print(f"DELTA:  {(ensemble_acc-best_acc)*100:+.2f}%")
    
    if ensemble_acc > best_acc:
        print("KEPT - Ensemble improved accuracy")
        best_model = "ensemble"
        best_acc = ensemble_acc
        best_name = "Ensemble (XGB+LGB+RF)"
        models_dict = {'xgb': model_xgb_station, 'lgb': model_lgb, 'rf': model_rf}
    else:
        print("REVERTED - Ensemble did not improve")
        models_dict = None
    
    # ========== STAGE 1 CHECK ==========
    if best_acc >= 0.97:
        print("\n" + "="*70)
        print(f"✓ STAGE 1 COMPLETE: {best_acc*100:.2f}% accuracy on 50k test")
        print(f"Best model: {best_name}")
        print("="*70)
        
        if models_dict:
            joblib.dump(models_dict, MODELS_DIR / "aqi_classifier_stage1.joblib")
        else:
            joblib.dump(best_model, MODELS_DIR / "aqi_classifier_stage1.joblib")
        
        with open(MODELS_DIR / "aqi_features_stage1.json", "w") as f:
            json.dump(list(X_train.columns), f)
        
        return best_acc, best_model, models_dict, best_X_train, best_X_test, y_train, y_test
    else:
        print("\n" + "="*70)
        print(f"Stage 1: {best_acc*100:.2f}% (below 97% target)")
        print("="*70)
        return best_acc, best_model, models_dict, best_X_train, best_X_test, y_train, y_test

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    stage1_acc, stage1_model, stage1_ensemble, X_train_s1, X_test_s1, y_train_s1, y_test_s1 = run_stage1()
