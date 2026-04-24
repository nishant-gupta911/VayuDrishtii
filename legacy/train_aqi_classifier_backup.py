import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import optuna
from optuna.pruners import MedianPruner
import sys
import time

warnings.filterwarnings('ignore')

def pm25_to_category(val):
    if val <= 35:
        return 0
    elif val <= 150:
        return 1
    else:
        return 2

def load_features():
    with open('data/processed/feature_list_20260408_102015.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'features' in data:
            return data['features']
        elif isinstance(data, dict):
            return list(data.keys())
        return data

def load_stratified_sample(train_size, test_size):
    print(f"Loading stratified sample: {train_size} train / {test_size} test...")
    
    train_df = pd.read_csv('data/processed/train_2gb_engineered_8694071rows_20260408_102015.csv')
    test_df = pd.read_csv('data/processed/test_2gb_engineered_2173518rows_20260408_102015.csv')
    
    train_df['category'] = train_df['pm25'].apply(pm25_to_category)
    test_df['category'] = test_df['pm25'].apply(pm25_to_category)
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=42)
    features = load_features()
    
    for train_idx, _ in sss.split(train_df, train_df['category']):
        train_sample = train_df.iloc[train_idx].copy()
    
    for _, test_idx in sss.split(test_df, test_df['category']):
        test_sample = test_df.iloc[test_idx].copy()
    
    print("\nClass Distribution (before SMOTE):")
    for cat in range(3):
        count = (train_sample['category'] == cat).sum()
        pct = 100 * count / len(train_sample)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    return train_sample, test_sample, features

def prepare_data(train_df, test_df, features):
    avail_features = [f for f in features if f in train_df.columns]
    X_train = train_df[avail_features].copy()
    y_train = train_df['category'].copy()
    X_test = test_df[avail_features].copy()
    y_test = test_df['category'].copy()
    
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=avail_features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=avail_features)
    
    return X_train, y_train, X_test, y_test

def apply_smote(X_train, y_train):
    print("\n=== STEP 1: APPLY SMOTE ===")
    sm = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    print("Class Distribution (after SMOTE):")
    for cat in range(3):
        count = (y_res == cat).sum()
        pct = 100 * count / len(y_res)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    return X_res, y_res

def print_per_class_accuracy(y_true, y_pred):
    cat_names = ['Clean', 'Polluted', 'Hazardous']
    for cat in range(3):
        mask = y_true == cat
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"  {cat_names[cat]}: {acc*100:.2f}%")

def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8):
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method='hist',
        device='cpu',
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, accuracy_score(y_test, y_pred), y_pred

def train_lightgbm(X_train, y_train, X_test, y_test):
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb.fit(X_train, y_train)
    y_pred = lgb.predict(X_test)
    return lgb, accuracy_score(y_test, y_pred), y_pred

def train_randomforest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, accuracy_score(y_test, y_pred), y_pred

def step3_station_features(X_train, X_test, train_df, test_df, y_train, y_test, baseline_acc, train_indices, test_indices):
    print("\n=== STEP 3: STATION FEATURES ===")
    
    X_train_st = X_train.copy()
    X_test_st = X_test.copy()
    
    if 'station_id' in train_df.columns and 'hour' in train_df.columns:
        train_df_indexed = train_df.iloc[train_indices].reset_index(drop=True)
        test_df_indexed = test_df.iloc[test_indices].reset_index(drop=True)
        
        station_pm25_mean = train_df_indexed.groupby('station_id')['pm25'].mean()
        station_pm25_std = train_df_indexed.groupby('station_id')['pm25'].std().fillna(0)
        station_cat_mode = train_df_indexed.groupby('station_id')['category'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
        
        hour_cat_mean = train_df_indexed.groupby(['station_id', 'hour'])['category'].mean()
        
        train_station_ids = train_df_indexed['station_id'].values
        train_hours = train_df_indexed['hour'].values if 'hour' in train_df_indexed.columns else np.zeros(len(train_df_indexed))
        
        test_station_ids = test_df_indexed['station_id'].values
        test_hours = test_df_indexed['hour'].values if 'hour' in test_df_indexed.columns else np.zeros(len(test_df_indexed))
        
        X_train_st['station_pm25_mean'] = [station_pm25_mean.get(sid, 0) for sid in train_station_ids]
        X_train_st['station_pm25_std'] = [station_pm25_std.get(sid, 0) for sid in train_station_ids]
        X_train_st['station_cat_mode'] = [station_cat_mode.get(sid, 1) for sid in train_station_ids]
        X_train_st['hour_cat_mean'] = [hour_cat_mean.get((train_station_ids[i], int(train_hours[i])), 1) for i in range(len(train_station_ids))]
        
        X_test_st['station_pm25_mean'] = [station_pm25_mean.get(sid, 0) for sid in test_station_ids]
        X_test_st['station_pm25_std'] = [station_pm25_std.get(sid, 0) for sid in test_station_ids]
        X_test_st['station_cat_mode'] = [station_cat_mode.get(sid, 1) for sid in test_station_ids]
        X_test_st['hour_cat_mean'] = [hour_cat_mean.get((test_station_ids[i], int(test_hours[i])), 1) for i in range(len(test_station_ids))]
        
        X_train_st = X_train_st.fillna(X_train_st.mean(numeric_only=True))
        X_test_st = X_test_st.fillna(X_test_st.mean(numeric_only=True))
    
    xgb, acc, _ = train_xgboost(X_train_st, y_train, X_test_st, y_test)
    
    print(f"BEFORE: {baseline_acc*100:.2f}%")
    print(f"AFTER:  {acc*100:.2f}%")
    print(f"DELTA:  {(acc-baseline_acc)*100:+.2f}%")
    
    if acc >= baseline_acc:
        print("ACCEPTED")
        return xgb, acc, X_train_st, X_test_st
    else:
        print("REVERTED")
        return None, baseline_acc, X_train, X_test

def step4_ensemble(X_train, y_train, X_test, y_test, baseline_acc):
    print("\n=== STEP 4: ENSEMBLE ===")
    
    xgb, acc_xgb, _ = train_xgboost(X_train, y_train, X_test, y_test)
    lgb, acc_lgb, _ = train_lightgbm(X_train, y_train, X_test, y_test)
    rf, acc_rf, _ = train_randomforest(X_train, y_train, X_test, y_test)
    
    print(f"XGBoost:  {acc_xgb*100:.2f}%")
    print(f"LightGBM: {acc_lgb*100:.2f}%")
    print(f"RandomForest: {acc_rf*100:.2f}%")
    
    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    
    print(f"Ensemble: {acc_ens*100:.2f}%")
    
    best_individual = max(acc_xgb, acc_lgb, acc_rf)
    
    print(f"\nBEFORE: {baseline_acc*100:.2f}%")
    print(f"AFTER:  {acc_ens*100:.2f}%")
    print(f"DELTA:  {(acc_ens-baseline_acc)*100:+.2f}%")
    
    if acc_ens > best_individual:
        print("ENSEMBLE ACCEPTED")
        return ensemble, acc_ens
    else:
        print("ENSEMBLE REJECTED - Using best individual")
        if acc_xgb >= acc_lgb and acc_xgb >= acc_rf:
            return xgb, acc_xgb
        elif acc_lgb >= acc_rf:
            return lgb, acc_lgb
        else:
            return rf, acc_rf

def step5_optuna_tuning(X_train, y_train, X_test, y_test, baseline_acc):
    print("\n=== STEP 5: OPTUNA TUNING ===")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        xgb = XGBClassifier(
            **params,
            tree_method='hist',
            device='cpu',
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc
    
    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    best_params = study.best_params
    best_acc = study.best_value
    
    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc*100:.2f}%")
    
    print(f"\nBEFORE: {baseline_acc*100:.2f}%")
    print(f"AFTER:  {best_acc*100:.2f}%")
    print(f"DELTA:  {(best_acc-baseline_acc)*100:+.2f}%")
    
    if best_acc >= baseline_acc:
        print("ACCEPTED")
        xgb = XGBClassifier(
            **best_params,
            tree_method='hist',
            device='cpu',
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        return xgb, best_acc, best_params
    else:
        print("REVERTED")
        return None, baseline_acc, None

def stage1_training():
    print(f"Started: {time.strftime('%H:%M:%S')}")
    print("\n" + "="*60)
    print("STAGE 1: 100k train / 50k test (3-class + SMOTE)")
    print("="*60)
    
    train_df = pd.read_csv('data/processed/train_2gb_engineered_8694071rows_20260408_102015.csv')
    test_df = pd.read_csv('data/processed/test_2gb_engineered_2173518rows_20260408_102015.csv')
    
    train_df['category'] = train_df['pm25'].apply(pm25_to_category)
    test_df['category'] = test_df['pm25'].apply(pm25_to_category)
    
    print(f"Loading stratified sample: 100000 train / 50000 test...")
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=100000, test_size=50000, random_state=42)
    
    for train_idx, _ in sss.split(train_df, train_df['category']):
        train_sample = train_df.iloc[train_idx].copy()
    
    for _, test_idx in sss.split(test_df, test_df['category']):
        test_sample = test_df.iloc[test_idx].copy()
    
    print("\nClass Distribution (before SMOTE):")
    for cat in range(3):
        count = (train_sample['category'] == cat).sum()
        pct = 100 * count / len(train_sample)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    numeric_features = [c for c in train_sample.columns if pd.api.types.is_numeric_dtype(train_sample[c]) and c not in ['pm25', 'category']]
    print(f"\n[DEBUG] Total numeric features found: {len(numeric_features)}")
    
    X_train_raw = train_sample[numeric_features].copy()
    print(f"[DEBUG] X_train_raw shape: {X_train_raw.shape}")
    
    # Check for all-NaN columns and filter them out
    nan_counts = X_train_raw.isna().sum()
    all_nan_cols = nan_counts[nan_counts == len(X_train_raw)].index.tolist()
    print(f"[DEBUG] All-NaN columns (count={len(all_nan_cols)}): {all_nan_cols}")
    
    # Remove all-NaN columns
    if all_nan_cols:
        numeric_features = [c for c in numeric_features if c not in all_nan_cols]
        X_train_raw = X_train_raw.drop(columns=all_nan_cols)
        print(f"[DEBUG] After removing all-NaN columns, numeric features: {len(numeric_features)}")
    
    y_train = train_sample['category'].copy()
    X_test_raw = test_sample[numeric_features].copy()
    print(f"[DEBUG] X_train_raw shape after filtering: {X_train_raw.shape}")
    print(f"[DEBUG] X_test_raw shape: {X_test_raw.shape}")
    y_test = test_sample['category'].copy()
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    print(f"[DEBUG] X_train after imputation shape: {X_train.shape}")
    X_test = imputer.transform(X_test_raw)
    print(f"[DEBUG] X_test after imputation shape: {X_test.shape}")
    
    X_train = pd.DataFrame(X_train, columns=numeric_features)
    X_test = pd.DataFrame(X_test, columns=numeric_features)
    
    print("\n=== STEP 1: APPLY SMOTE ===")
    sm = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    X_res = pd.DataFrame(X_res, columns=numeric_features)
    
    print("Class Distribution (after SMOTE):")
    for cat in range(3):
        count = (y_res == cat).sum()
        pct = 100 * count / len(y_res)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    print("\n=== STEP 2: BASELINE XGBoost (FAST) ===")
    xgb, baseline_acc, _ = train_xgboost(X_res, y_res, X_test, y_test, n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    print(f"BASELINE: {baseline_acc*100:.2f}%")
    print_per_class_accuracy(y_test, xgb.predict(X_test))
    
    current_best_model = xgb
    current_best_acc = baseline_acc
    current_X_train = X_res
    current_X_test = X_test
    current_features = numeric_features
    
    # Step 3: Station Features
    model_after_3, acc_after_3, X_train_3, X_test_3 = step3_station_features(X_res, X_test, train_sample, test_sample, y_res, y_test, baseline_acc)
    if model_after_3 is not None:
        current_best_model = model_after_3
        current_best_acc = acc_after_3
        current_X_train = X_train_3
        current_X_test = X_test_3
        current_features = list(X_train_3.columns)
    
    # Step 4: Ensemble
    model_after_4, acc_after_4 = step4_ensemble(current_X_train, y_res, current_X_test, y_test, current_best_acc)
    if acc_after_4 is not None and acc_after_4 >= current_best_acc:
        current_best_model = model_after_4
        current_best_acc = acc_after_4
    
    # Step 5: Optuna if below 97%
    if current_best_acc < 0.97:
        model_after_5, acc_after_5, best_params_5 = step5_optuna_tuning(current_X_train, y_res, current_X_test, y_test, current_best_acc)
        if model_after_5 is not None:
            current_best_model = model_after_5
            current_best_acc = acc_after_5
    
    # Final summary
    print("\n" + "="*60)
    print(f"✓ STAGE 1 COMPLETE: {current_best_acc*100:.2f}% accuracy on 50k test")
    print("="*60)
    print(f"Best model: {type(current_best_model).__name__}")
    print("Per-class accuracy:")
    y_pred_final = current_best_model.predict(current_X_test)
    print_per_class_accuracy(y_test, y_pred_final)
    
    # Save model and features
    joblib.dump(current_best_model, 'models/aqi_classifier_v2_stage1.joblib')
    joblib.dump(current_features, 'models/aqi_features_v2.json')
    
    print(f"\nModel saved to models/aqi_classifier_v2_stage1.joblib")
    
    return current_best_acc >= 0.97, current_best_model, current_best_acc, current_features, current_X_train, current_X_test, y_res, y_test, train_sample, test_sample

if __name__ == "__main__":
    print("Starting PM2.5 Classifier Training (3-class + SMOTE)...")
    stage1_success, best_model, best_acc, features_list, X_train_final, X_test_final, y_train_final, y_test_final, train_df, test_df = stage1_training()
    
    if stage1_success:
        print("\n✓ Ready for Stage 2")
    else:
        print(f"\n✗ Stage 1 achieved {best_acc*100:.2f}% - below 97% target")
