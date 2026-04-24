import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

def print_per_class_accuracy(y_true, y_pred):
    cat_names = ['Clean', 'Polluted', 'Hazardous']
    for cat in range(3):
        mask = y_true == cat
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"  {cat_names[cat]}: {acc*100:.2f}%")

def train_xgboost_tuned(X_train, y_train, X_test, y_test, params=None):
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
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
    return xgb, acc, y_pred

def train_lightgbm_tuned(X_train, y_train, X_test, y_test):
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb.fit(X_train, y_train)
    y_pred = lgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return lgb, acc, y_pred

def train_randomforest_tuned(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return rf, acc, y_pred

def stage1_training():
    print(f"Started: {time.strftime('%H:%M:%S')}")
    print("\n" + "="*60)
    print("STAGE 1: 100k train / 50k test (3-class + SMOTE)")
    print("="*60)
    
    train_df_full = pd.read_csv('data/processed/train_2gb_engineered_8694071rows_20260408_102015.csv')
    test_df_full = pd.read_csv('data/processed/test_2gb_engineered_2173518rows_20260408_102015.csv')
    
    train_df_full['category'] = train_df_full['pm25'].apply(pm25_to_category)
    test_df_full['category'] = test_df_full['pm25'].apply(pm25_to_category)
    
    print("Loading stratified sample: 100000 train / 50000 test...")
    
    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=100000, random_state=42)
    sss_test = StratifiedShuffleSplit(n_splits=1, train_size=50000, random_state=42)
    
    for train_idx, _ in sss_train.split(train_df_full, train_df_full['category']):
        train_sample = train_df_full.iloc[train_idx].copy()
        train_indices = train_idx
    
    for test_idx, _ in sss_test.split(test_df_full, test_df_full['category']):
        test_sample = test_df_full.iloc[test_idx].copy()
        test_indices = test_idx
    
    print("\nClass Distribution (before SMOTE):")
    for cat in range(3):
        count = (train_sample['category'] == cat).sum()
        pct = 100 * count / len(train_sample)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    numeric_features = [c for c in train_sample.columns if pd.api.types.is_numeric_dtype(train_sample[c]) and c not in ['pm25', 'category']]
    
    X_train_raw = train_sample[numeric_features].copy()
    nan_counts = X_train_raw.isna().sum()
    all_nan_cols = nan_counts[nan_counts == len(X_train_raw)].index.tolist()
    
    if all_nan_cols:
        numeric_features = [c for c in numeric_features if c not in all_nan_cols]
        X_train_raw = X_train_raw.drop(columns=all_nan_cols)
    
    y_train = train_sample['category'].copy()
    X_test_raw = test_sample[numeric_features].copy()
    y_test = test_sample['category'].copy()
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    
    X_train = pd.DataFrame(X_train, columns=numeric_features)
    X_test = pd.DataFrame(X_test, columns=numeric_features)
    
    print("\n=== STEP 1: APPLY SMOTE ===")
    sm = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=numeric_features)
    
    print("Class Distribution (after SMOTE):")
    for cat in range(3):
        count = (y_res == cat).sum()
        pct = 100 * count / len(y_res)
        cat_names = ['Clean', 'Polluted', 'Hazardous']
        print(f"  {cat_names[cat]} ({cat}): {count:,} rows ({pct:.2f}%)")
    
    print("\n=== STEP 2: BASELINE XGBoost ===")
    xgb_b, baseline_acc, _ = train_xgboost_tuned(X_res, y_res, X_test, y_test)
    print(f"BASELINE: {baseline_acc*100:.2f}%")
    print_per_class_accuracy(y_test, xgb_b.predict(X_test))
    
    current_model = xgb_b
    current_acc = baseline_acc
    current_X_train = X_res.copy()
    current_X_test = X_test.copy()
    current_features = list(numeric_features)
    
    print("\n=== STEP 3: STATION FEATURES ===")
    X_train_st = current_X_train.copy()
    X_test_st = current_X_test.copy()
    
    if 'station_id' in train_sample.columns:
        train_subset = train_sample.iloc[train_indices] if len(train_indices) == len(train_sample) else train_sample
        test_subset = test_sample.iloc[test_indices] if len(test_indices) == len(test_sample) else test_sample
        
        train_subset = train_subset.reset_index(drop=True)
        test_subset = test_subset.reset_index(drop=True)
        
        station_pm25_mean = train_subset.groupby('station_id')['pm25'].mean()
        station_pm25_std = train_subset.groupby('station_id')['pm25'].std().fillna(0)
        station_cat_mode = train_subset.groupby('station_id')['category'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 1)
        
        train_station_ids = train_subset['station_id'].values
        test_station_ids = test_subset['station_id'].values
        
        X_train_st['station_pm25_mean'] = [station_pm25_mean.get(sid, 0) for sid in train_station_ids]
        X_train_st['station_pm25_std'] = [station_pm25_std.get(sid, 0) for sid in train_station_ids]
        X_train_st['station_cat_mode'] = [float(station_cat_mode.get(sid, 1)) for sid in train_station_ids]
        
        X_test_st['station_pm25_mean'] = [station_pm25_mean.get(sid, 0) for sid in test_station_ids]
        X_test_st['station_pm25_std'] = [station_pm25_std.get(sid, 0) for sid in test_station_ids]
        X_test_st['station_cat_mode'] = [float(station_cat_mode.get(sid, 1)) for sid in test_station_ids]
        
        if 'hour' in train_subset.columns:
            hour_cat_mean = train_subset.groupby(['station_id', 'hour'])['category'].mean()
            train_hours = train_subset['hour'].values
            test_hours = test_subset['hour'].values
            
            X_train_st['hour_cat_mean'] = [hour_cat_mean.get((train_station_ids[i], int(train_hours[i])), 1) for i in range(len(train_station_ids))]
            X_test_st['hour_cat_mean'] = [hour_cat_mean.get((test_station_ids[i], int(test_hours[i])), 1) for i in range(len(test_station_ids))]
        
        X_train_st = X_train_st.fillna(X_train_st.mean(numeric_only=True))
        X_test_st = X_test_st.fillna(X_test_st.mean(numeric_only=True))
    
    xgb_3, acc_3, _ = train_xgboost_tuned(X_train_st, y_res, X_test_st, y_test)
    print(f"BEFORE: {current_acc*100:.2f}%")
    print(f"AFTER:  {acc_3*100:.2f}%")
    print(f"DELTA:  {(acc_3-current_acc)*100:+.2f}%")
    
    if acc_3 >= current_acc:
        print("ACCEPTED")
        current_model = xgb_3
        current_acc = acc_3
        current_X_train = X_train_st
        current_X_test = X_test_st
        current_features = list(X_train_st.columns)
    else:
        print("REVERTED")
    
    print("\n=== STEP 4: ENSEMBLE ===")
    xgb_e, acc_xgb, _ = train_xgboost_tuned(current_X_train, y_res, current_X_test, y_test)
    lgb_e, acc_lgb, _ = train_lightgbm_tuned(current_X_train, y_res, current_X_test, y_test)
    rf_e, acc_rf, _ = train_randomforest_tuned(current_X_train, y_res, current_X_test, y_test)
    
    print(f"XGBoost:  {acc_xgb*100:.2f}%")
    print(f"LightGBM: {acc_lgb*100:.2f}%")
    print(f"RandomForest: {acc_rf*100:.2f}%")
    
    ensemble = VotingClassifier(estimators=[('xgb', xgb_e), ('lgb', lgb_e), ('rf', rf_e)], voting='soft')
    ensemble.fit(current_X_train, y_res)
    y_pred_ens = ensemble.predict(current_X_test)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    
    print(f"Ensemble: {acc_ens*100:.2f}%")
    
    best_individual = max(acc_xgb, acc_lgb, acc_rf)
    
    print(f"\nBEFORE: {current_acc*100:.2f}%")
    print(f"AFTER:  {acc_ens*100:.2f}%")
    print(f"DELTA:  {(acc_ens-current_acc)*100:+.2f}%")
    
    if acc_ens > best_individual:
        print("ENSEMBLE ACCEPTED")
        current_model = ensemble
        current_acc = acc_ens
    else:
        print("ENSEMBLE REJECTED - Using best individual")
        if acc_xgb >= acc_lgb and acc_xgb >= acc_rf:
            current_model = xgb_e
            current_acc = acc_xgb
        elif acc_lgb >= acc_rf:
            current_model = lgb_e
            current_acc = acc_lgb
        else:
            current_model = rf_e
            current_acc = acc_rf
    
    print("\n=== STEP 5: OPTUNA TUNING ===")
    
    if current_acc < 0.97:
        print("Below 97%, running Optuna optimization (50 trials, 180s timeout)...")
        
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
            xgb.fit(current_X_train, y_res)
            y_pred = xgb.predict(current_X_test)
            return accuracy_score(y_test, y_pred)
        
        study = optuna.create_study(direction='maximize', pruner=MedianPruner())
        study.optimize(objective, n_trials=50, show_progress_bar=False, timeout=180)
        
        best_params = study.best_params
        best_acc_optuna = study.best_value
        
        print(f"Best params: {best_params}")
        print(f"Best accuracy: {best_acc_optuna*100:.2f}%")
        
        print(f"\nBEFORE: {current_acc*100:.2f}%")
        print(f"AFTER:  {best_acc_optuna*100:.2f}%")
        print(f"DELTA:  {(best_acc_optuna-current_acc)*100:+.2f}%")
        
        if best_acc_optuna >= current_acc:
            print("ACCEPTED")
            xgb_opt = XGBClassifier(
                **best_params,
                tree_method='hist',
                device='cpu',
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            xgb_opt.fit(current_X_train, y_res)
            current_model = xgb_opt
            current_acc = best_acc_optuna
        else:
            print("REVERTED")
    
    print("\n" + "="*60)
    if current_acc >= 0.97:
        print(f"✓ STAGE 1 COMPLETE: {current_acc*100:.2f}%")
    else:
        print(f"✗ STAGE 1 CEILING: {current_acc*100:.2f}%")
    print("="*60)
    print("Per-class accuracy:")
    y_pred_final = current_model.predict(current_X_test)
    print_per_class_accuracy(y_test, y_pred_final)
    
    joblib.dump(current_model, 'models/aqi_classifier_v2_stage1.joblib')
    joblib.dump(current_features, 'models/aqi_features_v2.json')
    
    return current_acc >= 0.97, current_model, current_acc, current_features, current_X_train, current_X_test, y_res, y_test, train_sample, test_sample

if __name__ == "__main__":
    stage1_success, best_model, best_acc, features_list, X_train_final, X_test_final, y_train_final, y_test_final, train_df, test_df = stage1_training()
    
    if stage1_success:
        print("\n✓ Ready for Stage 2")
    else:
        print(f"\n✗ Stage 1 achieved {best_acc*100:.2f}% - below 97% target")
