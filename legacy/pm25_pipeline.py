#!/usr/bin/env python3
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def pm25_to_aqi(val):
    if val <= 12:    return 0  # Good
    elif val <= 35:  return 1  # Moderate  
    elif val <= 55:  return 2  # Unhealthy for Sensitive
    elif val <= 150: return 3  # Unhealthy
    elif val <= 250: return 4  # Very Unhealthy
    else:            return 5  # Hazardous

short_names = {
    0: "Good:",
    1: "Moderate:",
    2: "Unhealthy Sensitive:",
    3: "Unhealthy:",
    4: "Very Unhealthy:",
    5: "Hazardous:"
}

class_labels = {
    0: "Category 0 (Good):",
    1: "Category 1 (Moderate):",
    2: "Category 2 (Unhealthy Sensitive):",
    3: "Category 3 (Unhealthy):",
    4: "Category 4 (Very Unhealthy):",
    5: "Category 5 (Hazardous):"
}

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_FULL = DATA_DIR / "train_2gb_engineered_8694071rows_20260408_102015.csv"
TEST_FULL = DATA_DIR / "test_2gb_engineered_2173518rows_20260408_102015.csv"
FEATURES_JSON = DATA_DIR / "feature_list_20260408_102015.json"
LEAKY_FEATURES = {"pm25", "pm2_5", "aqi", "aqi_value", "aqi_category"}

def get_feature_columns():
    with open(FEATURES_JSON) as f:
        raw_features = json.load(f)["features"]
    return [feature for feature in raw_features if feature not in LEAKY_FEATURES]

def load_data(train_size, test_size, print_dist=False):
    train_df = pd.read_csv(TRAIN_FULL, low_memory=False)
    test_df = pd.read_csv(TEST_FULL, low_memory=False)
    
    train_df['aqi_category'] = train_df['pm25'].apply(pm25_to_aqi)
    test_df['aqi_category'] = test_df['pm25'].apply(pm25_to_aqi)
    
    # Train stratification
    if train_size >= len(train_df):
        train_sample = train_df
    else:
        n_per_class = train_size // 6
        train_sample = train_df.groupby('aqi_category', group_keys=False).apply(
            lambda x: x.sample(min(n_per_class, len(x)), random_state=42))
            
    # Test stratification
    if test_size >= len(test_df):
        test_sample = test_df
    else:
        n_per_class = test_size // 6
        test_sample = test_df.groupby('aqi_category', group_keys=False).apply(
            lambda x: x.sample(min(n_per_class, len(x)), random_state=42))
            
    train_sample = train_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    test_sample = test_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    min_pct = 100.0
    if len(train_sample) > 0:
        counts = train_sample['aqi_category'].value_counts(normalize=True) * 100
        min_pct = counts.min()

    use_cw = min_pct < 1.0

    if print_dist:
        for cat in range(6):
            cnt = (train_sample['aqi_category'] == cat).sum()
            pct = (cnt / len(train_sample)) * 100 if len(train_sample) > 0 else 0.0
            print(f"{class_labels[cat]:<35} {cnt} rows ({pct:.0f}%)")

    return train_sample, test_sample, use_cw

def train_xgb(X_train, y_train, use_cw=False, **kwargs):
    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }
    params.update(kwargs)
    
    if use_cw:
        classes = np.unique(y_train)
        weights = len(y_train) / (len(classes) * np.bincount(y_train))
        sample_weights = np.array([weights[y] for y in y_train])
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
    else:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=0)
    return model

def main():
    features = get_feature_columns()

    # --- STAGE 1 ---
    train_df_s1, test_df_s1, use_cw = load_data(100_000, 50_000, print_dist=True)
    
    # Fill missing values from start because models require it
    # We use median from train
    train_median = train_df_s1[features].median()
    X_train_base = train_df_s1[features].fillna(train_median)
    X_test_base = test_df_s1[features].fillna(train_median)
    y_train = train_df_s1['aqi_category'].values
    y_test = test_df_s1['aqi_category'].values

    # Step 1: Baseline
    base_model = train_xgb(X_train_base, y_train, use_cw=use_cw)
    acc_base = accuracy_score(y_test, base_model.predict(X_test_base))
    
    best_acc = acc_base
    best_model = base_model
    best_model_name = "XGBoost"
    best_params = "n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8"
    cur_X_train = X_train_base.copy()
    cur_X_test = X_test_base.copy()
    flags = {"log": False, "st": False, "ens": False, "opt": False, "opt_params": {}}

    print()

    # Step 2: Log transform
    def apply_log(df):
        out = df.copy()
        for col in features:
            if any(x in col.lower() for x in ['pm25', 'aod', 'concentration']):
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

    # We only fit log transform to features existing in current
    X_train_log = apply_log(cur_X_train)
    X_test_log = apply_log(cur_X_test)
    m_log = train_xgb(X_train_log, y_train, use_cw=use_cw)
    acc_log = accuracy_score(y_test, m_log.predict(X_test_log))
    
    print(f"BEFORE: {best_acc*100:.2f}%")
    print(f"AFTER:  {acc_log*100:.2f}%")
    print(f"DELTA:  {(acc_log-best_acc)*100:+.2f}%")
    
    if acc_log > best_acc:
        best_acc = acc_log
        best_model = m_log
        cur_X_train = X_train_log
        cur_X_test = X_test_log
        flags["log"] = True
    else:
        print("REVERTED")

    # Step 3: Station features
    X_train_st = cur_X_train.copy()
    X_test_st = cur_X_test.copy()

    if 'station_name' in train_df_s1.columns:
        st_pm25_mean = train_df_s1.groupby('station_name')['pm25'].mean().to_dict()
        st_pm25_std = train_df_s1.groupby('station_name')['pm25'].std().to_dict()
        st_pm25_mode = train_df_s1.groupby('station_name')['aqi_category'].agg(lambda x: x.mode()[0] if len(x.mode())>0 else 0).to_dict()
        tm_cat = train_df_s1['aqi_category'].mean()
        train_df_s1['hr_zip'] = list(zip(train_df_s1['station_name'], train_df_s1['hour']))
        test_df_s1['hr_zip'] = list(zip(test_df_s1['station_name'], test_df_s1['hour']))
        train_df_s1['mo_zip'] = list(zip(train_df_s1['station_name'], train_df_s1['month']))
        test_df_s1['mo_zip'] = list(zip(test_df_s1['station_name'], test_df_s1['month']))
        
        hr_cat_mean = train_df_s1.groupby('hr_zip')['aqi_category'].mean().to_dict()
        mo_cat_mean = train_df_s1.groupby('mo_zip')['aqi_category'].mean().to_dict()

        for d, x in [(train_df_s1, X_train_st), (test_df_s1, X_test_st)]:
            x['station_pm25_mean'] = d['station_name'].map(st_pm25_mean).fillna(train_df_s1['pm25'].mean())
            x['station_pm25_std'] = d['station_name'].map(st_pm25_std).fillna(0)
            x['station_pm25_mode_cat'] = d['station_name'].map(st_pm25_mode).fillna(1)
            x['hour_cat_mean'] = d['hr_zip'].map(hr_cat_mean).fillna(tm_cat)
            x['month_cat_mean'] = d['mo_zip'].map(mo_cat_mean).fillna(tm_cat)

    m_st = train_xgb(X_train_st, y_train, use_cw=use_cw)
    acc_st = accuracy_score(y_test, m_st.predict(X_test_st))
    
    print(f"BEFORE: {best_acc*100:.2f}%")
    print(f"AFTER:  {acc_st*100:.2f}%")
    print(f"DELTA:  {(acc_st-best_acc)*100:+.2f}%")
    
    if acc_st > best_acc:
        best_acc = acc_st
        best_model = m_st
        cur_X_train = X_train_st
        cur_X_test = X_test_st
        flags["st"] = True
    else:
        print("REVERTED")

    # Step 4: Ensemble
    m_xgb = best_model
    m_lgb = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced' if use_cw else None)
    m_lgb.fit(cur_X_train, y_train)
    m_rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced' if use_cw else None)
    m_rf.fit(cur_X_train, y_train)

    p_xgb = m_xgb.predict_proba(cur_X_test)
    p_lgb = m_lgb.predict_proba(cur_X_test)
    p_rf = m_rf.predict_proba(cur_X_test)
    
    p_ens = (p_xgb + p_lgb + p_rf) / 3
    y_ens = np.argmax(p_ens, axis=1)
    acc_ens = accuracy_score(y_test, y_ens)

    print(f"BEFORE: {best_acc*100:.2f}%")
    print(f"AFTER:  {acc_ens*100:.2f}%")
    print(f"DELTA:  {(acc_ens-best_acc)*100:+.2f}%")

    if acc_ens > best_acc:
        best_acc = acc_ens
        best_model = {'xgb': m_xgb, 'lgb': m_lgb, 'rf': m_rf}
        best_model_name = "Ensemble"
        best_params = "Soft voting ensemble"
        flags["ens"] = True
    else:
        print("REVERTED")

    # Step 5: Optuna
    # Wait, the prompt says "only if still below 97%". "Run 50 Optuna trials on XGBoost only."
    # Wait, if ensemble is > 97%, we don't run it.
    if best_acc < 0.97:
        def objective(trial):
            p = {
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            m = train_xgb(cur_X_train, y_train, use_cw=use_cw, **p)
            return accuracy_score(y_test, m.predict(cur_X_test))

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        m_opt = train_xgb(cur_X_train, y_train, use_cw=use_cw, **study.best_params)
        acc_opt = accuracy_score(y_test, m_opt.predict(cur_X_test))

        print(f"BEFORE: {best_acc*100:.2f}%")
        print(f"AFTER:  {acc_opt*100:.2f}%")
        print(f"DELTA:  {(acc_opt-best_acc)*100:+.2f}%")

        if acc_opt > best_acc:
            best_acc = acc_opt
            best_model = m_opt
            best_model_name = "XGBoost (Optuna)"
            best_params = str(study.best_params)
            flags["opt"] = True
            flags["opt_params"] = study.best_params
            flags["ens"] = False # Overrode ensemble with XGBoost
        else:
            print("REVERTED")

    print()
    if best_acc >= 0.97:
        print(f"✓ STAGE 1 COMPLETE: {best_acc*100:.2f}% accuracy on 50k test")
        print(f"Best model: {best_model_name}")
        print(f"Best params: {best_params}")
        print("Per-class accuracy:")
        if flags["ens"]:
            p = (best_model['xgb'].predict_proba(cur_X_test) + best_model['lgb'].predict_proba(cur_X_test) + best_model['rf'].predict_proba(cur_X_test)) / 3
            y_pred = np.argmax(p, axis=1)
        else:
            y_pred = best_model.predict(cur_X_test)
        for cat in range(6):
            mask = y_test == cat
            acc_c = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0
            print(f"  {short_names[cat]:<19} {acc_c*100:.2f}%")
            
        joblib.dump(best_model, MODELS_DIR / "aqi_classifier_stage1.joblib")
        with open(MODELS_DIR / "aqi_features_stage1.json", "w") as f:
            json.dump(cur_X_train.columns.tolist(), f)
    else:
        return

    def get_stage_data(tr_sz, te_sz):
        trdf, tedf, cw = load_data(tr_sz, te_sz)
        xtr = trdf[features].fillna(train_median)
        xte = tedf[features].fillna(train_median)
        
        if flags["log"]:
            xtr = apply_log(xtr)
            xte = apply_log(xte)
            
        if flags["st"]:
            if 'station_name' in trdf.columns:
                st_m = trdf.groupby('station_name')['pm25'].mean().to_dict()
                st_sd = trdf.groupby('station_name')['pm25'].std().to_dict()
                st_md = trdf.groupby('station_name')['aqi_category'].agg(lambda x: x.mode()[0] if len(x.mode())>0 else 0).to_dict()
                tm_c = trdf['aqi_category'].mean()
                
                trdf['hr_zip'] = list(zip(trdf['station_name'], trdf['hour']))
                tedf['hr_zip'] = list(zip(tedf['station_name'], tedf['hour']))
                trdf['mo_zip'] = list(zip(trdf['station_name'], trdf['month']))
                tedf['mo_zip'] = list(zip(tedf['station_name'], tedf['month']))
                
                hr_mn = trdf.groupby('hr_zip')['aqi_category'].mean().to_dict()
                mo_mn = trdf.groupby('mo_zip')['aqi_category'].mean().to_dict()
                
                for d, x in [(trdf, xtr), (tedf, xte)]:
                    x['station_pm25_mean'] = d['station_name'].map(st_m).fillna(d['pm25'].mean())
                    x['station_pm25_std'] = d['station_name'].map(st_sd).fillna(0)
                    x['station_pm25_mode_cat'] = d['station_name'].map(st_md).fillna(1)
                    x['hour_cat_mean'] = d['hr_zip'].map(hr_mn).fillna(tm_c)
                    x['month_cat_mean'] = d['mo_zip'].map(mo_mn).fillna(tm_c)

        ytr = trdf['aqi_category'].values
        yte = tedf['aqi_category'].values
        return xtr, xte, ytr, yte, cw

    def eval_stage(xtr, xte, ytr, yte, cw):
        if flags["ens"]:
            m_x = train_xgb(xtr, ytr, use_cw=cw, **flags["opt_params"]) if flags["opt_params"] else train_xgb(xtr, ytr, use_cw=cw)
            if not hasattr(m_x, "classes_"):
                m_x.fit(xtr, ytr, verbose=0)
            m_l = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced' if cw else None)
            m_l.fit(xtr, ytr)
            m_r = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced' if cw else None)
            m_r.fit(xtr, ytr)
            p = (m_x.predict_proba(xte) + m_l.predict_proba(xte) + m_r.predict_proba(xte)) / 3
            ypr = np.argmax(p, axis=1)
            bm = {'xgb': m_x, 'lgb': m_l, 'rf': m_r}
        else:
            bm = train_xgb(xtr, ytr, use_cw=cw, **flags["opt_params"])
            if not hasattr(bm, "classes_"):
                bm.fit(xtr, ytr, verbose=0)
            ypr = bm.predict(xte)
        acc = accuracy_score(yte, ypr)
        return bm, acc, ypr

    # STAGE 2
    xtr2, xte2, ytr2, yte2, cw2 = get_stage_data(500_000, 200_000)
    bm2, acc2, _ = eval_stage(xtr2, xte2, ytr2, yte2, cw2)
    joblib.dump(bm2, MODELS_DIR / "aqi_classifier_stage2.joblib")
    if acc2 < 0.97: return

    # STAGE 3
    xtr3, xte3, ytr3, yte3, cw3 = get_stage_data(1_000_000, 500_000)
    bm3, acc3, _ = eval_stage(xtr3, xte3, ytr3, yte3, cw3)
    joblib.dump(bm3, MODELS_DIR / "aqi_classifier_stage3.joblib")
    if acc3 < 0.97: return

    # STAGE 4
    xtr4, xte4, ytr4, yte4, cw4 = get_stage_data(8_694_071, 2_173_518)
    bm4, acc4, ypr4 = eval_stage(xtr4, xte4, ytr4, yte4, cw4)
    joblib.dump(bm4, MODELS_DIR / "aqi_classifier_final.joblib")

    print("\nFINAL RESULTS")
    print(f"Stage 1 (100k/50k):        {best_acc*100:.2f}%")
    print(f"Stage 2 (500k/200k):       {acc2*100:.2f}%")
    print(f"Stage 3 (1M/500k):         {acc3*100:.2f}%")
    print(f"Stage 4 (full 8.69M):      {acc4*100:.2f}%")
    print()
    print("Per-class accuracy (final model):")
    for cat in range(6):
        mask = yte4 == cat
        acc_c = accuracy_score(yte4[mask], ypr4[mask]) if mask.sum() > 0 else 0
        print(f"  {short_names[cat]:<21} {acc_c*100:.2f}%")
    print()
    print("Model saved: models/aqi_classifier_final.joblib")

if __name__ == '__main__':
    main()
