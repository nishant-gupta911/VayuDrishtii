#!/usr/bin/env python3
"""
🔧 DATA PREPROCESSING PIPELINE
Cleans, validates, and prepares data for machine learning
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')  # Changed from '../data' to 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("🔧 DATA PREPROCESSING PIPELINE")
print("=" * 100)


def load_merged_data():
    """Load merged data from previous step"""
    print("\n1️⃣  Loading merged data...")
    
    merged_files = list((DATA_DIR / 'merged').glob('unified_dataset_*.csv'))
    
    if not merged_files:
        # Try loading from ml_ready or other sources
        merged_files = list((DATA_DIR / 'ml_ready').glob('*.csv'))
    
    if merged_files:
        # Load the most recent file
        file = sorted(merged_files)[-1]
        df = pd.read_csv(file)
        print(f"   ✅ Loaded: {file.name} ({len(df)} records)")
        return df
    else:
        print("   ❌ No merged data found. Run merge_all_data.py first!")
        return pd.DataFrame()


def handle_missing_values(df):
    """Handle missing values with multiple strategies"""
    print("\n2️⃣  Handling missing values...")
    
    print(f"   Missing values before:")
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 0:
            print(f"      {col}: {missing_pct:.1f}%")
    
    # Keep target variable (pm2_5) rows only
    df = df[df['pm2_5'].notna()].copy()
    print(f"\n   Rows with PM2.5 target: {len(df)}")
    
    # Forward fill and backward fill for datetime-based data (only if column exists)
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    
    # Handle weather columns only if they exist
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Fill AOD with spatial interpolation (use city/region averages)
    for col in ['aod_550', 'aod_380']:
        if col in df.columns:
            # Fill with city averages
            if 'city' in df.columns:
                city_means = df.groupby('city')[col].transform('mean')
                df[col] = df[col].fillna(city_means)
            # Fill remaining with global mean
            df[col] = df[col].fillna(df[col].mean())
    
    # Fill any remaining with column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    print(f"\n   ✅ Missing values filled")
    print(f"   Remaining NaNs: {df.isna().sum().sum()}")
    
    return df


def remove_outliers(df, method='iqr'):
    """Remove statistical outliers"""
    print("\n3️⃣  Removing outliers...")
    
    initial_rows = len(df)
    
    if method == 'iqr':
        # IQR method
        Q1 = df['pm2_5'].quantile(0.25)
        Q3 = df['pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df['pm2_5'] >= lower_bound) & (df['pm2_5'] <= upper_bound)]
        
        print(f"   IQR method: {lower_bound:.1f} - {upper_bound:.1f} µg/m³")
    
    elif method == 'zscore':
        # Z-score method (3 sigma)
        z_scores = np.abs((df['pm2_5'] - df['pm2_5'].mean()) / df['pm2_5'].std())
        df = df[z_scores < 3]
        
        print(f"   Z-score method: |Z| < 3")
    
    removed = initial_rows - len(df)
    print(f"   ✅ Removed {removed} outliers ({removed/initial_rows*100:.1f}%)")
    print(f"   Remaining: {len(df)} records")
    
    return df


def validate_data_ranges(df):
    """Validate data is within reasonable ranges"""
    print("\n4️⃣  Validating data ranges...")
    
    validations = {
        'pm2_5': (0, 500),  # µg/m³
        'temperature': (-20, 55),  # °C
        'humidity': (0, 100),  # %
        'wind_speed': (0, 50),  # m/s
        'pressure': (87000, 107000),  # Pa
        'aod_550': (0, 2.0),  # unitless
        'latitude': (8, 37),  # India bounds
        'longitude': (68, 97)  # India bounds
    }
    
    invalid_count = 0
    for col, (min_val, max_val) in validations.items():
        if col in df.columns:
            invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if invalid > 0:
                print(f"   ⚠️  {col}: {invalid} invalid values (outside {min_val}-{max_val})")
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
                invalid_count += invalid
    
    print(f"   ✅ Validation complete. Removed {invalid_count} invalid records")
    print(f"   Final records: {len(df)}")
    
    return df


def feature_engineering(df):
    """Engineer temporal and spatial features"""
    print("\n5️⃣  Engineering features...")
    
    features_created = 0
    
    # Temporal features
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour.fillna(0).astype(int)
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['dayofyear'] = df['datetime'].dt.dayofyear
            
            # Seasonal feature (0-3: Winter, Spring, Summer, Autumn)
            df['season'] = ((df['month'] % 12) // 3).astype(int)
            
            # Time of day
            df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
            df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
            df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
            
            features_created += 10
        except:
            print(f"   ⚠️  Could not create temporal features")
    
    # Spatial features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            # Distance from major metros (for urban effect modeling)
            major_cities = {
                'Delhi': (28.7041, 77.1025),
                'Mumbai': (19.0760, 72.8777),
                'Bangalore': (12.9716, 77.5946),
            }
            
            for city, (city_lat, city_lon) in major_cities.items():
                df[f'distance_to_{city.lower()}'] = np.sqrt(
                    (df['latitude'] - city_lat) ** 2 +
                    (df['longitude'] - city_lon) ** 2
                )
                features_created += 1
        except:
            print(f"   ⚠️  Could not create spatial features")
    
    # Meteorological derived features (only if weather columns exist)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        try:
            # Dew point approximation
            df['dew_point'] = df['temperature'] - ((100 - df['humidity']) / 5)
            
            # Temperature humidity index
            df['temp_humidity_index'] = df['temperature'] + 0.5555 * (df['humidity'] / 100 * (6.112 * np.exp((17.62 * df['temperature']) / (df['temperature'] + 243.12))) - 10)
            
            features_created += 2
        except:
            pass
    
    if 'wind_speed' in df.columns:
        try:
            # Wind categories
            df['wind_calm'] = (df['wind_speed'] < 1).astype(int)
            df['wind_light'] = ((df['wind_speed'] >= 1) & (df['wind_speed'] < 5)).astype(int)
            df['wind_moderate'] = ((df['wind_speed'] >= 5) & (df['wind_speed'] < 10)).astype(int)
            df['wind_strong'] = (df['wind_speed'] >= 10).astype(int)
            
            features_created += 4
        except:
            pass
    
    # AOD-based features
    if 'aod_550' in df.columns and 'aod_380' in df.columns:
        try:
            df['aod_ratio'] = df['aod_380'] / (df['aod_550'] + 1e-6)  # Angstrom exponent proxy
            features_created += 1
        except:
            pass
    
    print(f"   ✅ Created {features_created} new features")
    print(f"   Total columns: {len(df.columns)}")
    
    return df


def normalize_features(df):
    """Normalize/standardize numeric features"""
    print("\n6️⃣  Normalizing features...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    normalization_stats = {}
    
    for col in numeric_cols:
        if col in ['pm2_5']:  # Don't normalize target variable
            continue
        
        mean = df[col].mean()
        std = df[col].std()
        
        if std > 0:
            df[f'{col}_normalized'] = (df[col] - mean) / std
            normalization_stats[col] = {'mean': float(mean), 'std': float(std)}
    
    print(f"   ✅ Normalized {len(normalization_stats)} features")
    
    return df, normalization_stats


def detect_and_handle_duplicates(df):
    """Detect and remove duplicate records"""
    print("\n7️⃣  Handling duplicates...")
    
    initial_rows = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Remove near-duplicates (same location, same hour, very similar PM2.5) only if datetime exists and is valid
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if df['datetime'].notna().sum() > 0:  # Only if we have valid datetime values
                df['hour_rounded'] = df['datetime'].dt.floor('h')
                duplicate_subset = df.duplicated(subset=['latitude', 'longitude', 'hour_rounded'], keep='first')
                df = df[~duplicate_subset]
                df = df.drop(columns=['hour_rounded'], errors='ignore')
        except:
            pass  # Skip duplicate detection if datetime cannot be processed
    
    removed = initial_rows - len(df)
    print(f"   ✅ Removed {removed} duplicates ({removed/initial_rows*100:.1f}%)")
    
    return df


def generate_summary_report(df, output_file):
    """Generate preprocessing summary report"""
    print("\n8️⃣  Generating summary report...")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'columns': list(df.columns),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns},
        'statistics': {
            'pm2_5': {
                'mean': float(df['pm2_5'].mean()),
                'std': float(df['pm2_5'].std()),
                'min': float(df['pm2_5'].min()),
                'max': float(df['pm2_5'].max()),
                'median': float(df['pm2_5'].median())
            }
        },
        'spatial_coverage': {
            'cities': int(df['city'].nunique()) if 'city' in df.columns else 'N/A',
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())] if 'latitude' in df.columns else 'N/A',
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())] if 'longitude' in df.columns else 'N/A'
        },
        'temporal_coverage': {
            'start': str(df['datetime'].min()) if 'datetime' in df.columns else 'N/A',
            'end': str(df['datetime'].max()) if 'datetime' in df.columns else 'N/A'
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = PROCESSED_DIR / f'preprocessing_summary_{timestamp}.json'
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved: {summary_file}")
    
    return summary


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("\n🚀 Starting preprocessing...\n")
    
    # Load merged data
    df = load_merged_data()
    
    if df.empty:
        print("❌ Cannot proceed without data!")
        return None
    
    print(f"\n   Initial records: {len(df)}")
    
    # Apply preprocessing steps
    df = handle_missing_values(df)
    df = detect_and_handle_duplicates(df)
    df = validate_data_ranges(df)
    df = remove_outliers(df, method='iqr')
    df = feature_engineering(df)
    df, norm_stats = normalize_features(df)
    
    # Save processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROCESSED_DIR / f'preprocessed_data_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved preprocessed data: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Generate summary
    summary = generate_summary_report(df, output_file)
    
    # Save normalization stats for later use
    norm_file = PROCESSED_DIR / f'normalization_stats_{timestamp}.json'
    with open(norm_file, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print("\n" + "=" * 100)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 100)
    print(f"\n📊 Preprocessed data: {output_file}")
    print(f"   Records: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    print(f"\n📌 Next step: Run feature_engineering.py then train with training script")
    
    return output_file


if __name__ == "__main__":
    main()
