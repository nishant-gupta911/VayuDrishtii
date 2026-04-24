#!/usr/bin/env python3
"""
📊 MERGE ALL DATA SOURCES
Combines PM2.5, Satellite AOD, and Weather data into unified datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings

from src.pipeline import vectorized_merge_nearest

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')  # Changed from '../data' to 'data'
MERGED_DIR = DATA_DIR / 'merged'
MERGED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("📊 DATA MERGING PIPELINE")
print("=" * 100)


def load_pm25_data():
    """Load all PM2.5 data from CPCB and OpenAQ sources"""
    print("\n1️⃣  Loading PM2.5 Ground Truth Data...")
    
    pm25_files = list((DATA_DIR / 'cpcb').glob('*.csv'))
    
    dfs = []
    for file in pm25_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"   ✅ Loaded: {file.name} ({len(df)} records)")
        except Exception as e:
            print(f"   ❌ Error loading {file.name}: {e}")
    
    if dfs:
        pm25_df = pd.concat(dfs, ignore_index=True)
        print(f"\n   📊 Total PM2.5 records: {len(pm25_df)}")
        return pm25_df
    else:
        print("   ⚠️  No PM2.5 data found")
        return pd.DataFrame()


def load_satellite_aod_data():
    """Load satellite AOD data (primarily from CSV demo files)"""
    print("\n2️⃣  Loading Satellite AOD Data...")
    
    satellite_dir = DATA_DIR / 'satellite'
    aod_files = list(satellite_dir.glob('*.csv'))
    
    dfs = []
    for file in aod_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"   ✅ Loaded: {file.name} ({len(df)} records)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if dfs:
        aod_df = pd.concat(dfs, ignore_index=True)
        print(f"\n   📊 Total AOD records: {len(aod_df)}")
        return aod_df
    else:
        print("   ⚠️  No satellite AOD data found")
        return generate_sample_aod_data()


def load_weather_data():
    """Load weather/meteorological data"""
    print("\n3️⃣  Loading Weather Data...")
    
    weather_dir = DATA_DIR / 'weather'
    if not weather_dir.exists():
        weather_dir.mkdir(exist_ok=True)
    
    weather_files = list(weather_dir.glob('*.csv'))
    
    dfs = []
    for file in weather_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"   ✅ Loaded: {file.name} ({len(df)} records)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if dfs:
        weather_df = pd.concat(dfs, ignore_index=True)
        print(f"\n   📊 Total weather records: {len(weather_df)}")
        return weather_df
    else:
        print("   ⚠️  No weather data found. Generating sample...")
        return generate_sample_weather_data()


def generate_sample_aod_data(num_records=5000):
    """Generate realistic sample AOD data if files not available"""
    print("   Generating sample AOD data...")
    
    lat_min, lat_max = 8.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='12H')
    
    data = {
        'date': np.random.choice(dates, num_records),
        'latitude': np.random.uniform(lat_min, lat_max, num_records),
        'longitude': np.random.uniform(lon_min, lon_max, num_records),
        'aod_550': np.clip(np.random.normal(0.35, 0.25), 0.01, 1.0),
        'aod_380': np.clip(np.random.normal(0.45, 0.30), 0.01, 1.5),
        'source': np.random.choice(['MODIS_TERRA', 'MODIS_AQUA'], num_records)
    }
    
    return pd.DataFrame(data)


def generate_sample_weather_data(num_records=5000):
    """Generate realistic sample weather data if files not available"""
    print("   Generating sample weather data...")
    
    lat_min, lat_max = 8.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='6H')
    
    data = {
        'date': np.random.choice(dates, num_records),
        'latitude': np.random.uniform(lat_min, lat_max, num_records),
        'longitude': np.random.uniform(lon_min, lon_max, num_records),
        'temperature_2m': np.random.uniform(5, 45, num_records),
        'relative_humidity': np.random.uniform(20, 95, num_records),
        'wind_speed_10m': np.random.uniform(0, 15, num_records),
        'surface_pressure': np.random.uniform(95000, 105000, num_records),
        'total_precipitation': np.random.gamma(2, 2, num_records)
    }
    
    return pd.DataFrame(data)


def standardize_column_names(df, data_type='pm25'):
    """Standardize column names across different sources"""
    
    column_mapping = {
        'pm25': {
            'city': 'city',
            'location': 'location',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'pm2_5': 'pm2_5',
            'datetime': 'datetime',
            'date': 'datetime'
        },
        'aod': {
            'latitude': 'latitude',
            'longitude': 'longitude',
            'date': 'datetime',
            'aod_550': 'aod_550',
            'aod_380': 'aod_380'
        },
        'weather': {
            'latitude': 'latitude',
            'longitude': 'longitude',
            'date': 'datetime',
            'temperature_2m': 'temperature',
            'relative_humidity': 'humidity',
            'wind_speed_10m': 'wind_speed',
            'surface_pressure': 'pressure'
        }
    }
    
    mapping = column_mapping.get(data_type, {})
    
    # Rename columns that exist
    rename_dict = {old: new for old, new in mapping.items() if old in df.columns}
    df = df.rename(columns=rename_dict)
    
    return df


def merge_all_data(pm25_df, aod_df, weather_df):
    """
    Merge PM2.5, AOD, and Weather data by spatial-temporal proximity
    """
    print("\n4️⃣  Merging all data sources...")
    
    # Standardize column names
    pm25_df = standardize_column_names(pm25_df, 'pm25')
    aod_df = standardize_column_names(aod_df, 'aod')
    weather_df = standardize_column_names(weather_df, 'weather')
    
    # Convert datetime columns
    for df in [pm25_df, aod_df, weather_df]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
    
    merged_df = pm25_df.copy()
    merged_df = merged_df.dropna(subset=['latitude', 'longitude', 'datetime']).sort_values('datetime')

    print("   Merging PM2.5 with Satellite AOD...")
    if not aod_df.empty and {'datetime', 'latitude', 'longitude'}.issubset(aod_df.columns):
        aod_df = aod_df.dropna(subset=['latitude', 'longitude', 'datetime']).sort_values('datetime')
        merged_df = vectorized_merge_nearest(
            merged_df,
            aod_df,
            time_column='datetime',
            latitude_column='latitude',
            longitude_column='longitude',
            tolerance=pd.Timedelta('12H'),
            suffix='_aod',
        )
        distance_mask = merged_df['merge_distance_deg'].fillna(np.inf) < 1.0
        aod_550_source = 'aod_550_aod' if 'aod_550_aod' in merged_df.columns else 'aod_550'
        aod_380_source = 'aod_380_aod' if 'aod_380_aod' in merged_df.columns else 'aod_380'
        if aod_550_source in merged_df.columns:
            merged_df['aod_550'] = merged_df[aod_550_source].where(distance_mask, np.nan)
        if aod_380_source in merged_df.columns:
            merged_df['aod_380'] = merged_df[aod_380_source].where(distance_mask, np.nan)
        merged_df = merged_df.drop(
            columns=[col for col in ['latitude_aod', 'longitude_aod', 'merge_distance_deg'] if col in merged_df.columns],
            errors='ignore',
        )

    print("   Merging PM2.5 with Weather...")
    if not weather_df.empty and {'datetime', 'latitude', 'longitude'}.issubset(weather_df.columns):
        weather_df = weather_df.dropna(subset=['latitude', 'longitude', 'datetime']).sort_values('datetime')
        merged_df = vectorized_merge_nearest(
            merged_df,
            weather_df,
            time_column='datetime',
            latitude_column='latitude',
            longitude_column='longitude',
            tolerance=pd.Timedelta('6H'),
            suffix='_wx',
        )
        weather_mask = merged_df['merge_distance_deg'].fillna(np.inf) < 1.0
        for source_col, merged_col in [
            ('temperature_wx', 'temperature'),
            ('humidity_wx', 'humidity'),
            ('wind_speed_wx', 'wind_speed'),
            ('pressure_wx', 'pressure'),
        ]:
            if source_col in merged_df.columns:
                merged_df[merged_col] = merged_df[source_col].where(weather_mask, np.nan)
        merged_df = merged_df.drop(
            columns=[col for col in merged_df.columns if col.endswith('_wx')] + ['merge_distance_deg'],
            errors='ignore',
        )

    print(f"   ✅ Merged {len(merged_df)} records")
    print(f"   Coverage: {merged_df['pm2_5'].notna().sum() / len(merged_df) * 100:.1f}%")
    
    return merged_df


def save_merged_data(df):
    """Save merged dataset"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_file = MERGED_DIR / f'unified_dataset_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved merged data: {output_file}")
    print(f"   Records: {len(df)}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save summary statistics
    stats_file = MERGED_DIR / f'merge_summary_{timestamp}.json'
    
    stats = {
        'timestamp': timestamp,
        'total_records': len(df),
        'columns': list(df.columns),
        'data_coverage': {
            'pm2_5': float(df['pm2_5'].notna().sum() / len(df) * 100),
            'aod_550': float(df.get('aod_550', pd.Series()).notna().sum() / len(df) * 100) if 'aod_550' in df.columns else 0,
            'temperature': float(df.get('temperature', pd.Series()).notna().sum() / len(df) * 100) if 'temperature' in df.columns else 0,
        },
        'date_range': {
            'min': str(df['datetime'].min()) if 'datetime' in df.columns else 'N/A',
            'max': str(df['datetime'].max()) if 'datetime' in df.columns else 'N/A'
        },
        'statistics': {
            'pm2_5_mean': float(df['pm2_5'].mean()) if 'pm2_5' in df.columns else None,
            'pm2_5_std': float(df['pm2_5'].std()) if 'pm2_5' in df.columns else None,
            'rows_with_all_features': int((df[['pm2_5', 'aod_550', 'temperature']].notna().all(axis=1).sum()) if all(col in df.columns for col in ['pm2_5', 'aod_550', 'temperature']) else 0)
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   Summary: {stats_file}")
    
    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("\n🚀 Starting data merge...\n")
    
    # Load all data sources
    pm25_df = load_pm25_data()
    aod_df = load_satellite_aod_data()
    weather_df = load_weather_data()
    
    # Handle empty dataframes
    if pm25_df.empty:
        from download_pm25_data import generate_sample_pm25_data
        print("\n   Generating sample PM2.5 data...")
        pm25_df = generate_sample_pm25_data(num_records=30000)
    
    # Merge all sources
    merged_df = merge_all_data(pm25_df, aod_df, weather_df)
    
    # Save merged data
    output_file = save_merged_data(merged_df)
    
    print("\n" + "=" * 100)
    print("✅ DATA MERGING COMPLETE!")
    print("=" * 100)
    print(f"\n📊 Merged dataset ready at: {output_file}")
    print(f"   Next step: Run preprocessing script")
    
    return output_file


if __name__ == "__main__":
    main()
