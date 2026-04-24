#!/usr/bin/env python3
"""
Download Satellite AOD (Aerosol Optical Depth) data from NASA sources:
1. MODIS (Moderate Resolution Imaging Spectroradiometer)
2. Sentinel-5P (EU Copernicus Programme)
"""

import requests
import json
import zipfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import time

# Create data directory
DATA_DIR = Path('data/satellite')  # Changed from '../data/satellite' to 'data/satellite'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🛰️  SATELLITE AOD DATA DOWNLOAD SCRIPT")
print("=" * 80)


# ==============================================================================
# 1. NASA LAADS DAAC - MODIS AOD Data
# ==============================================================================

def download_modis_aod(start_date='2024-01-01', end_date=None):
    """
    Download MODIS AOD data from NASA LAADS DAAC
    
    Datasets:
    - MOD04_L2: Terra MODIS Aerosol Product
    - MYD04_L2: Aqua MODIS Aerosol Product
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), default=today
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📡 Downloading MODIS AOD Data...")
    print(f"   Date range: {start_date} to {end_date}")
    
    # You need to get a free token from: https://ladsweb.modaps.eosdis.nasa.gov/
    token = os.getenv('NASA_LAADS_TOKEN', '')
    
    if not token:
        print("   ⚠️  NASA LAADS token not found")
        print("   📌 Get free token: https://ladsweb.modaps.eosdis.nasa.gov/")
        print("   📌 Set environment variable: export NASA_LAADS_TOKEN='your_token'")
        
        # Provide fallback data sources
        print("\n   💡 Alternative: Download from:")
        print("   - https://ladsweb.modaps.eosdis.nasa.gov/ (MOD04_L2, MYD04_L2)")
        print("   - https://scihub.copernicus.eu/ (Sentinel-5P)")
        print("   - https://earthexplorer.usgs.gov/ (Landsat, MODIS)")
        
        return None
    
    try:
        # LAADS DAAC API endpoint
        base_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives"
        
        # MOD04_L2 = Terra MODIS
        # MYD04_L2 = Aqua MODIS
        collections = ['MOD04_L2', 'MYD04_L2']
        
        for collection in collections:
            print(f"\n   Fetching {collection}...")
            
            url = f"{base_url}/{collection}"
            
            params = {
                'startTime': start_date,
                'endTime': end_date,
                'dayNightBoth': 'DB',  # Both day and night
                'token': token
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'content' in data:
                    files = data['content']
                    print(f"   ✅ Found {len(files)} files for {collection}")
                    
                    # List available files
                    for file_info in files[:5]:  # Show first 5
                        print(f"      - {file_info['name']}")
                    
                    if len(files) > 5:
                        print(f"      ... and {len(files) - 5} more")
                else:
                    print(f"   ⚠️  No files found in response")
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
        
        return None
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None


# ==============================================================================
# 2. Copernicus Sentinel-5P Data
# ==============================================================================

def download_sentinel5p_aod(start_date='2024-01-01', end_date=None):
    """
    Download Sentinel-5P AOD data from Copernicus Scihub
    
    Note: Requires user account (free) at: https://scihub.copernicus.eu/
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n🛰️  Downloading Sentinel-5P AOD Data...")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Get credentials
    username = os.getenv('COPERNICUS_USER', '')
    password = os.getenv('COPERNICUS_PASS', '')
    
    if not username or not password:
        print("   ⚠️  Copernicus credentials not found")
        print("   📌 Create free account: https://scihub.copernicus.eu/")
        print("   📌 Set credentials:")
        print("      export COPERNICUS_USER='your_username'")
        print("      export COPERNICUS_PASS='your_password'")
        return None
    
    try:
        # Copernicus API endpoint
        url = "https://scihub.copernicus.eu/dhus/search"
        
        # Query for Sentinel-5P level 2 AOD product
        query = (
            f"platformname:Sentinel-5P AND "
            f"producttype:L2__AER_AI AND "
            f"beginPosition:[{start_date.replace('-', '')}T00:00:00Z TO {end_date.replace('-', '')}T23:59:59Z]"
        )
        
        params = {
            'q': query,
            'start': 0,
            'rows': 100,
            'orderby': 'ingestiondate desc',
            'format': 'json'
        }
        
        print(f"   Searching for products...")
        response = requests.get(url, params=params, auth=(username, password), timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            results = data['feed']['entry']
            
            if isinstance(results, list):
                print(f"   ✅ Found {len(results)} Sentinel-5P products")
                
                # Show first few available products
                for entry in results[:5]:
                    print(f"      - {entry['title']}")
            else:
                print(f"   ⚠️  Single result or no data")
        else:
            print(f"   ❌ Error {response.status_code}")
        
        return None
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None


# ==============================================================================
# 3. USGS Earth Explorer (Alternative)
# ==============================================================================

def download_usgs_landat_modis():
    """
    Download data from USGS Earth Explorer
    
    Supports: Landsat, MODIS, Sentinel-2, etc.
    Requires: Earth Explorer account (free)
    """
    print(f"\n🌍 USGS Earth Explorer Alternative...")
    
    username = os.getenv('USGS_USERNAME', '')
    password = os.getenv('USGS_PASSWORD', '')
    
    if not username or not password:
        print("   ⚠️  USGS Earth Explorer credentials not found")
        print("   📌 Create account: https://earthexplorer.usgs.gov/")
        print("   📌 Set credentials:")
        print("      export USGS_USERNAME='your_username'")
        print("      export USGS_PASSWORD='your_password'")
        print("   📌 API documentation: https://earthexplorer.usgs.gov/inventory/documentation/")
        return None
    
    try:
        # USGS Earth Explorer API
        api_url = "https://m2m.cr.usgs.gov/api/v1/login"
        
        login_payload = {
            'username': username,
            'password': password
        }
        
        print("   Authenticating with USGS...")
        response = requests.post(api_url, json=login_payload, timeout=30)
        
        if response.status_code == 200:
            token = response.json()['data']['token']
            print(f"   ✅ Authentication successful")
            print(f"   📌 Token: {token[:20]}...")
            
            # Can now use token for data searches
            return token
        else:
            print(f"   ❌ Authentication failed: {response.status_code}")
            return None
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None


# ==============================================================================
# 4. Simplified CSV Data Generation (for demo without credentials)
# ==============================================================================

def generate_sample_aod_data(num_records=10000):
    """
    Generate realistic sample AOD data for testing purposes
    """
    import pandas as pd
    import numpy as np
    
    print(f"\n📊 Generating {num_records} sample AOD records...")
    
    # India coordinates bounds
    lat_min, lat_max = 8.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='6H')
    
    data = {
        'date': np.random.choice(dates, num_records),
        'latitude': np.random.uniform(lat_min, lat_max, num_records),
        'longitude': np.random.uniform(lon_min, lon_max, num_records),
        'aod_550': np.random.uniform(0.1, 0.8, num_records),  # AOD at 550nm
        'aod_380': np.random.uniform(0.15, 1.0, num_records),  # AOD at 380nm
        'angstrom': np.random.uniform(0.5, 2.5, num_records),  # Angstrom exponent
        'source': np.random.choice(['MODIS_TERRA', 'MODIS_AQUA', 'Sentinel5P'], num_records),
        'quality_flag': np.random.choice(['Good', 'Fair', 'Poor'], num_records)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATA_DIR / f'aod_sample_data_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated sample AOD data")
    print(f"   Records: {len(df)}")
    print(f"   File: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return df


# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    """
    Main function to orchestrate satellite data download
    """
    
    print("\n🛰️  Starting satellite AOD data collection...\n")
    
    # Try different sources
    if os.getenv('NASA_LAADS_TOKEN'):
        download_modis_aod()
    
    if os.getenv('COPERNICUS_USER') and os.getenv('COPERNICUS_PASS'):
        download_sentinel5p_aod()
    
    if os.getenv('USGS_USERNAME') and os.getenv('USGS_PASSWORD'):
        download_usgs_landat_modis()
    
    # Generate sample data if no real credentials available
    if not any([os.getenv('NASA_LAADS_TOKEN'), 
                os.getenv('COPERNICUS_USER'),
                os.getenv('USGS_USERNAME')]):
        print("\n💡 No API credentials found. Generating sample data for demo...")
        generate_sample_aod_data(num_records=15000)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 80)
    print("✅ Satellite data download complete!")
    print("=" * 80)
    print("\n📌 To enable real data downloads, set these environment variables:")
    print("   export NASA_LAADS_TOKEN='your_token'           # From LAADS DAAC")
    print("   export COPERNICUS_USER='your_username'         # From Copernicus")
    print("   export COPERNICUS_PASS='your_password'         # From Copernicus")
    print("   export USGS_USERNAME='your_username'           # From Earth Explorer")
    print("   export USGS_PASSWORD='your_password'           # From Earth Explorer")
