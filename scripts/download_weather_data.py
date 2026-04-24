#!/usr/bin/env python3
"""
Download Weather/Meteorological Data from ERA5 Reanalysis
ECMWF - Copernicus Climate Data Store
"""

import cdsapi
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path

# Create data directory
DATA_DIR = Path('data/weather')  # Changed from '../data/weather' to 'data/weather'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🌦️  WEATHER (ERA5) DATA DOWNLOAD SCRIPT")
print("=" * 80)


# ==============================================================================
# 1. ERA5 Data Download from CDS
# ==============================================================================

def download_era5_weather_data(start_year=2023, end_year=None, variables=None):
    """
    Download ERA5 meteorological data from Copernicus Climate Data Store
    
    Required variables for PM2.5 modeling:
    - Temperature
    - Relative Humidity
    - Wind Speed (U & V components)
    - Atmospheric Pressure
    - Precipitation
    
    Args:
        start_year: Start year (default: 2023)
        end_year: End year (default: current year)
        variables: List of variables to download
    """
    
    if end_year is None:
        end_year = datetime.now().year
    
    if variables is None:
        variables = [
            '2m_temperature',
            '2m_dewpoint_temperature',
            'relative_humidity',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m',
            'surface_pressure',
            'total_precipitation',
            'instantaneous_10m_wind_speed'
        ]
    
    print(f"\n🌐 Downloading ERA5 Weather Data...")
    print(f"   Period: {start_year}-{end_year}")
    print(f"   Variables: {len(variables)} parameters")
    
    # Check if CDS API key is configured
    cds_api_url = os.getenv('CDS_API_URL', '')
    cds_api_key = os.getenv('CDS_API_KEY', '')
    
    if not cds_api_key or not cds_api_url:
        print("\n   ⚠️  CDS API credentials not configured")
        print("   📌 Setup instructions:")
        print("   1. Register at: https://cds.climate.copernicus.eu/")
        print("   2. Accept Terms & Conditions")
        print("   3. View your API credentials")
        print("   4. Create ~/.cdsapirc file with:")
        print("      url: https://cds.climate.copernicus.eu/api/v2")
        print("      key: {your-UID}:{your-API-KEY}")
        print("\n   📌 Or set environment variables:")
        print("      export CDS_API_URL='https://cds.climate.copernicus.eu/api/v2'")
        print("      export CDS_API_KEY='your-uid:your-api-key'")
        
        # Provide alternative
        print("\n   💡 Alternative: Download from:")
        print("      - https://cds.climate.copernicus.eu/ (Manual download)")
        print("      - https://www.ecmwf.int/ (Raw data)")
        
        return None
    
    try:
        # Initialize CDS client
        client = cdsapi.Client(
            url=cds_api_url,
            key=cds_api_key,
            debug=False
        )
        
        # Download for each year
        for year in range(start_year, end_year + 1):
            print(f"\n   📥 Downloading {year}...")
            
            # Request ERA5 data
            request = {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': [f'{m:02d}' for m in range(1, 13)],  # Jan-Dec
                'day': [f'{d:02d}' for d in range(1, 32)],    # All days
                'time': [f'{h:02d}:00' for h in range(0, 24)],  # Hourly data
                'format': 'netcdf'
            }
            
            # Output filename
            output_file = DATA_DIR / f'era5_weather_{year}.nc'
            
            print(f"      Variables: {len(request['variable'])} parameters")
            print(f"      Output: {output_file}")
            
            # Download (this may take a while)
            client.retrieve(
                'reanalysis-era5-complete',
                request,
                str(output_file)
            )
            
            if output_file.exists():
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"      ✅ Downloaded {year}: {size_mb:.2f} MB")
            else:
                print(f"      ❌ Failed to download {year}")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error: {str(e)}")
        if 'cdsapi' not in str(e):
            print("\n   💡 Install cdsapi first: pip install cdsapi")
        return False


# ==============================================================================
# 2. Generate Sample Weather Data (for demo)
# ==============================================================================

def generate_sample_weather_data(num_records=50000):
    """
    Generate realistic sample weather data for testing
    """
    import numpy as np
    
    print(f"\n📊 Generating {num_records} sample weather records...")
    
    # India coordinates
    lat_min, lat_max = 8.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    
    # Date range
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='3H')
    
    # Sample realistic values for India
    data = {
        'date': np.random.choice(dates, num_records),
        'latitude': np.random.uniform(lat_min, lat_max, num_records),
        'longitude': np.random.uniform(lon_min, lon_max, num_records),
        'temperature_2m': np.random.uniform(5, 45, num_records),  # 5-45°C
        'dew_point_2m': np.random.uniform(0, 30, num_records),    # 0-30°C
        'relative_humidity': np.random.uniform(20, 95, num_records),  # 20-95%
        'u_wind_10m': np.random.uniform(-10, 10, num_records),    # m/s
        'v_wind_10m': np.random.uniform(-10, 10, num_records),    # m/s
        'wind_speed_10m': np.random.uniform(0, 15, num_records),  # m/s
        'surface_pressure': np.random.uniform(95000, 105000, num_records),  # Pa
        'total_precipitation': np.random.gamma(2, 2, num_records), # mm
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived variables
    df['wind_direction'] = np.arctan2(df['v_wind_10m'], df['u_wind_10m']) * 180 / np.pi
    df['wind_direction'] = (df['wind_direction'] + 360) % 360
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATA_DIR / f'weather_sample_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated sample weather data")
    print(f"   Records: {len(df)}")
    print(f"   File: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return df


# ==============================================================================
# 3. Download NOAA Weather Data (Alternative, free)
# ==============================================================================

def download_noaa_weather_data():
    """
    Download freely available NOAA weather data
    """
    import requests
    
    print(f"\n🌦️  Checking NOAA Weather Data...")
    
    try:
        # NOAA GFS (Global Forecast System) data
        # Available from: https://www.ncei.noaa.gov/
        
        # Example: Download historical weather from NOAA
        print("   📌 NOAA data available at:")
        print("      https://www.ncei.noaa.gov/products/global-hourly-data")
        print("      https://www.ncei.noaa.gov/products/global-data-assimilation-system-gdas")
        
        # Can also use NOAA's REST API for specific stations
        years = [2023, 2024, 2025]
        
        for year in years:
            print(f"\n   Checking NOAA data for {year}...")
            
            # ISD (Integrated Surface Database) data URL
            url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/"
            
            response = requests.head(url, timeout=10, allow_redirects=True)
            
            if response.status_code == 200:
                print(f"      ✅ Data available for {year}")
                print(f"      📌 Download from: {url}")
            else:
                print(f"      ⚠️  Status: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    """
    Main function to orchestrate weather data download
    """
    
    print("\n🌦️  Starting weather data collection...\n")
    
    # Try to download real ERA5 data if credentials available
    if os.getenv('CDS_API_KEY'):
        print("Found CDS API credentials. Attempting ERA5 download...")
        success = download_era5_weather_data(start_year=2024, end_year=2025)
        
        if success:
            return
    
    # Try NOAA data
    download_noaa_weather_data()
    
    # Generate sample data for testing
    print("\n💡 Generating sample weather data for demo...")
    generate_sample_weather_data(num_records=50000)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 80)
    print("✅ Weather data collection complete!")
    print("=" * 80)
    print("\n📌 To enable ERA5 data downloads:")
    print("   1. Register: https://cds.climate.copernicus.eu/")
    print("   2. Create ~/.cdsapirc with your credentials")
    print("   3. Install: pip install cdsapi")
    print("\n📌 Or use NOAA data (free, no authentication needed)")
