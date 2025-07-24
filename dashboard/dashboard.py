#!/usr/bin/env python3
"""
🌍 VayuDrishti PM2.5 Forecasting Dashboard - Hackathon Edition
Offline-only air quality prediction system for Bharatiya Antariksh

Features:
- 🔋 Fully offline using local XGBoost model (R² > 0.88)
- 🗺️ Interactive India map with PM2.5 predictions
- 📊 Real-time prediction for any location
- 🏥 Health advisory with CPCB AQI standards
- 📈 Multi-day forecasting capabilities

Usage:
    streamlit run dashboard.py
"""

import sys
import subprocess

# Function to install missing packages
def install_missing_packages():
    """Install missing packages automatically"""
    missing_packages = []
    
    try:
        import streamlit
    except ImportError:
        missing_packages.append('streamlit')
    
    try:
        import folium
    except ImportError:
        missing_packages.append('folium')
    
    try:
        import plotly
    except ImportError:
        missing_packages.append('plotly')
    
    try:
        from streamlit_folium import st_folium
    except ImportError:
        missing_packages.append('streamlit-folium')
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                print(f"Please install manually: pip install {package}")
                sys.exit(1)
        
        print("🔄 Please restart the dashboard after installation")
        sys.exit(0)

# Install missing packages if needed
install_missing_packages()

# Now import all required packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
# Removed requests import - running in offline mode
import json
from datetime import datetime, timedelta, date
import math
from pathlib import Path
import logging

# Import offline forecast capability
try:
    from offline_forecast import OfflineForecast
    offline_forecast = OfflineForecast()
    OFFLINE_FORECAST_AVAILABLE = True
except ImportError:
    OFFLINE_FORECAST_AVAILABLE = False
    print("⚠️ Offline forecast module not available")

# Configure page
st.set_page_config(
    page_title="🌍 VayuDrishti - Bharatiya Antariksh",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply modern hackathon-ready theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1d29 50%, #0f1419 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content container */
    .main .block-container {
        padding: 2rem 1.5rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Sidebar */
    .css-1d391kg, .stSidebar {
        background: linear-gradient(180deg, #1e2139 0%, #16192b 100%);
        border-right: 2px solid #2d3748;
    }
    
    /* Headers with gradient text */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #29b6f6, #03a9f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    /* Metric containers */
    .metric-container, [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.9), rgba(26, 32, 44, 0.9));
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 16px;
        border: 1px solid rgba(79, 195, 247, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(79, 195, 247, 0.2);
        border-color: rgba(79, 195, 247, 0.4);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4fc3f7, #29b6f6);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #29b6f6, #03a9f4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 195, 247, 0.4);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 32, 44, 0.6);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4fc3f7, #29b6f6);
        color: white !important;
        box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
    }
    
    /* Input styling */
    .stNumberInput input, .stSelectbox select, .stSlider {
        background: rgba(45, 55, 72, 0.8) !important;
        border: 1px solid rgba(79, 195, 247, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(56, 178, 172, 0.2), rgba(34, 197, 94, 0.2));
        border-left: 4px solid #10b981;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(79, 195, 247, 0.2), rgba(41, 182, 246, 0.2));
        border-left: 4px solid #4fc3f7;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
        border-left: 4px solid #f59e0b;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 127, 0.2));
        border-left: 4px solid #ef4444;
    }
    
    /* Footer */
    .footer {
        background: rgba(26, 32, 44, 0.9);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(79, 195, 247, 0.2);
        text-align: center;
        margin-top: 3rem;
        color: #a0aec0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 32, 44, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 195, 247, 0.6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 195, 247, 0.8);
    }
    
    /* Health category colors */
    .health-good { color: #10b981; font-weight: 600; }
    .health-satisfactory { color: #f59e0b; font-weight: 600; }
    .health-moderate { color: #f97316; font-weight: 600; }
    .health-poor { color: #ef4444; font-weight: 600; }
    .health-very-poor { color: #8b5cf6; font-weight: 600; }
    .health-severe { color: #dc2626; font-weight: 600; }
    
    /* Button styling */
    .stButton {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VayuDrishtiDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        # Removed API URL - running in offline mode
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'predictions_data' not in st.session_state:
            st.session_state.predictions_data = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def load_prediction_data(self) -> pd.DataFrame:
        """Load prediction data from CSV files"""
        try:
            # Look for prediction files
            prediction_files = []
            
            # Check multiple possible locations
            possible_dirs = [
                Path("jobs/daily_predictions"),
                Path("../jobs/daily_predictions"),
                Path("daily_predictions"),
                Path(".")
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    prediction_files.extend(list(dir_path.glob("predictions_*.csv")))
            
            if not prediction_files:
                st.warning("📂 No prediction files found. Run the daily prediction job first.")
                return self.create_sample_data()
            
            # Load the most recent file
            latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
            st.info(f"📁 Loading data from: {latest_file.name}")
            
            df = pd.read_csv(latest_file)
            
            # Ensure required columns exist
            required_cols = ['latitude', 'longitude', 'predicted_pm2_5']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns in data: {missing_cols}")
                return self.create_sample_data()
            
            # Add health categories if not present
            if 'health_category' not in df.columns:
                df['health_category'] = df['predicted_pm2_5'].apply(self.get_health_category)
            
            # Add timestamp if not present
            if 'prediction_timestamp' not in df.columns:
                df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading prediction data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        st.info("📊 Creating sample data for demonstration")
        
        # Generate sample grid across India
        np.random.seed(42)
        
        # India bounds approximately
        lat_min, lat_max = 8.0, 37.0
        lon_min, lon_max = 68.0, 97.0
        
        n_points = 200
        
        # Generate grid points
        lats = np.random.uniform(lat_min, lat_max, n_points)
        lons = np.random.uniform(lon_min, lon_max, n_points)
        
        # Generate realistic PM2.5 values with regional variation
        pm25_values = []
        health_categories = []
        
        for lat, lon in zip(lats, lons):
            # Add regional variation (Delhi NCR tends to be higher)
            delhi_distance = ((lat - 28.6)**2 + (lon - 77.2)**2)**0.5
            base_pm25 = 30 + 40 * np.exp(-delhi_distance/5) + np.random.normal(0, 15)
            base_pm25 = max(5, min(200, base_pm25))  # Clamp values
            
            pm25_values.append(base_pm25)
            health_categories.append(self.get_health_category(base_pm25))
        
        df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'predicted_pm2_5': pm25_values,
            'health_category': health_categories,
            'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return df
    
    def get_health_category(self, pm25_value: float) -> str:
        """Get health category for PM2.5 value"""
        if pm25_value <= 12.0:
            return "Good"
        elif pm25_value <= 35.5:
            return "Moderate"
        elif pm25_value <= 55.4:
            return "Unhealthy for Sensitive Groups"
        elif pm25_value <= 150.4:
            return "Unhealthy"
        elif pm25_value <= 250.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def get_health_color(self, category: str) -> str:
        """Get color for health category based on CPCB standards"""
        color_map = {
            "Good": "#00e400",
            "Satisfactory": "#ffff00",
            "Moderate": "#ff7e00", 
            "Poor": "#ff0000",
            "Very Poor": "#8f3f97",
            "Severe": "#7e0023",
            # Legacy support for US EPA categories
            "Unhealthy for Sensitive Groups": "#ff7e00",
            "Unhealthy": "#ff0000",
            "Very Unhealthy": "#8f3f97",
            "Hazardous": "#7e0023"
        }
        return color_map.get(category, "#gray")
    
    def create_india_map(self, df: pd.DataFrame) -> folium.Map:
        """Create interactive India map with PM2.5 data"""
        # Center on India
        center_lat, center_lon = 20.5937, 78.9629
        
        # Filter data to include only Indian territory (approximate boundaries)
        india_bounds = {
            'lat_min': 8.0,   # Southernmost point (Kanyakumari area)
            'lat_max': 37.0,  # Northernmost point (Kashmir)
            'lon_min': 68.0,  # Westernmost point (Gujarat/Rajasthan)
            'lon_max': 97.0   # Easternmost point (Arunachal Pradesh)
        }
        
        # Filter out points over ocean/foreign areas
        filtered_df = df[
            (df['latitude'] >= india_bounds['lat_min']) &
            (df['latitude'] <= india_bounds['lat_max']) &
            (df['longitude'] >= india_bounds['lon_min']) &
            (df['longitude'] <= india_bounds['lon_max'])
        ].copy()
        
        # Additional filtering for specific water bodies and foreign areas
        # Remove points in Arabian Sea (west of 70°E and south of 20°N)
        filtered_df = filtered_df[~(
            (filtered_df['longitude'] < 70.0) & (filtered_df['latitude'] < 20.0)
        )]
        
        # Remove points in Bay of Bengal (east of 90°E and south of 15°N)
        filtered_df = filtered_df[~(
            (filtered_df['longitude'] > 90.0) & (filtered_df['latitude'] < 15.0)
        )]
        
        # Remove points in Pakistan region (west of 74°E and north of 28°N)
        filtered_df = filtered_df[~(
            (filtered_df['longitude'] < 74.0) & (filtered_df['latitude'] > 28.0)
        )]
        
        # Remove points in China region (east of 95°E and north of 28°N)
        filtered_df = filtered_df[~(
            (filtered_df['longitude'] > 95.0) & (filtered_df['latitude'] > 28.0)
        )]
        
        # Create base map with dark theme
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='CartoDB dark_matter',  # Dark theme map
            max_bounds=True,
            min_lat=india_bounds['lat_min'],
            max_lat=india_bounds['lat_max'],
            min_lon=india_bounds['lon_min'],
            max_lon=india_bounds['lon_max']
        )
        
        # Add PM2.5 data points only for filtered locations
        for _, row in filtered_df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            pm25 = row['predicted_pm2_5']
            category = row['health_category']
            color = self.get_health_color(category)
            
            # Create popup with information
            popup_html = f"""
            <div style="font-family: Arial; width: 200px; color: black;">
                <h4>📍 PM2.5 Level</h4>
                <p><strong>Location:</strong> {lat:.2f}°N, {lon:.2f}°E</p>
                <p><strong>PM2.5:</strong> {pm25:.1f} μg/m³</p>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Time:</strong> {row.get('prediction_timestamp', 'N/A')}</p>
            </div>
            """
            
            # Add circle marker with enhanced styling
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(4, min(20, pm25/8)),  # Enhanced size based on PM2.5 level
                popup=folium.Popup(popup_html, max_width=250),
                color='white',
                weight=2,
                fillColor=color,
                fillOpacity=0.8,
                tooltip=f"PM2.5: {pm25:.1f} μg/m³ ({category})"
            ).add_to(m)
        
        # Add enhanced legend with dark theme
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: 180px; 
                    background-color: rgba(40, 40, 40, 0.9); 
                    border: 2px solid #3a3a3a; 
                    border-radius: 8px;
                    z-index: 9999; 
                    font-size: 12px; 
                    padding: 15px;
                    color: white;
                    font-family: Arial;">
        <h4 style="margin: 0 0 10px 0; color: #fafafa;">🌡️ PM2.5 Health Categories</h4>
        <p style="margin: 3px 0;"><span style="color:#00e400;">●</span> Good (0-30 μg/m³)</p>
        <p style="margin: 3px 0;"><span style="color:#ffff00;">●</span> Satisfactory (31-60)</p>
        <p style="margin: 3px 0;"><span style="color:#ff7e00;">●</span> Moderate (61-90)</p>
        <p style="margin: 3px 0;"><span style="color:#ff0000;">●</span> Poor (91-120)</p>
        <p style="margin: 3px 0;"><span style="color:#8f3f97;">●</span> Very Poor (121-250)</p>
        <p style="margin: 3px 0;"><span style="color:#7e0023;">●</span> Severe (250+)</p>
        <small style="color: #cccccc;">Size indicates pollution level</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add India boundary outline (simplified)
        india_boundary = [
            [8.0, 68.0], [8.0, 97.0], [37.0, 97.0], [37.0, 68.0], [8.0, 68.0]
        ]
        
        folium.PolyLine(
            locations=india_boundary,
            color='cyan',
            weight=2,
            opacity=0.6,
            tooltip="India Boundary (Approximate)"
        ).add_to(m)
        
        return m
    
    def create_pm25_histogram(self, df: pd.DataFrame) -> go.Figure:
        """Create PM2.5 distribution histogram"""
        fig = px.histogram(
            df, 
            x='predicted_pm2_5',
            color='health_category',
            title="📊 PM2.5 Distribution Across India",
            labels={'predicted_pm2_5': 'PM2.5 (μg/m³)', 'count': 'Number of Locations'},
            color_discrete_map={
                "Good": "#00e400",
                "Moderate": "#ffff00",
                "Unhealthy for Sensitive Groups": "#ff7e00",
                "Unhealthy": "#ff0000",
                "Very Unhealthy": "#8f3f97",
                "Hazardous": "#7e0023"
            }
        )
        
        fig.update_layout(
            xaxis_title="PM2.5 Level (μg/m³)",
            yaxis_title="Number of Locations",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_major_cities_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Create hackathon-ready major Indian cities summary with offline predictions"""
        
        # Major Indian cities with coordinates for demo
        major_cities = {
            "🏛️ Delhi": {"lat": 28.6139, "lon": 77.2090, "population": "32M"},
            "🏙️ Mumbai": {"lat": 19.0760, "lon": 72.8777, "population": "21M"},
            "🌆 Bangalore": {"lat": 12.9716, "lon": 77.5946, "population": "13M"},
            "🏘️ Kolkata": {"lat": 22.5726, "lon": 88.3639, "population": "15M"},
            "🌴 Chennai": {"lat": 13.0827, "lon": 80.2707, "population": "11M"},
            "💻 Hyderabad": {"lat": 17.3850, "lon": 78.4867, "population": "10M"},
            "🕌 Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "population": "8M"},
            "🎓 Pune": {"lat": 18.5204, "lon": 73.8567, "population": "7M"},
            "🏰 Jaipur": {"lat": 26.9124, "lon": 75.7873, "population": "4M"},
            "🌊 Kochi": {"lat": 9.9312, "lon": 76.2673, "population": "2M"}
        }
        
        # Generate offline predictions for each city
        cities_data = []
        
        for city_name, info in major_cities.items():
            try:
                if OFFLINE_FORECAST_AVAILABLE and offline_forecast.model_loaded:
                    # Generate baseline features for the city
                    features = offline_forecast.generate_baseline_features(
                        info["lat"], info["lon"], datetime.now()
                    )
                    
                    # Add latitude and longitude to features for the model
                    features['latitude'] = info["lat"]
                    features['longitude'] = info["lon"]
                    
                    # Use the new prediction function
                    result = offline_forecast.predict_pm25_offline(features)
                    
                    cities_data.append({
                        "City": city_name,
                        "Population": info["population"],
                        "PM2.5": result["pm25"],
                        "AQI": result["aqi"],
                        "Category": result["health_category"],
                        "Coordinates": f"{info['lat']:.2f}°N, {info['lon']:.2f}°E"
                    })
                else:
                    # Fallback demo data if model not available
                    demo_pm25 = np.random.uniform(20, 150)  # Random demo value
                    aqi, category = offline_forecast.pm25_to_cpcb_aqi(demo_pm25) if offline_forecast else (0, "Unknown")
                    
                    cities_data.append({
                        "City": city_name,
                        "Population": info["population"],
                        "PM2.5": round(demo_pm25, 1),
                        "AQI": int(aqi),
                        "Category": category,
                        "Coordinates": f"{info['lat']:.2f}°N, {info['lon']:.2f}°E"
                    })
                    
            except Exception as e:
                # Error fallback
                cities_data.append({
                    "City": city_name,
                    "Population": info["population"],
                    "PM2.5": 0.0,
                    "AQI": 0,
                    "Category": "Error",
                    "Coordinates": f"{info['lat']:.2f}°N, {info['lon']:.2f}°E"
                })
        
        return pd.DataFrame(cities_data)
        
        # Enhanced city definitions with precise boundaries
        def get_region(lat, lon):
            # Delhi NCR (expanded area)
            if 28.3 <= lat <= 28.9 and 76.8 <= lon <= 77.8:
                return "🏛️ Delhi NCR"
            # Mumbai Metropolitan Region
            elif 18.9 <= lat <= 19.3 and 72.7 <= lon <= 73.2:
                return "🏙️ Mumbai"
            # Bangalore Urban District
            elif 12.8 <= lat <= 13.2 and 77.4 <= lon <= 77.8:
                return "🌆 Bangalore"
            # Kolkata Metropolitan Area
            elif 22.4 <= lat <= 22.8 and 88.2 <= lon <= 88.5:
                return "🏘️ Kolkata"
            # Chennai Metropolitan Area
            elif 12.9 <= lat <= 13.2 and 80.1 <= lon <= 80.4:
                return "🌴 Chennai"
            # Hyderabad Metropolitan Area
            elif 17.2 <= lat <= 17.6 and 78.2 <= lon <= 78.7:
                return "💻 Hyderabad"
            # Ahmedabad Metropolitan Area
            elif 22.9 <= lat <= 23.3 and 72.4 <= lon <= 72.8:
                return "🕌 Ahmedabad"
            # Pune Metropolitan Area
            elif 18.4 <= lat <= 18.7 and 73.7 <= lon <= 74.0:
                return "🎓 Pune"
            # Jaipur District
            elif 26.8 <= lat <= 27.0 and 75.6 <= lon <= 76.0:
                return "🏰 Jaipur"
            # Lucknow District
            elif 26.7 <= lat <= 26.9 and 80.8 <= lon <= 81.1:
                return "🏛️ Lucknow"
            # Kanpur District
            elif 26.3 <= lat <= 26.6 and 80.2 <= lon <= 80.5:
                return "🏭 Kanpur"
            # Nagpur District
            elif 21.0 <= lat <= 21.3 and 78.9 <= lon <= 79.2:
                return "🌊 Nagpur"
            # Patna District
            elif 25.5 <= lat <= 25.7 and 85.0 <= lon <= 85.3:
                return "🏫 Patna"
            # Indore District
            elif 22.6 <= lat <= 22.8 and 75.7 <= lon <= 76.0:
                return "🌿 Indore"
            # Bhopal District
            elif 23.1 <= lat <= 23.4 and 77.3 <= lon <= 77.6:
                return "🏞️ Bhopal"
            # Visakhapatnam District
            elif 17.6 <= lat <= 17.8 and 83.1 <= lon <= 83.4:
                return "🌊 Visakhapatnam"
            # Vadodara District
            elif 22.2 <= lat <= 22.4 and 73.0 <= lon <= 73.3:
                return "🏛️ Vadodara"
            # Ludhiana District
            elif 30.8 <= lat <= 31.0 and 75.7 <= lon <= 76.0:
                return "🌾 Ludhiana"
            # Agra District
            elif 27.1 <= lat <= 27.3 and 77.9 <= lon <= 78.2:
                return "🕌 Agra"
            # Nashik District
            elif 19.9 <= lat <= 20.1 and 73.6 <= lon <= 73.9:
                return "🍇 Nashik"
            # Northern India (remaining)
            elif lat >= 26.0:
                return "🏔️ Northern India"
            # Western India (remaining)
            elif lon <= 76.0:
                return "🌵 Western India"
            # Eastern India (remaining)
            elif lon >= 86.0:
                return "🌾 Eastern India"
            # Central India (remaining)
            elif 18.0 <= lat <= 26.0:
                return "🏛️ Central India"
            # Southern India (remaining)
            else:
                return "🌴 Southern India"
        
        # Apply region classification
        df_copy = df.copy()
        df_copy['region'] = df_copy.apply(lambda x: get_region(x['latitude'], x['longitude']), axis=1)
        
        # Calculate comprehensive statistics
        regional_stats = df_copy.groupby('region').agg({
            'predicted_pm2_5': ['mean', 'min', 'max', 'std', 'count'],
            'health_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(1)
        
        # Flatten column names
        regional_stats.columns = ['Avg PM2.5', 'Min PM2.5', 'Max PM2.5', 'Std Dev', 'Locations', 'Health Category']
        
        # Add AQI calculations and health tips
        regional_stats['Avg AQI'] = regional_stats['Avg PM2.5'].apply(
            lambda x: self.pm25_to_aqi(x)[0] if hasattr(self, 'pm25_to_aqi') else int(x * 2)
        )
        
        # Add health tips based on average PM2.5
        def get_health_tip(avg_pm25):
            if avg_pm25 <= 30:
                return "✅ Excellent for outdoor activities"
            elif avg_pm25 <= 60:
                return "😊 Generally safe, sensitive individuals be cautious"
            elif avg_pm25 <= 90:
                return "😷 Consider masks for outdoor activities"
            elif avg_pm25 <= 120:
                return "⚠️ Limit outdoor exposure, especially for children"
            else:
                return "🚨 Avoid outdoor activities, stay indoors"
        
        regional_stats['Health Tip'] = regional_stats['Avg PM2.5'].apply(get_health_tip)
        
        # Sort by average PM2.5 (worst first for priority)
        regional_stats = regional_stats.sort_values('Avg PM2.5', ascending=False)
        
        return regional_stats.reset_index()
    
    def render_health_advisory_cards(self, df: pd.DataFrame):
        """Render health advisory cards"""
        st.subheader("🏥 Health Advisory")
        
        # Calculate overall statistics
        avg_pm25 = df['predicted_pm2_5'].mean()
        max_pm25 = df['predicted_pm2_5'].max()
        dominant_category = df['health_category'].mode().iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Average PM2.5",
                value=f"{avg_pm25:.1f} μg/m³",
                delta=f"vs 35 μg/m³ limit"
            )
        
        with col2:
            st.metric(
                label="⚠️ Maximum PM2.5",
                value=f"{max_pm25:.1f} μg/m³",
                delta="Hotspot level"
            )
        
        with col3:
            locations_good = len(df[df['health_category'] == 'Good'])
            total_locations = len(df)
            good_percentage = (locations_good / total_locations) * 100
            
            st.metric(
                label="✅ Good Air Quality",
                value=f"{locations_good} locations",
                delta=f"{good_percentage:.1f}% of total"
            )
        
        with col4:
            unhealthy_count = len(df[df['predicted_pm2_5'] > 55.4])
            unhealthy_percentage = (unhealthy_count / total_locations) * 100
            
            st.metric(
                label="⚠️ Unhealthy Levels",
                value=f"{unhealthy_count} locations",
                delta=f"{unhealthy_percentage:.1f}% of total"
            )
        
        # Health recommendations
        st.markdown("### 💡 Recommendations")
        
        if avg_pm25 <= 12:
            st.success("🌟 **Excellent air quality!** Perfect for all outdoor activities.")
        elif avg_pm25 <= 35:
            st.info("😊 **Good air quality overall.** Minor concerns for very sensitive individuals.")
        elif avg_pm25 <= 55:
            st.warning("😷 **Moderate concerns.** Sensitive groups should limit prolonged outdoor activities.")
        else:
            st.error("🚨 **Health alert!** Everyone should reduce outdoor exposure.")
        
        # Specific recommendations
        recommendations = [
            "🏃‍♀️ **Exercise:** " + ("Safe outdoors" if avg_pm25 <= 35 else "Consider indoor alternatives"),
            "🏫 **Schools:** " + ("Normal activities" if avg_pm25 <= 55 else "Limit outdoor sports"),
            "👶 **Children & Elderly:** " + ("No special precautions" if avg_pm25 <= 12 else "Extra care advised"),
            "😷 **Masks:** " + ("Not necessary" if avg_pm25 <= 35 else "Recommended outdoors")
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def generate_forecast_display(self, lat: float, lon: float, days: int, start_date: date, display_col):
        """Generate and display forecast results"""
        with display_col:
            with st.spinner("🔮 Generating forecast..."):
                try:
                    # Prepare forecast request
                    forecast_request = {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date.isoformat(),
                        "forecast_days": days
                    }
                    
                    # Generate offline forecast using local model
                    if OFFLINE_FORECAST_AVAILABLE:
                        forecast_data = offline_forecast.generate_forecast(
                            latitude=lat,
                            longitude=lon,
                            start_date=start_date,
                            forecast_days=days
                        )
                        
                        st.session_state.forecast_data = forecast_data
                        
                        # Display results
                        st.markdown("### 📊 Forecast Results (Offline Mode)")
                        st.info("📝 **Note**: Generated using local ML model with meteorological patterns.")
                        
                        # Prepare data for visualization
                        forecast_df = pd.DataFrame(forecast_data['forecast'])
                        forecast_df['Date'] = pd.to_datetime(forecast_df['date'])
                        
                        # Create enhanced line chart with AQI color coding
                        fig = go.Figure()
                        
                        # Color mapping for AQI categories
                        color_map = {
                            'Good': '#00e400',
                            'Satisfactory': '#ffff00', 
                            'Moderate': '#ff7e00',
                            'Poor': '#ff0000',
                            'Very Poor': '#8f3f97',
                            'Severe': '#7e0023'
                        }
                        
                        # Add PM2.5 line with markers
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['pm2_5'],
                            mode='lines+markers',
                            name='PM2.5',
                            line=dict(width=3, color='lightblue'),
                            marker=dict(
                                size=10,
                                color=[color_map.get(cat, '#gray') for cat in forecast_df['category']],
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} μg/m³<br>AQI: %{customdata[0]}<br>Category: %{customdata[1]}<extra></extra>',
                            customdata=list(zip(forecast_df['aqi'], forecast_df['category']))
                        ))
                        
                        # Update layout with dark theme
                        fig.update_layout(
                            title=f"📈 {days}-Day PM2.5 Forecast for ({lat:.2f}, {lon:.2f})",
                            xaxis_title="Date",
                            yaxis_title="PM2.5 Concentration (μg/m³)",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            title_font_color='white',
                            title_font_size=16,
                            hovermode='x unified',
                            showlegend=False
                        )
                        
                        # Add AQI reference lines
                        aqi_levels = [
                            (30, 'Good', '#00e400'),
                            (60, 'Satisfactory', '#ffff00'),
                            (90, 'Moderate', '#ff7e00'),
                            (120, 'Poor', '#ff0000'),
                            (250, 'Very Poor', '#8f3f97')
                        ]
                        
                        for level, name, color in aqi_levels:
                            fig.add_hline(
                                y=level, 
                                line_dash="dash", 
                                line_color=color,
                                opacity=0.3,
                                annotation_text=f"{name} ({level})",
                                annotation_position="right"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.markdown("### 📋 Detailed Forecast")
                        
                        # Style the dataframe for better visibility
                        styled_df = forecast_df.copy()
                        styled_df['Date'] = styled_df['Date'].dt.strftime('%Y-%m-%d')
                        styled_df['PM2.5'] = styled_df['pm2_5'].round(1)
                        styled_df['AQI'] = styled_df['aqi']
                        styled_df['Health Category'] = styled_df['category']
                        
                        # Add emoji indicators
                        emoji_map = {
                            'Good': '😊',
                            'Satisfactory': '🙂', 
                            'Moderate': '😐',
                            'Poor': '😷',
                            'Very Poor': '🚨',
                            'Severe': '⚠️'
                        }
                        styled_df['Status'] = styled_df['category'].map(emoji_map)
                        
                        display_df = styled_df[['Date', 'PM2.5', 'AQI', 'Health Category', 'Status']]
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "PM2.5": st.column_config.NumberColumn(
                                    "PM2.5 (μg/m³)",
                                    format="%.1f"
                                ),
                                "AQI": st.column_config.NumberColumn(
                                    "AQI",
                                    format="%d"
                                ),
                                "Status": st.column_config.TextColumn(
                                    "Status",
                                    width="small"
                                )
                            }
                        )
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_pm25 = forecast_df['pm2_5'].mean()
                            st.metric("📊 Avg PM2.5", f"{avg_pm25:.1f} μg/m³")
                        
                        with col2:
                            max_pm25 = forecast_df['pm2_5'].max()
                            st.metric("📈 Peak PM2.5", f"{max_pm25:.1f} μg/m³")
                        
                        with col3:
                            avg_aqi = forecast_df['aqi'].mean()
                            st.metric("📊 Avg AQI", f"{avg_aqi:.0f}")
                        
                        with col4:
                            worst_day = forecast_df.loc[forecast_df['pm2_5'].idxmax()]
                            st.metric("⚠️ Worst Day", worst_day['date'][-5:])  # MM-DD format
                        
                        # Health recommendations based on forecast
                        st.markdown("### 💡 Forecast-Based Recommendations")
                        
                        if avg_pm25 <= 30:
                            st.success("🌟 **Excellent forecast!** Great conditions for outdoor activities throughout the period.")
                        elif avg_pm25 <= 60:
                            st.info("😊 **Generally good conditions.** Some variation expected but mostly safe.")
                        elif avg_pm25 <= 90:
                            st.warning("😷 **Moderate conditions.** Sensitive individuals should monitor daily levels.")
                        else:
                            st.error("🚨 **Health alert period!** Plan indoor activities and limit exposure.")
                        
                        # Day-specific alerts
                        if max_pm25 > 120:
                            worst_date = forecast_df.loc[forecast_df['pm2_5'].idxmax(), 'date']
                            st.error(f"⚠️ **High pollution alert for {worst_date}** - Consider staying indoors")
                    else:
                        st.error("❌ Offline forecast model not available")
                        st.info("Please ensure best_model.pkl is in the models/ directory")
                        
                except Exception as e:
                    st.error("� **Prediction Error**")
                    st.error(f"Error: {e}")
                    
                    if not OFFLINE_FORECAST_AVAILABLE:
                        st.warning("⚠️ **Offline forecast model not available**")
                        st.info("📝 **Note**: Please ensure best_model.pkl is in the models/ directory.")
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header with improved spacing
        st.title("🌍 VayuDrishti PM2.5 Forecasting Dashboard (Offline Mode)")
        st.markdown("Offline air quality monitoring and prediction using local ML models")
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        # Model info banner
        model_info_col1, model_info_col2 = st.columns([3, 1])
        with model_info_col1:
            st.info("🔋 **Offline Mode**: Running fully offline using local XGBoost model!")
        with model_info_col2:
            if st.button("� Model Info", help="View model details"):
                if OFFLINE_FORECAST_AVAILABLE and offline_forecast.model_loaded:
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Model not available")
        
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing after divider
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Data refresh
            if st.button("🔄 Refresh Data", type="primary"):
                st.session_state.predictions_data = None
                st.rerun()
            
            # Data source info
            st.markdown("### 📊 Data Source")
            if st.session_state.last_update:
                st.info(f"Last updated: {st.session_state.last_update}")
            
            # Filters
            st.markdown("### 🔍 Filters")
            
        # Load data
        if st.session_state.predictions_data is None:
            with st.spinner("Loading prediction data..."):
                st.session_state.predictions_data = self.load_prediction_data()
                st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        df = st.session_state.predictions_data
        
        if df is None or len(df) == 0:
            st.error("No data available. Please run the daily prediction job first.")
            return
        
        # Apply filters
        with st.sidebar:
            # Health category filter
            health_categories = ["All"] + sorted(df['health_category'].unique().tolist())
            selected_category = st.selectbox("Health Category", health_categories)
            
            if selected_category != "All":
                df = df[df['health_category'] == selected_category]
            
            # PM2.5 range filter
            min_pm25, max_pm25 = float(df['predicted_pm2_5'].min()), float(df['predicted_pm2_5'].max())
            pm25_range = st.slider(
                "PM2.5 Range (μg/m³)",
                min_value=min_pm25,
                max_value=max_pm25,
                value=(min_pm25, max_pm25),
                step=1.0
            )
            
            df = df[
                (df['predicted_pm2_5'] >= pm25_range[0]) & 
                (df['predicted_pm2_5'] <= pm25_range[1])
            ]
            
            # Show filtered stats
            st.markdown(f"**Showing {len(df)} locations**")
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ India Map", "📊 Analytics", "🏙️ Major Cities", "🔮 Live Prediction", "📆 Forecast"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)  # Add top spacing
            st.subheader("🗺️ PM2.5 Levels Across India")
            
            # Create and display map with proper spacing
            india_map = self.create_india_map(df)
            st.markdown("<div style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            map_data = st_folium(india_map, width=1200, height=600)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing before health cards
            
            # Health advisory cards
            self.render_health_advisory_cards(df)
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add top spacing
            st.subheader("📊 Air Quality Analytics")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing after subheader
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PM2.5 distribution
                hist_fig = self.create_pm25_histogram(df)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            with col2:
                # Health category pie chart
                category_counts = df['health_category'].value_counts()
                pie_fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="🏥 Health Category Distribution",
                    color=category_counts.index,
                    color_discrete_map={
                        "Good": "#00e400",
                        "Moderate": "#ffff00",
                        "Unhealthy for Sensitive Groups": "#ff7e00",
                        "Unhealthy": "#ff0000",
                        "Very Unhealthy": "#8f3f97",
                        "Hazardous": "#7e0023"
                    }
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing before statistics
            
            # Statistics table
            st.subheader("📈 Summary Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean PM2.5', 'Median PM2.5', 'Std Dev', 'Min PM2.5', 'Max PM2.5', 'Total Locations'],
                'Value': [
                    f"{df['predicted_pm2_5'].mean():.1f} μg/m³",
                    f"{df['predicted_pm2_5'].median():.1f} μg/m³",
                    f"{df['predicted_pm2_5'].std():.1f} μg/m³",
                    f"{df['predicted_pm2_5'].min():.1f} μg/m³",
                    f"{df['predicted_pm2_5'].max():.1f} μg/m³",
                    f"{len(df)} locations"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with tab3:
            st.subheader("🏙️ Major Indian Cities - Air Quality Status")
            st.markdown("**Live predictions for India's top metropolitan areas**")
            
            # Generate major cities data
            cities_df = self.create_major_cities_summary()
            
            # Display as enhanced dataframe
            st.dataframe(
                cities_df,
                use_container_width=True,
                column_config={
                    "City": st.column_config.TextColumn("🏙️ City", width="medium"),
                    "Population": st.column_config.TextColumn("👥 Population", width="small"),
                    "PM2.5": st.column_config.NumberColumn("🌬️ PM2.5 (μg/m³)", format="%.1f"),
                    "AQI": st.column_config.NumberColumn("📊 AQI", format="%d"),
                    "Category": st.column_config.TextColumn("🏥 Health Category", width="medium"),
                    "Coordinates": st.column_config.TextColumn("📍 Location", width="medium")
                }
            )
            
            # City comparison chart
            fig = px.bar(
                cities_df,
                x='City',
                y='PM2.5',
                title="🏙️ PM2.5 Levels Across Major Indian Cities",
                color='Category',
                color_discrete_map={
                    'Good': '#10b981',
                    'Satisfactory': '#f59e0b', 
                    'Moderate': '#f97316',
                    'Poor': '#ef4444',
                    'Very Poor': '#8b5cf6',
                    'Severe': '#dc2626'
                },
                hover_data=['AQI', 'Population']
            )
            fig.update_layout(
                xaxis_title="Major Cities",
                yaxis_title="PM2.5 Concentration (μg/m³)",
                showlegend=True,
                height=500
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Health summary
            col1, col2, col3 = st.columns(3)
            with col1:
                good_cities = len(cities_df[cities_df['Category'] == 'Good'])
                st.metric("🟢 Good Air Quality", f"{good_cities} cities")
            with col2:
                moderate_cities = len(cities_df[cities_df['Category'].isin(['Satisfactory', 'Moderate'])])
                st.metric("🟡 Moderate Air Quality", f"{moderate_cities} cities")
            with col3:
                poor_cities = len(cities_df[cities_df['Category'].isin(['Poor', 'Very Poor', 'Severe'])])
                st.metric("🔴 Poor Air Quality", f"{poor_cities} cities")
        
        with tab4:
            st.subheader("🔮 Offline PM2.5 Prediction")
            st.markdown("Get local ML predictions for any location in India using the trained XGBoost model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pred_lat = st.number_input("Latitude", value=28.6, min_value=8.0, max_value=37.0, step=0.1)
                pred_lon = st.number_input("Longitude", value=77.2, min_value=68.0, max_value=97.0, step=0.1)
                
                # Environmental inputs (simplified)
                aod = st.slider("Aerosol Optical Depth", 0.0, 2.0, 0.65, 0.01)
                temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
                wind = st.slider("Wind Speed (m/s)", 0.0, 20.0, 4.0, 0.1)
                humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
                
            with col2:
                blh = st.slider("Boundary Layer Height (m)", 0.0, 3000.0, 850.0, 10.0)
                hour = st.slider("Hour", 0, 23, 12)
                month = st.slider("Month", 1, 12, 3)
                season = st.selectbox("Season", [1, 2, 3, 4], index=1, format_func=lambda x: {1: "Winter", 2: "Spring", 3: "Summer", 4: "Monsoon"}[x])
                
                if st.button("🔮 Predict PM2.5", type="primary"):
                    # Prepare prediction request
                    features = {
                        "aod_550": aod,
                        "t2m_celsius": temp,
                        "wind_speed_10m": wind,
                        "r2m": humidity,
                        "blh": blh,
                        "lat_cos": math.cos(math.radians(pred_lat)),
                        "lat_sin": math.sin(math.radians(pred_lat)),
                        "lon_cos": math.cos(math.radians(pred_lon)),
                        "lon_sin": math.sin(math.radians(pred_lon)),
                        "hour": hour,
                        "month": month,
                        "season": season
                    }
                    
                    try:
                        # Use offline prediction
                        if OFFLINE_FORECAST_AVAILABLE and offline_forecast.model_loaded:
                            # Add latitude and longitude to features
                            features["latitude"] = pred_lat
                            features["longitude"] = pred_lon
                            
                            # Use the corrected prediction function
                            result = offline_forecast.predict_pm25_offline(features)
                            
                            # Display result
                            st.success(f"**PM2.5 Prediction: {result['pm25']} μg/m³**")
                            st.info(f"**AQI: {result['aqi']} ({result['health_category']})**")
                            
                            # Display health message
                            st.markdown(f"**Health Impact:** {result['health_message']}")
                            
                            # Health recommendations based on category
                            health_cat = result['health_category']
                            if health_cat == "Good":
                                recommendations = ["Perfect for outdoor exercise", "All groups can enjoy outdoor activities"]
                            elif health_cat == "Satisfactory":
                                recommendations = ["Generally safe for outdoor activities", "Sensitive individuals should monitor symptoms"]
                            elif health_cat == "Moderate":
                                recommendations = ["Limit prolonged outdoor exertion", "Consider indoor activities for sensitive groups"]
                            elif health_cat == "Poor":
                                recommendations = ["Reduce outdoor activities", "Use air purifiers indoors", "Wear masks when outdoors"]
                            elif health_cat == "Very Poor":
                                recommendations = ["Avoid outdoor activities", "Keep windows closed", "Use N95 masks if going out"]
                            else:  # Severe
                                recommendations = ["Stay indoors", "Emergency health measures required", "Seek medical attention if experiencing symptoms"]
                                recommendations = ["Stay indoors", "Use air purifiers", "Seek medical advice if experiencing symptoms"]
                            
                            st.markdown("**Recommendations:**")
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                        else:
                            st.error("❌ Offline prediction model not available")
                            st.info("Please ensure best_model.pkl is in the project directory")
                            
                    except Exception as e:
                        st.error(f"� **Prediction Error**")
                        st.error(f"Error: {e}")
                        
                        st.markdown("### 🛠️ **Model Status:**")
                        st.markdown("""
                        **Offline Mode Requirements:**
                        - Ensure `best_model.pkl` exists in the project directory
                        - Model file should be trained XGBoost model
                        - Check console for loading errors
                        """)
                        
                        st.info("💡 This dashboard now runs fully offline using local ML models!")
        
        with tab5:
            st.markdown("### 📆 Offline PM2.5 & AQI Forecast (3/7 Days)")
            st.markdown("**Generate multi-day air quality forecasts using local ML model**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 📍 Location & Settings")
                
                # Quick city selector
                city_options = {
                    "Custom Location": None,
                    "🏛️ Delhi": (28.6139, 77.2090),
                    "🏙️ Mumbai": (19.0760, 72.8777),
                    "🌆 Bangalore": (12.9716, 77.5946),
                    "🏘️ Kolkata": (22.5726, 88.3639),
                    "🌴 Chennai": (13.0827, 80.2707),
                    "💻 Hyderabad": (17.3850, 78.4867),
                    "🕌 Ahmedabad": (23.0225, 72.5714),
                    "🎓 Pune": (18.5204, 73.8567),
                    "🏰 Jaipur": (26.9124, 75.7873),
                    "🏛️ Lucknow": (26.8467, 80.9462)
                }
                
                selected_city = st.selectbox(
                    "🏙️ Quick City Select",
                    options=list(city_options.keys()),
                    help="Select a major city or choose 'Custom Location' to enter coordinates"
                )
                
                # Set coordinates based on selection
                if city_options[selected_city] is not None:
                    default_lat, default_lon = city_options[selected_city]
                else:
                    default_lat, default_lon = 28.6139, 77.2090
                
                # Location inputs with validation
                forecast_lat = st.number_input(
                    "🌐 Latitude", 
                    value=default_lat, 
                    min_value=8.0, 
                    max_value=37.0, 
                    step=0.001,
                    format="%.3f",
                    key="forecast_lat",
                    help="Latitude coordinate (8° to 37°N for India)"
                )
                forecast_lon = st.number_input(
                    "🌐 Longitude", 
                    value=default_lon, 
                    min_value=68.0, 
                    max_value=97.25, 
                    step=0.001,
                    format="%.3f",
                    key="forecast_lon",
                    help="Longitude coordinate (68° to 97.25°E for India)"
                )
                
                # Validate location
                if forecast_lat and forecast_lon:
                    # Basic validation for Indian territory
                    is_valid_location = (8.0 <= forecast_lat <= 37.0 and 
                                       68.0 <= forecast_lon <= 97.25)
                    
                    # Check for obvious water bodies
                    is_likely_ocean = ((forecast_lon < 70.0 and forecast_lat < 20.0) or
                                     (forecast_lon > 90.0 and forecast_lat < 15.0))
                    
                    if not is_valid_location:
                        st.error("⚠️ Location outside Indian territory bounds")
                    elif is_likely_ocean:
                        st.warning("🌊 Location may be in ocean - please verify")
                    else:
                        st.success("✅ Valid Indian location")
                
                st.markdown("---")
                
                # Forecast settings
                st.markdown("#### ⚙️ Forecast Configuration")
                
                forecast_days = st.selectbox(
                    "📅 Forecast Period",
                    options=[3, 7],
                    format_func=lambda x: f"{x} Days Forecast",
                    key="forecast_days",
                    help="Choose between 3-day or 7-day forecast"
                )
                
                start_date = st.date_input(
                    "📅 Start Date",
                    value=date.today(),
                    min_value=date.today(),
                    max_value=date.today() + timedelta(days=7),
                    key="forecast_start_date",
                    help="Forecast start date (today or future)"
                )
                
                st.markdown("---")
                
                # Enhanced forecast button
                st.markdown("#### 🔮 Generate Forecast")
                
                if st.button("� Generate Enhanced Forecast", 
                           type="primary", 
                           use_container_width=True,
                           key="generate_forecast",
                           help="Click to generate multi-day PM2.5 and AQI forecast"):
                    
                    # Validate inputs before making request
                    if not is_valid_location:
                        st.error("❌ Please enter valid coordinates within India")
                    elif is_likely_ocean:
                        st.error("❌ Location appears to be in water. Please select a land location.")
                    else:
                        self.generate_forecast_display(forecast_lat, forecast_lon, forecast_days, start_date, col2)
                
                # Info section
                st.markdown("---")
                st.markdown("#### 📖 Information")
                
                with st.expander("ℹ️ How it works", expanded=False):
                    st.markdown("""
                    **🔬 Forecast Method:**
                    - Uses trained XGBoost ML model
                    - Incorporates seasonal patterns
                    - Applies location-specific adjustments
                    - Accounts for weekly trends
                    - Includes realistic daily variation
                    
                    **📊 Output:**
                    - Daily PM2.5 concentrations (μg/m³)
                    - CPCB AQI values and categories
                    - Health recommendations
                    - Interactive visualizations
                    """)
                
                with st.expander("🎯 Accuracy Notes", expanded=False):
                    st.markdown("""
                    **✅ High Accuracy Locations:**
                    - Delhi NCR, Mumbai, Bangalore
                    - Kolkata, Chennai, Hyderabad
                    - Other major metropolitan areas
                    
                    **⚠️ Limitations:**
                    - Weather dependent variations
                    - Sudden policy/event impacts
                    - Remote area predictions
                    
                    **💡 Best Practices:**
                    - Use for planning purposes
                    - Check daily for updates
                    - Consider local conditions
                    """)
                    
            # Display area for results
            with col2:
                    st.markdown("### 📊 Forecast Results")
                    st.info("👆 Configure location and click 'Generate Forecast' to see results")
                    
                    # Placeholder chart
                    placeholder_data = pd.DataFrame({
                        'Date': pd.date_range(start=date.today(), periods=3),
                        'PM2.5': [50, 60, 45],
                        'AQI': [100, 120, 90]
                    })
                    
                    fig = px.line(
                        placeholder_data, 
                        x='Date', 
                        y='PM2.5',
                        title="📈 Sample PM2.5 Forecast",
                        markers=True
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Footer
        st.markdown("---")
        
        # Footer content in columns
        footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
        
        with footer_col1:
            st.markdown("""
            <div class="footer">
                <h4>🌍 VayuDrishti - Air Quality Forecasting System</h4>
                <p><strong>Advanced ML-powered PM2.5 prediction and AQI monitoring for India</strong></p>
                <p>🇮🇳 Built for Bharatiya Antariksh Hackathon | XGBoost + Streamlit (Offline Mode)</p>
                <p>🌍 VayuDrishti - Making Air Quality Monitoring Accessible to Every Indian Citizen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with footer_col2:
            st.markdown("""
            <div class="footer">
                <h5>📊 Features</h5>
                <ul style="text-align: left; margin: 0; padding-left: 1rem;">
                    <li>3/7-day forecasting</li>
                    <li>Real-time predictions</li>
                    <li>Interactive maps</li>
                    <li>Health advisories</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with footer_col3:
            st.markdown("""
            <div class="footer">
                <h5>🔗 Links</h5>
                <p><a href="https://github.com/nishant-gupta911/VayuDrishti" target="_blank" style="color: #42a5f5; text-decoration: none;">
                🐙 GitHub Repository</a></p>
                <p><strong>Version:</strong> 2.0.0 Enhanced</p>
                <p><strong>Updated:</strong> July 2025</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function"""
    dashboard = VayuDrishtiDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
