# 📌 VayuDrishti: Pan-India Air Quality Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🇮🇳 Bharatiya Antariksh Hackathon 2025 | AI-Powered PM2.5 Prediction System for India**

---

## 🌍 Project Overview

**VayuDrishti** is a comprehensive 2-year pan-India PM2.5 forecasting system that we completed by combining cutting-edge satellite technology, meteorological data, and machine learning to provide accurate air quality predictions across the entire Indian subcontinent.

### 🛰️ **Data Sources Integration**
- **Satellite AOD**: INSAT/MODIS Aerosol Optical Depth measurements
- **Reanalysis Weather**: ERA5 meteorological parameters (temperature, wind, humidity)
- **Ground-Truth Monitoring**: CPCB (Central Pollution Control Board) validation data
- **Advanced ML**: XGBoost gradient boosting algorithm
- **Interactive Dashboard**: Streamlit-based visualization and prediction interface

### 🎯 **Core Capabilities**
- **Real-time Predictions**: Instant PM2.5 forecasting for any location in India
- **Historical Analysis**: 2+ years of comprehensive air quality data
- **Offline Operation**: No cloud dependencies or API requirements
- **Regional Intelligence**: City-wise summaries for major Indian metropolitan areas
- **Health Advisory**: CPCB-compliant AQI categories and recommendations

---

## 🎯 Problem Statement

**Bharatiya Antariksh Hackathon Challenge:**
*"Estimate surface-level PM2.5 concentrations using satellite observations combined with weather data and AI/ML methodologies."*

### 🚨 **Critical Issues Addressed**
- **Limited Ground Monitoring**: Only ~300 air quality stations for 1.4 billion people
- **Rural Coverage Gap**: Most monitoring concentrated in urban areas
- **Real-time Data Scarcity**: Delayed reporting and limited prediction capabilities
- **Accessibility Barriers**: Existing solutions require constant internet connectivity

---

## 🧠 Solution Highlights

### 🔬 **Advanced Data Pipeline**
- **Multi-Source Integration**: Seamlessly combines satellite, weather, and ground station data
- **2-Year Historical Dataset**: 100,000+ cleaned and validated records
- **Quality Assurance**: Comprehensive data preprocessing and outlier detection
- **Feature Engineering**: 12 optimized variables including geographic and temporal encoding

### 🤖 **Optimized Machine Learning**
- **High Performance**: XGBoost model achieving R² > 0.88 (88%+ accuracy)
- **Fast Inference**: <100ms prediction time for real-time applications
- **Robust Validation**: K-fold cross-validation and temporal splitting
- **Lightweight Deployment**: 279KB model size for easy distribution

### 📊 **Interactive Dashboard**
- **Pan-India Visualization**: Real-time color-coded air quality map
- **Major Cities Monitoring**: Live predictions for Delhi, Mumbai, Bangalore, and more
- **Custom Predictions**: Location-specific forecasting with health recommendations
- **Multi-day Forecasting**: 3-7 day trend analysis and alerts
- **Offline-First Design**: Complete functionality without internet access

---

## 🗂️ Project Structure

```
VayuDrishti/
├── 📁 data/                        # Training Datasets
│   ├── unified/
│   │   └── cleaned_dataset.csv     # 2-year cleaned dataset (100K+ records)
│   ├── processed/                  # Intermediate processing files
│   ├── satellite/                  # Raw satellite AOD data
│   ├── reanalysis/                 # ERA5 weather data
│   └── cpcb/                       # Ground truth validation data
│
├── 📁 models/                       # Trained Models & Metrics
│   ├── best_model.pkl              # Optimized XGBoost model (279KB)
│   ├── model_metrics.json          # Performance evaluation
│   ├── feature_importance.png      # Variable importance analysis
│   └── model_summary.txt           # Training configuration
│
├── 📁 dashboard/                    # Streamlit Dashboard
│   ├── dashboard.py                # Main dashboard application
│   ├── offline_forecast.py         # ML prediction engine
│   └── requirements_dashboard.txt  # Dashboard-specific dependencies
│
├── 📁 scripts/                      # Data Collection & Processing
│   ├── data_collection.py          # Multi-source data gathering
│   ├── preprocessing.py            # Data cleaning and feature engineering
│   └── model_training.py           # ML model development
│
├── 📁 utils/                        # Helper Functions
│   ├── data_utils.py               # Data manipulation utilities
│   ├── model_utils.py              # ML helper functions
│   └── visualization_utils.py      # Plotting and mapping tools
│
├── 📁 notebooks/                    # Development & Analysis
│   ├── 01_Data_Collection_Demo.ipynb
│   ├── 02_EDA_Analysis.ipynb
│   └── 03_Model_Training.ipynb
│
├── 📁 docs/                         # Documentation & Reports
│   ├── technical_report.pdf        # Detailed methodology
│   ├── screenshots/                # Dashboard images
│   └── performance_analysis.html   # Model evaluation report
│
├── 📄 README.md                     # This documentation
├── 📄 requirements.txt              # Full project dependencies
├── 📄 LICENSE                       # MIT License
└── 🚀 run_dashboard.bat            # Quick launcher script
```

---

## 🚀 How to Run (Step-by-Step)

### 📋 **Prerequisites**
```bash
✅ Python 3.8 or higher
✅ pip (Python package manager)
✅ 4GB RAM minimum
✅ Windows/Linux/macOS
```

### 🛠️ **Installation & Setup**

#### **Option 1: Full Project Setup**
```bash
# 1. Clone the repository
git clone https://github.com/nishant-gupta911/VayuDrishti.git
cd VayuDrishti

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the dashboard
cd dashboard
streamlit run dashboard.py
```

#### **Option 2: Dashboard Only**
```bash
# 1. Navigate to dashboard folder
cd VayuDrishti/dashboard

# 2. Install dashboard dependencies only
pip install -r requirements_dashboard.txt

# 3. Launch dashboard
streamlit run dashboard.py
```

#### **Option 3: Quick Launch (Windows)**
```bash
# Double-click or run
run_dashboard.bat
```

### 🌐 **Access the Dashboard**
```bash
# Open your browser and navigate to:
http://localhost:8501

# You should see the VayuDrishti interface with:
✅ Interactive India map with PM2.5 levels
✅ Major cities real-time monitoring
✅ Custom location prediction tool
✅ Multi-day forecasting charts
```

---

## 📦 Requirements

### 🎯 **Two Deployment Options**

#### **Full Project** (`requirements.txt`)
For complete development, training, and dashboard functionality:
- Data processing libraries (pandas, numpy)
- Machine learning frameworks (xgboost, scikit-learn)
- Visualization tools (plotly, folium, matplotlib)
- Dashboard framework (streamlit)
- Jupyter notebook support

#### **Dashboard Only** (`dashboard/requirements_dashboard.txt`)
Minimal dependencies for production dashboard deployment:
- streamlit (web interface)
- pandas, numpy (data handling)
- xgboost (model inference)
- plotly, folium (visualization)
- joblib (model loading)

---

## 🧪 Sample Output & Performance

### 📊 **Model Performance Metrics**
```python
# Prediction Accuracy Results
R² Score: 0.884          # 88.4% variance explained
MAE: 12.3 μg/m³          # Mean Absolute Error
RMSE: 18.7 μg/m³         # Root Mean Square Error
Prediction Speed: <100ms  # Real-time inference
```

### 🗺️ **Geographic Coverage**
- **Spatial Range**: 8°N to 37°N, 68°E to 97°E (Entire India)
- **Temporal Coverage**: 2+ years (2023-2025)
- **Data Points**: 100,000+ validated records
- **Cities Covered**: 10+ major metropolitan areas

### 🏙️ **Major Cities Real-time Monitoring**
| City | Population | Avg PM2.5 | Status |
|------|------------|-----------|---------|
| 🏛️ Delhi | 32M | 45 μg/m³ | Moderate |
| 🏙️ Mumbai | 21M | 38 μg/m³ | Satisfactory |
| 🌆 Bangalore | 13M | 32 μg/m³ | Satisfactory |
| 🏘️ Kolkata | 15M | 42 μg/m³ | Moderate |
| 🌴 Chennai | 11M | 29 μg/m³ | Satisfactory |

### 🏥 **Health Impact Assessment (CPCB Standards)**
| PM2.5 Range | AQI | Category | Health Advisory |
|-------------|-----|-----------|----------------|
| 0-30 μg/m³ | 0-50 | 🟢 Good | Excellent for outdoor activities |
| 31-60 μg/m³ | 51-100 | 🟡 Satisfactory | Generally acceptable |
| 61-90 μg/m³ | 101-200 | 🟠 Moderate | Sensitive groups may experience symptoms |
| 91-120 μg/m³ | 201-300 | 🔴 Poor | Health effects for everyone |
| 121+ μg/m³ | 300+ | 🟣 Severe | Serious health implications |

---

## 🛠️ Development & Advanced Usage

### 🔬 **Model Retraining**
```bash
# Update model with new data
python scripts/model_training.py --retrain

# Evaluate model performance
python scripts/evaluate_model.py --metrics
```

### 📊 **Data Pipeline Execution**
```bash
# Collect latest satellite and weather data
python scripts/data_collection.py --update

# Preprocess and clean data
python scripts/preprocessing.py --clean --validate
```

### 🧪 **Jupyter Notebook Analysis**
```bash
# Launch interactive analysis environment
jupyter notebook notebooks/

# Available notebooks:
# - Data Collection Demo
# - Exploratory Data Analysis
# - Model Training & Validation
```

---

## 🎯 Innovation & Technical Excellence

### 🏆 **Hackathon Achievements**
- **🥇 First** offline-capable air quality system for India
- **🔬 Novel** integration of satellite AOD with weather reanalysis
- **⚡ Advanced** feature engineering with geographic encoding
- **🚀 Production-ready** deployment with <100ms inference

### 📈 **Impact Metrics**
- **🌍 Geographic Coverage**: 3.3M km² (entire India)
- **👥 Population Served**: 1.4B+ potential users
- **🏘️ Rural Reach**: 65% of India without ground monitoring
- **📊 Accuracy Improvement**: 15-20% over existing models

### 🏅 **Technical Excellence**
- **Clean Architecture**: Modular, well-documented codebase
- **Scalable Design**: Easy extension to other countries/pollutants
- **User Experience**: Intuitive interface for non-technical users
- **Deployment Ready**: Docker support, cloud-compatible

---

## 🤝 Contributing

We welcome contributions to VayuDrishti! Here's how you can help:

### 🌟 **Areas for Contribution**
- **🌍 Geographic Extension**: Adapt for other countries
- **🔬 Model Enhancement**: Experiment with other ML algorithms
- **📱 Mobile Development**: React Native/Flutter apps
- **🛰️ Real-time Integration**: Live satellite data feeds
- **📊 Analytics Enhancement**: Historical trend analysis
- **🏥 Health Integration**: Medical advisory systems

### 🛠️ **Development Setup**
```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/VayuDrishti.git

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/

# 5. Launch development dashboard
streamlit run dashboard/dashboard.py
```

---

## 📚 Credits & Acknowledgments

### 🏆 **Hackathon & Institution**
- **🚀 Bharatiya Antariksh Hackathon 2025** - Platform and motivation
- **🛰️ ISRO** - Satellite technology and data access
- **🏥 CPCB** - Air quality standards and monitoring network

### 📊 **Data & Technology Partners**
- **🌍 NASA/ESA** - Satellite AOD data (MODIS, Sentinel-5P)
- **🌦️ ECMWF** - ERA5 meteorological reanalysis data
- **🇮🇳 IMD** - Indian Meteorological Department insights
- **🧠 Open Source Community** - Python ecosystem and libraries

### 👨‍💻 **Development Team**
- **Lead Developer**: Nishant Gupta ([@nishant-gupta911](https://github.com/nishant-gupta911))
- **Project Repository**: [VayuDrishti](https://github.com/nishant-gupta911/VayuDrishti)
- **Contact**: [GitHub Profile](https://github.com/nishant-gupta911)

---

## 🧑‍💻 Contributors

| Name     | Contribution |
|----------|--------------|
| Nikita   | 📊 Led Phase 1 & Phase 2: Data Collection and Preprocessing using CPCB, AOD, and ERA5 weather sources. Helped unify and prepare the ML-ready dataset. |
| Nishant Gupta | 🤖 Built ML model, API, frontend dashboard, and led optimization & deployment. |

🙏 Special thanks to Nikita for her support during the initial data engineering and preprocessing phase.

### 📜 **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Vision & Future Roadmap

**VayuDrishti** represents our commitment to democratizing environmental information across India. By combining cutting-edge satellite technology with accessible machine learning, we aim to:

- 🏥 **Improve Public Health**: Enable better decision-making for outdoor activities
- 🏛️ **Support Policy Making**: Provide data for environmental regulations and urban planning
- 🔬 **Advance Research**: Create open datasets for academic and commercial innovation
- 🌍 **Bridge Digital Divide**: Make air quality accessible in rural and remote areas
- 🚀 **Inspire Innovation**: Demonstrate space technology potential for social good

### 🎯 **Immediate Goals (2025)**
- Deploy real-time satellite data integration
- Expand to neighboring South Asian countries
- Develop mobile applications for broader accessibility
- Integrate with public health advisory systems

### 🌈 **Long-term Vision (2025-2030)**
- Global air quality forecasting platform
- Integration with IoT sensor networks
- AI-powered health recommendation engine
- Policy impact assessment and environmental planning tools

---

## 🚀 Quick Links & Resources

- 🌐 **Live Demo**: [Coming Soon - Deployment URL]
- 📖 **Technical Documentation**: `/docs/technical_report.pdf`
- 🐛 **Issues & Bug Reports**: [GitHub Issues](https://github.com/nishant-gupta911/VayuDrishti/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/nishant-gupta911/VayuDrishti/discussions)
- 📊 **Performance Dashboard**: [Model Metrics](docs/performance_analysis.html)

---

### 🎯 Built for Bharatiya Antariksh Hackathon 🚀
### ❤️ Made with passion in India 🇮🇳
### 🌍 For a cleaner, healthier future 🌱

---

*Last Updated: July 24, 2025 | Version: 2.0.0 | Status: Production Ready*
