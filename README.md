# VayuDrishti: Advanced PM2.5 Forecasting System
## *Intelligence Through Ambition - A Slytherin Approach to Air Quality Prediction*

> *"Where methodical precision meets innovative intelligence to decode atmospheric complexities across the Indian subcontinent"*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Performance](https://img.shields.io/badge/R²_Score-0.884-brightgreen.svg)](models/)
[![Coverage](https://img.shields.io/badge/Geographic_Coverage-Pan_India-orange.svg)](data/)

---

## 🏆 **TRIWIZARDATHON 1.0 SUBMISSION**
**Team:** House Slytherin | **Track:** AI/ML - Advanced Forecasting Systems  
**Focus:** Precision-driven atmospheric intelligence with multi-source data fusion

> **Real-time PM2.5 prediction system leveraging satellite data, meteorological parameters, and machine learning for comprehensive air quality assessment across India**The Slytherin Air Prophecy System

> *"Summoned from the depths of Slytherin's dungeons, where ambition meets intelligence to unveil the mysteries of the skies..."*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏰 **TRIWIZARDATHON 1.0** 🏆
**🐍 House Slytherin | 🧙‍♂️ Track: AI/ML - Forecasting the Unseen**

*Where cunning minds craft spells to pierce through the veil of atmospheric mysteries...*

> **🔮 The Dark Arts of PM2.5 Divination Across the Indian Realm**

---

## 📊 Project Overview

**VayuDrishti** is a sophisticated, production-ready PM2.5 forecasting system developed with Slytherin precision - combining methodical data engineering, advanced machine learning, and strategic innovation to deliver accurate air quality predictions across the Indian subcontinent.

*"Strategic intelligence applied to environmental challenges - where computational precision meets ambitious problem-solving"*

### 🛰️ **Multi-Source Data Integration**
- **Satellite Intelligence**: INSAT/MODIS Aerosol Optical Depth measurements
- **Meteorological Analytics**: ERA5 comprehensive weather parameter analysis (temperature, wind vectors, humidity)
- **Ground Truth Validation**: CPCB (Central Pollution Control Board) monitoring network integration
- **Machine Learning Engine**: XGBoost gradient boosting with optimized hyperparameters
- **Interactive Dashboard**: Streamlit-powered visualization and prediction interface

### 🎯 **Core Capabilities**
- **Real-time Inference**: Sub-100ms PM2.5 predictions for any Indian coordinate
- **Historical Analysis**: 2+ years of comprehensive atmospheric data processing
- **Offline Operation**: Complete functionality without external API dependencies
- **Regional Intelligence**: City-specific monitoring for major Indian metropolitan areas
- **Health Advisory**: CPCB-compliant AQI categorization with health recommendations

---

## 🚨 Problem Statement

*"Addressing the critical gap in accessible, accurate air quality forecasting across India's diverse geographic and economic landscape"*

### 📈 **Current Challenges**
- **Limited Infrastructure**: Approximately 300 monitoring stations serving 1.4 billion people
- **Geographic Inequality**: Dense urban monitoring with significant rural coverage gaps  
- **Temporal Delays**: Real-time reporting limitations and insufficient predictive capabilities
- **Technical Barriers**: Existing solutions require continuous internet connectivity and specialized knowledge

---

## 🔬 Technical Architecture & Innovation

### 🏗️ **Advanced Data Pipeline**
- **Multi-Modal Fusion**: Seamless integration of satellite, meteorological, and ground sensor data
- **Data Quality Assurance**: Comprehensive preprocessing with outlier detection and temporal consistency validation
- **Feature Engineering**: 12 optimized features including geographic encoding and temporal patterns
- **Scalable Processing**: Efficient handling of 100,000+ validated atmospheric records

### 🤖 **Machine Learning Implementation**
- **Model Performance**: XGBoost achieving R² > 0.88 (88%+ prediction accuracy)
- **Inference Speed**: Optimized for real-time applications with <100ms response time
- **Validation Framework**: K-fold cross-validation with temporal splitting for robust evaluation
- **Deployment Efficiency**: Compressed model size (279KB) for rapid deployment

### � **Interactive Web Application**
- **Geospatial Visualization**: Real-time color-coded air quality mapping across India
- **Metropolitan Monitoring**: Live predictions for major Indian cities
- **Custom Predictions**: Location-specific forecasting with health impact assessment
- **Trend Analysis**: Multi-day forecasting with confidence intervals
- **Offline-First Architecture**: Complete functionality without external dependencies

---

## 📁 Project Structure

```

VayuDrishti/
├── dashboard/ # Main application interface
│ ├── dashboard.py # Primary Streamlit application
│ ├── check_dependencies.py # Dependency verification utilities
│ ├── install_dependencies.py # Automated dependency installation
│ ├── offline_forecast.py # Core PM2.5 prediction engine
│ └── requirements_dashboard.txt # Production deployment requirements
│
├── data/ # Organized datasets and processing results
│ ├── cpcb/ # CPCB ground monitoring station data
│ ├── ml_ready/ # Preprocessed, ML-ready datasets
│ ├── processed/ # Intermediate processing results
│ └── satellite/ # MODIS AOD satellite data
│ └── unified/ # Merged datasets for model training
│
├── models/ # Trained models and evaluation metrics
│ ├── best_model.pkl # Production XGBoost model
│ ├── feature_importance_optimized.png # Feature analysis visualization
│ ├── model_metrics.json # Performance metrics and validation results
│ ├── model_summary.txt # Detailed model documentation
│ └── predictions_optimized.csv # Model prediction outputs
│
├── notebooks/ # Jupyter notebooks for analysis
│ ├── 01_Pan_India_Data_Collection.ipynb
│ └── VayuDrishti_PM25_Training.ipynb # EDA and model development
│
├── scripts/ # Core scripts for training and validation
│ ├── model_training.py # Model training script
│ ├── evaluate_model.py # Evaluation script
│ ├── data_collection.py # Update datasets
│ ├── preprocessing.py # Preprocessing utilities
│ └── verify_production.py # Production readiness validation
│
├── launch_hackathon.py # Application entry point
├── requirements.txt # Complete project dependencies
├── HOW_TO_RUN.md # Detailed installation guide
├── LICENSE # MIT License
├── README.md # Project documentation
└── .gitignore # Version control exclusions

```

---

## 🚀 Installation & Setup Guide

### 📋 **System Requirements**
```bash
✅ Python 3.8+ (Recommended: Python 3.9 or 3.10)
✅ pip package manager
✅ Minimum 4GB RAM for optimal performance
✅ Operating System: Windows/Linux/macOS
✅ Internet connection for initial setup (optional for runtime)
```

### ⚙️ **Installation Options**

#### **Option 1: Complete Development Environment**
```bash
# Clone the repository
git clone https://github.com/nishant-gupta911/VayuDrishti.git
cd VayuDrishti

# Install all dependencies
pip install -r requirements.txt

# Launch the application
cd dashboard
streamlit run dashboard.py
```

#### **Option 2: Production Dashboard Only**
```bash
# Navigate to dashboard directory
cd VayuDrishti/dashboard

# Install minimal production dependencies
pip install -r requirements_dashboard.txt

# Run the dashboard
streamlit run dashboard.py
```

#### **Option 3: Quick Start (Windows)**
```bash
# Execute batch file for automated setup
run_dashboard.bat
```

### 🌐 **Accessing the Application**
```bash
# Open your web browser and navigate to:
http://localhost:8501

# Application features:
✅ Interactive India-wide PM2.5 visualization
✅ Real-time metropolitan area monitoring
✅ Custom location prediction tools
✅ Multi-day forecasting with trend analysis
✅ Health advisory based on CPCB standards
```

---

## 📦 Dependencies & Requirements

### 🎯 **Deployment Configurations**

#### **Full Development Stack** (`requirements.txt`)
Complete environment for development, training, and production deployment:
- Data processing: pandas, numpy, scipy
- Machine learning: xgboost, scikit-learn, joblib
- Visualization: plotly, folium, matplotlib, seaborn
- Web framework: streamlit
- Analysis tools: jupyter, ipython

#### **Production Deployment** (`dashboard/requirements_dashboard.txt`)
Optimized dependencies for production dashboard deployment:
- streamlit (web application framework)
- pandas, numpy (data manipulation)
- xgboost (model inference)
- plotly, folium (interactive visualizations)
- joblib (model serialization)

---

## � Performance Metrics & Results

### 🎯 **Model Performance Validation**
```python
# Production Model Metrics
R² Score: 0.884          # 88.4% variance explained
MAE: 12.3 μg/m³          # Mean Absolute Error
RMSE: 18.7 μg/m³         # Root Mean Square Error
Inference Time: <100ms   # Real-time prediction capability
Model Size: 279KB        # Efficient deployment footprint
```

### 🗺️ **Geographic & Temporal Coverage**
- **Spatial Extent**: 8°N to 37°N, 68°E to 97°E (Complete Indian territory)
- **Temporal Range**: 2+ years comprehensive data (2023-2025)
- **Data Volume**: 100,000+ validated atmospheric measurements
- **Urban Coverage**: 10+ major metropolitan areas with continuous monitoring

### 🏙️ **Metropolitan Area Performance Analysis**
| City | Population | Average PM2.5 | Air Quality Status |
|------|------------|---------------|--------------------|
| Delhi | 32M | 45 μg/m³ | Moderate |
| Mumbai | 21M | 38 μg/m³ | Satisfactory |
| Bangalore | 13M | 32 μg/m³ | Satisfactory |
| Kolkata | 15M | 42 μg/m³ | Moderate |
| Chennai | 11M | 29 μg/m³ | Satisfactory |

### 🏥 **Health Impact Assessment (CPCB Standards)**
| PM2.5 Concentration | AQI Range | Health Category | Public Health Advisory |
|---------------------|-----------|-----------------|------------------------|
| 0-30 μg/m³ | 0-50 | 🟢 Good | Optimal for outdoor activities |
| 31-60 μg/m³ | 51-100 | 🟡 Satisfactory | Generally acceptable for most individuals |
| 61-90 μg/m³ | 101-200 | 🟠 Moderate | Sensitive groups may experience symptoms |
| 91-120 μg/m³ | 201-300 | 🔴 Poor | Health effects for general population |
| 121+ μg/m³ | 300+ | 🟣 Severe | Serious health implications for all |

---

## 🔧 Development & Advanced Usage

### 🔬 **Model Development & Training**
```bash
# Retrain model with updated datasets
python scripts/model_training.py --retrain

# Comprehensive model evaluation
python scripts/evaluate_model.py --metrics --validation
```

### 📊 **Data Pipeline Management**
```bash
# Update data sources with latest measurements
python scripts/data_collection.py --update --validate

# Execute preprocessing pipeline
python scripts/preprocessing.py --clean --normalize --feature-engineering
```

### 🧪 **Research & Analysis Environment**
```bash
# Launch Jupyter environment for analysis
jupyter notebook notebooks/

# Available analysis notebooks:
# - 01_Pan_India_Data_Collection.ipynb: Data acquisition methodology
# - VayuDrishti_PM25_Training.ipynb: Model development and validation
```




### 🛠️ **Development Environment Setup**
```bash
# 1. Fork and clone repository
git clone https://github.com/YOUR_USERNAME/VayuDrishti.git

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt

# 4. Run test suite
python -m pytest tests/ --coverage

# 5. Launch development server
streamlit run dashboard/dashboard.py
```

---


## 📚 **Acknowledgments & Technical Credits**

### 🏆 **Competition & Institutional Support**
- **Triwizardathon 1.0** - Competition platform and technical challenge framework
- **ISRO (Indian Space Research Organisation)** - Satellite data access and space technology support
- **CPCB (Central Pollution Control Board)** - Air quality standards and monitoring network data

### 📊 **Data Sources & Technology Partners**
- **NASA/ESA** - Satellite AOD data (MODIS, Sentinel-5P missions)
- **ECMWF** - ERA5 meteorological reanalysis datasets
- **IMD (India Meteorological Department)** - Regional meteorological insights
- **Open Source Community** - Python ecosystem, libraries, and frameworks

### 👨‍💻 **Development Team - House Slytherin**
- **Lead Developer**: Nishant Gupta ([@nishant-gupta911](https://github.com/nishant-gupta911))
- **Repository**: [VayuDrishti](https://github.com/nishant-gupta911/VayuDrishti)
- **Technical Contact**: [GitHub Profile](https://github.com/nishant-gupta911)

---

## � **Project Contributors**

| Contributor | Technical Contribution |
|-------------|------------------------|
| Nikita | **Phase 1 & 2 Lead**: Data collection and preprocessing pipeline development using CPCB, AOD, and ERA5 sources. ML-ready dataset preparation and validation. |
| Nishant Gupta | **Technical Lead**: Machine learning model development, API design, frontend implementation, optimization, and production deployment. |

*Special recognition to Nikita for her foundational work in data engineering and preprocessing infrastructure.*

### 📜 **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete terms and conditions.

---

## 🌟 **Vision & Strategic Roadmap**

**VayuDrishti** represents our commitment to democratizing environmental intelligence across India through innovative technology. Our strategic approach combines cutting-edge satellite technology with accessible machine learning to:

- 🏥 **Enhance Public Health**: Enable data-driven decisions for outdoor activities and health protection
- 🏛️ **Support Policy Development**: Provide evidence-based data for environmental regulations and urban planning
- 🔬 **Advance Research**: Create open datasets for academic research and commercial innovation
- 🌍 **Bridge Technology Gaps**: Make sophisticated air quality analysis accessible in underserved areas
- 🚀 **Demonstrate Innovation**: Showcase space technology applications for social and environmental impact

### 🎯 **Immediate Goals (2025)**
- Deploy real-time satellite data integration capabilities
- Expand coverage to neighboring South Asian countries
- Develop mobile applications for enhanced accessibility
- Integrate with public health advisory and alert systems

### 🌈 **Long-term Vision (2025-2030)**
- Global air quality forecasting platform
- Integration with IoT sensor networks and smart city infrastructure
- AI-powered personalized health recommendation engine
- Policy impact assessment and environmental planning tools

---

## 🚀 **Resources & Links**

- 🌐 **Live Application**: [Coming Soon - Production Deployment]
- 📖 **Technical Documentation**: `/docs/technical_report.pdf`
- 🐛 **Issue Tracking**: [GitHub Issues](https://github.com/nishant-gupta911/VayuDrishti/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/nishant-gupta911/VayuDrishti/discussions)
- 📊 **Performance Analytics**: [Model Metrics Dashboard](docs/performance_analysis.html)

---

### 🎯 **Developed for Triwizardathon 1.0** 🏆
### 🐍 **House Slytherin - Precision Through Ambition** 
### ❤️ **Engineered in India** 🇮🇳
### 🌍 **For sustainable environmental intelligence** 🌱

---

*"Strategic intelligence applied to environmental challenges - where computational precision meets innovative problem-solving. VayuDrishti demonstrates that methodical ambition and technical excellence can address complex atmospheric forecasting challenges across diverse geographic and socioeconomic landscapes."*

---

