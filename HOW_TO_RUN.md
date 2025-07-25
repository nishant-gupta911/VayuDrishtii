# VayuDrishti: Installation & Setup Guide
## *Professional Air Quality Forecasting System - House Slytherin*

> **Complete deployment guide for VayuDrishti PM2.5 prediction system**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg)](.)

---

## 📋 System Requirements

### **Minimum Specifications**
- **Python**: 3.8 or higher (Recommended: Python 3.9-3.11)
- **Memory**: 4GB RAM minimum (8GB recommended for optimal performance)
- **Storage**: 2GB free disk space for dependencies and data
- **Network**: Internet connection for initial setup (optional for runtime)
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)

### **Required Software**
- Git version control system
- Python package manager (pip)
- Web browser (Chrome, Firefox, Safari, or Edge)

---

## 🚀 Installation Options

### **Option 1: Complete Development Environment** *(Recommended)*

#### **1. Repository Setup**
```bash
# Clone the repository
git clone https://github.com/nishant-gupta911/VayuDrishtii.git
cd VayuDrishtii

# Verify repository structure
ls -la
```

#### **2. Virtual Environment Configuration**
```bash
# Create isolated Python environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell/CMD):
venv\Scripts\activate

# Windows (Git Bash):
source venv/Scripts/activate

# macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

#### **3. Dependency Installation**
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(streamlit|xgboost|pandas)"
```

#### **4. Application Launch**
```bash
# Navigate to dashboard directory
cd dashboard

# Launch the application
streamlit run dashboard.py

# Alternative: Use Python module execution
python -m streamlit run dashboard.py
```

### **Option 2: Production Dashboard Only** *(Lightweight)*

For users who only need the dashboard functionality without development tools:

```bash
# Navigate to dashboard directory
cd VayuDrishtii/dashboard

# Install minimal production dependencies
pip install -r requirements_dashboard.txt

# Launch production dashboard
streamlit run dashboard.py
```

### **Option 3: Quick Start** *(Windows Users)*

```powershell
# Execute automated setup script (if available)
.\run_dashboard.bat

# Or use PowerShell execution policy bypass
powershell -ExecutionPolicy Bypass -File setup.ps1
```

---

## 🌐 Accessing the Application

### **Local Access**
```bash
# Default application URL
http://localhost:8501

# Custom port (if needed)
streamlit run dashboard.py --server.port 8080
```

### **Application Features**
Once launched, you'll have access to:
- ✅ **Interactive Map**: Pan-India PM2.5 visualization
- ✅ **City Monitoring**: Real-time predictions for major metropolitan areas
- ✅ **Custom Predictions**: Location-specific air quality forecasting
- ✅ **Health Advisory**: CPCB-compliant AQI categorization
- ✅ **Trend Analysis**: Multi-day forecasting capabilities

---

## 🔧 Advanced Configuration

### **Environment Variables** *(Optional)*
```bash
# Set custom configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows equivalent
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_ADDRESS=localhost
```

### **Performance Optimization**
```bash
# Enable multi-threading (for faster processing)
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Increase memory limit for large datasets
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
```

---

## 🛠️ Troubleshooting Guide

### **Common Issues & Solutions**

#### **1. Python Version Conflicts**
```bash
# Check Python version
python --version

# Use specific Python version (if multiple installed)
python3.9 -m venv venv
# or
py -3.9 -m venv venv  # Windows with Python Launcher
```

#### **2. Package Installation Errors**
```bash
# Clear pip cache
pip cache purge

# Reinstall with verbose output
pip install -r requirements.txt --verbose

# Install with user privileges (if permission denied)
pip install -r requirements.txt --user
```

#### **3. XGBoost Compatibility Issues**
```bash
# Install specific XGBoost version
pip uninstall xgboost
pip install xgboost==1.7.6

# For Apple Silicon Macs
pip install xgboost --no-binary xgboost
```

#### **4. Streamlit Port Conflicts**
```bash
# Kill existing Streamlit processes
pkill -f streamlit  # macOS/Linux
taskkill /F /IM python.exe  # Windows

# Use alternative port
streamlit run dashboard.py --server.port 8502
```

#### **5. Missing Model Files**
```bash
# Verify model file exists
ls -la models/best_model.pkl

# If missing, check data directory structure
ls -la data/*/
```

### **Performance Diagnostics**
```bash
# Check system resources
# macOS/Linux:
htop
free -h

# Windows:
tasklist /FI "IMAGENAME eq python.exe"
wmic computersystem get TotalPhysicalMemory
```

---

## 📊 Verification & Testing

### **Application Health Check**
```bash
# Verify all components are working
python -c "
import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
print('✅ All core dependencies loaded successfully')
"

# Test model loading
python -c "
import joblib
model = joblib.load('models/best_model.pkl')
print(f'✅ Model loaded: {type(model).__name__}')
"
```

### **Sample Prediction Test**
```bash
# Navigate to dashboard directory
cd dashboard

# Run prediction test
python -c "
from offline_forecast import predict_single
result = predict_single(28.6139, 77.2090)  # Delhi coordinates
print(f'✅ Sample prediction for Delhi: {result:.2f} μg/m³')
"
```

---

## 🔄 Updates & Maintenance

### **Keeping VayuDrishti Updated**
```bash
# Pull latest changes
git pull origin master

# Update dependencies
pip install -r requirements.txt --upgrade

# Clear Python cache (if needed)
find . -type d -name "__pycache__" -exec rm -rf {} +  # macOS/Linux
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"  # Windows
```

### **Backup & Recovery**
```bash
# Backup custom configurations
cp dashboard/config.toml dashboard/config.toml.backup

# Export environment for reproducibility
pip freeze > requirements_current.txt
```

---

## 🆘 Support & Resources

### **Getting Help**
- 📖 **Documentation**: [Project README](README.md)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/nishant-gupta911/VayuDrishtii/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/nishant-gupta911/VayuDrishtii/discussions)
- 📧 **Direct Contact**: [Nishant Gupta](https://github.com/nishant-gupta911)

### **Additional Resources**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost User Guide](https://xgboost.readthedocs.io/)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

---

## 🏆 Success Indicators

When everything is working correctly, you should see:
- ✅ Streamlit server starting without errors
- ✅ Application accessible at `http://localhost:8501`
- ✅ Interactive map loading with Indian geographical boundaries
- ✅ PM2.5 predictions generating for sample locations
- ✅ No missing dependency warnings in console

---

### 🎯 **Ready for Production | House Slytherin Excellence** 🐍
### ❤️ **Engineered for Accessibility | Made in India** 🇮🇳

*Successfully deployed? You're now ready to explore India's air quality intelligence with VayuDrishti!*

---

**For Triwizardathon 1.0 | Advanced AI/ML Forecasting | Version: 2.1.0 Professional Edition**
