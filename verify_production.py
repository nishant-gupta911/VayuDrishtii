#!/usr/bin/env python3
"""
VayuDrishti Production Verification
Final check for Bharatiya Antariksh Hackathon submission
"""

import os
import sys
from pathlib import Path

def verify_project_structure():
    """Verify all required files are present"""
    required_files = [
        'README.md',
        'requirements.txt',
        'LICENSE',
        'dashboard/dashboard.py',
        'dashboard/offline_forecast.py',
        'models/best_model.pkl',
        'data/unified/cleaned_dataset.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def verify_dependencies():
    """Check if all dependencies can be imported"""
    dependencies = [
        'streamlit', 'pandas', 'numpy', 'xgboost', 
        'plotly', 'folium', 'sklearn', 'joblib'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        return False
    
    print("✅ All dependencies available")
    return True

def verify_model():
    """Check if the model can be loaded"""
    try:
        import joblib
        model = joblib.load('models/best_model.pkl')
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def verify_dashboard():
    """Test dashboard imports"""
    try:
        sys.path.append('dashboard')
        import offline_forecast
        print("✅ Dashboard modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 VayuDrishti Production Verification")
    print("🇮🇳 Bharatiya Antariksh Hackathon")
    print("=" * 50)
    
    checks = [
        ("Project Structure", verify_project_structure),
        ("Dependencies", verify_dependencies),
        ("ML Model", verify_model),
        ("Dashboard", verify_dashboard)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 SUCCESS: VayuDrishti is production-ready!")
        print("🚀 Run: streamlit run dashboard/dashboard.py")
        print("🌐 URL: http://localhost:8501")
    else:
        print("❌ ISSUES FOUND: Please fix the above errors")
    
    print("\n📊 Project Stats:")
    try:
        import pandas as pd
        df = pd.read_csv('data/unified/cleaned_dataset.csv')
        print(f"   - Training data: {len(df):,} samples")
        print(f"   - Features: {len(df.columns)} columns")
        print(f"   - Model size: {os.path.getsize('models/best_model.pkl'):,} bytes")
    except:
        print("   - Could not load dataset stats")

if __name__ == "__main__":
    main()
