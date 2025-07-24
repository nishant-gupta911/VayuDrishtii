#!/usr/bin/env python3
"""
Dashboard Dependencies Installer and Validator
This script ensures all required packages are installed for the VayuDrishti dashboard
"""

import subprocess
import sys
import importlib
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_and_install_packages():
    """Check and install required packages"""
    
    required_packages = {
        'streamlit': 'streamlit>=1.28.0',
        'streamlit_folium': 'streamlit-folium>=0.15.0',
        'folium': 'folium>=0.14.0',
        'plotly': 'plotly>=5.17.0',
        'pandas': 'pandas>=2.0.0',
        'numpy': 'numpy>=1.24.0',
        'requests': 'requests>=2.31.0',
        'dateutil': 'python-dateutil>=2.8.0'
    }
    
    missing_packages = []
    
    print("🔍 Checking dashboard dependencies...")
    print("-" * 50)
    
    for module_name, package_spec in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} - OK")
        except ImportError:
            print(f"❌ {module_name} - MISSING")
            missing_packages.append(package_spec)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        print("-" * 50)
        
        for package in missing_packages:
            install_package(package)
        
        print("\n🔄 Re-checking dependencies...")
        print("-" * 50)
        
        # Re-check after installation
        for module_name, package_spec in required_packages.items():
            try:
                importlib.import_module(module_name)
                print(f"✅ {module_name} - OK")
            except ImportError:
                print(f"❌ {module_name} - STILL MISSING")
                return False
    
    print("\n🎉 All dashboard dependencies are satisfied!")
    return True

def main():
    """Main function"""
    print("🌍 VayuDrishti Dashboard - Dependency Installer")
    print("=" * 60)
    
    success = check_and_install_packages()
    
    if success:
        print("\n🚀 Dashboard is ready to run!")
        print("Start with: streamlit run dashboard.py")
        sys.exit(0)
    else:
        print("\n❌ Some dependencies could not be installed.")
        print("Please install them manually:")
        print("pip install streamlit folium streamlit-folium plotly pandas numpy requests python-dateutil")
        sys.exit(1)

if __name__ == "__main__":
    main()
