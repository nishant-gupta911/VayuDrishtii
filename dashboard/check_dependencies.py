#!/usr/bin/env python3
"""
Simple dependency checker for VayuDrishti Dashboard
Run this to see which packages need to be installed
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} - INSTALLED")
        return True
    except ImportError:
        print(f"❌ {package_name} - MISSING")
        return False

def main():
    print("🔍 VayuDrishti Dashboard - Dependency Check")
    print("=" * 50)
    
    packages = [
        ("streamlit", "streamlit"),
        ("folium", "folium"), 
        ("streamlit-folium", "streamlit_folium"),
        ("plotly", "plotly"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("requests", "requests")
    ]
    
    missing = []
    installed = 0
    
    for pkg_name, import_name in packages:
        if check_package(pkg_name, import_name):
            installed += 1
        else:
            missing.append(pkg_name)
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: {installed}/{len(packages)} packages installed")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n💡 To install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
        print("\n🔧 Or run the fix script: fix_dependencies.bat")
        return 1
    else:
        print("\n🎉 All packages are installed!")
        print("🚀 Dashboard is ready to run: streamlit run dashboard.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())
