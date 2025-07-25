#!/usr/bin/env python3
"""
🌍 VayuDrishti Hackathon Launcher
Quick launch script for Bharatiya Antariksh demo
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("\n" + "="*50)
    print("🌍 VayuDrishti - Bharatiya Antariksh Edition")
    print("="*50)
    print("\n🚀 Launching Air Quality Intelligence Dashboard...\n")

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    dashboard_dir = Path(__file__).parent / "dashboard"
    requirements_file = dashboard_dir / "requirements_hackathon.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file), "--quiet"
        ], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_model():
    """Check if model file exists"""
    model_path = Path(__file__).parent / "models" / "best_model.pkl"
    if model_path.exists():
        print("✅ XGBoost model found")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure best_model.pkl is in the models directory")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard"""
    dashboard_dir = Path(__file__).parent / "dashboard"
    dashboard_file = dashboard_dir / "dashboard.py"
    
    print("\n🌐 Starting Streamlit dashboard...")
    print("Open your browser and go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")
    
    os.chdir(dashboard_dir)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Thank you for using VayuDrishti!")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Please run the launcher again")

def main():
    """Main launcher function"""
    print_banner()
    
    if not check_python():
        input("Press Enter to exit...")
        return
    
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    if not check_model():
        input("Press Enter to exit...")
        return
    
    launch_dashboard()

if __name__ == "__main__":
    main()
