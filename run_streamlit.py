#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Streamlit WDBC Breast Cancer Prediction
"""

import subprocess
import sys
import os

def main():
    print("🏥 Starting WDBC Breast Cancer Prediction App...")
    print("📱 Opening Streamlit application...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Run streamlit app
    print("🚀 Launching Streamlit app...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main() 