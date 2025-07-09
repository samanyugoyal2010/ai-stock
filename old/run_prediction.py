#!/usr/bin/env python3
"""
Interactive Stock Price Prediction Runner
Easy-to-use script to predict stock prices for tomorrow.
"""

import os
import sys
import subprocess
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'scikit-learn', 'yfinance'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_available_stocks():
    """Check what stocks are available for prediction."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in csv_files]
    return sorted(tickers)

def main():
    """Main function to run the prediction."""
    print("📈 Interactive Stock Price Predictor")
    print("="*50)
    print("This script will predict stock prices for tomorrow")
    print("using OHLCV (Open, High, Low, Close, Volume) data.")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot proceed without required packages.")
        return
    
    # Check if prediction script exists
    if not os.path.exists("predict_tomorrow.py"):
        print("❌ predict_tomorrow.py not found!")
        print("Please ensure all project files are in the same directory.")
        return
    
    # Check available stocks
    available_stocks = check_available_stocks()
    if not available_stocks:
        print("\n❌ No trained stocks found!")
        print("Please train some stocks first using:")
        print("   python3 train_specific.py TICKER")
        print("   Examples:")
        print("     python3 train_specific.py AAPL")
        print("     python3 train_specific.py TSLA")
        print("     python3 train_specific.py MSFT")
        print("\nAfter training, run this script again to make predictions.")
        return
    
    print(f"\n✅ Found {len(available_stocks)} trained stocks: {', '.join(available_stocks)}")
    print("\n🚀 Starting interactive prediction...")
    print("This may take a few minutes as the model needs to:")
    print("1. Fetch latest stock data")
    print("2. Train the neural network model")
    print("3. Make the prediction for tomorrow")
    print("\n⏳ Please wait...\n")
    
    try:
        # Run the prediction script
        result = subprocess.run([sys.executable, "predict_tomorrow.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Prediction completed successfully!")
        else:
            print(f"\n❌ Prediction failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running prediction: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have internet connection for data fetching")
        print("2. Check if all required files are present")
        print("3. Try running 'python3 predict_tomorrow.py' directly")

if __name__ == "__main__":
    main() 