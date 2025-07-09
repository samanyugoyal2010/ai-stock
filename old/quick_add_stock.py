#!/usr/bin/env python3
"""
Quick Stock Adder
Add a single stock to your prediction system quickly.
"""

import sys
import subprocess
from fetch_data import fetch_stock_data, save_stock_data

def quick_add_stock(ticker, epochs=50):
    """Quickly add a stock to the system."""
    print(f"🚀 Adding {ticker} to your prediction system...")
    
    # Step 1: Fetch data
    print(f"📊 Fetching {ticker} data...")
    data = fetch_stock_data(ticker, period="2y")
    
    if data is None:
        print(f"❌ Failed to fetch data for {ticker}")
        return False
    
    # Step 2: Save data
    print(f"💾 Saving {ticker} data...")
    success = save_stock_data(data, ticker, "data")
    
    if not success:
        print(f"❌ Failed to save data for {ticker}")
        return False
    
    # Step 3: Train model
    print(f"🤖 Training model for {ticker}...")
    try:
        result = subprocess.run([
            sys.executable, "train_specific.py", ticker, 
            "--epochs", str(epochs)
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {ticker} successfully added to your prediction system!")
            return True
        else:
            print(f"❌ Training failed for {ticker}")
            return False
            
    except Exception as e:
        print(f"❌ Error training {ticker}: {str(e)}")
        return False

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python3 quick_add_stock.py TICKER [EPOCHS]")
        print("Example: python3 quick_add_stock.py GOOGL 50")
        return
    
    ticker = sys.argv[1].upper()
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    print(f"📈 Quick Stock Adder")
    print("="*50)
    
    success = quick_add_stock(ticker, epochs)
    
    if success:
        print(f"\n🎉 {ticker} is now available for prediction!")
        print("Run: python3 run_prediction.py")
        print("Then select from the expanded menu.")
    else:
        print(f"\n❌ Failed to add {ticker}")

if __name__ == "__main__":
    main() 