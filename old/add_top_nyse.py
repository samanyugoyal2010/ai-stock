#!/usr/bin/env python3
"""
Top NYSE Stocks Adder
Add the top 50 most liquid and important NYSE stocks to your prediction system.
"""

import subprocess
import sys
import os
import time

def add_stock(ticker, epochs=30):
    """Add a single stock to the system."""
    print(f"\n📈 Adding {ticker}...")
    try:
        result = subprocess.run([
            sys.executable, "quick_add_stock.py", ticker, str(epochs)
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {ticker} added successfully!")
            return True
        else:
            print(f"❌ Failed to add {ticker}")
            return False
            
    except Exception as e:
        print(f"❌ Error adding {ticker}: {str(e)}")
        return False

def main():
    """Main function to add top NYSE stocks."""
    print("📈 Top NYSE Stocks Adder")
    print("="*60)
    print("Adding the top 50 most liquid and important NYSE stocks")
    print("These are the stocks that matter most for trading and prediction")
    print("="*60)
    
    # Top 50 NYSE stocks by market cap and liquidity
    top_50_nyse = [
        # Tech Giants (10)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
        
        # Financial (8)
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
        
        # Healthcare (8)
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR",
        
        # Consumer (8)
        "PG", "KO", "PEP", "WMT", "HD", "DIS", "NKE", "MCD",
        
        # Industrial (6)
        "BA", "CAT", "GE", "MMM", "HON", "UPS",
        
        # Energy (5)
        "XOM", "CVX", "COP", "EOG", "SLB",
        
        # ETFs (5)
        "SPY", "QQQ", "IWM", "VTI", "VOO"
    ]
    
    print(f"\n📊 Top 50 NYSE Stocks to Add:")
    print("="*40)
    for i, ticker in enumerate(top_50_nyse, 1):
        print(f"{i:2d}. {ticker}")
    print("="*40)
    
    print(f"\n🎯 This will add {len(top_50_nyse)} high-quality stocks")
    print("Estimated time: 2-3 hours")
    print("Each stock takes 2-5 minutes to process")
    
    # Get training parameters
    epochs = input("Enter number of training epochs [default: 30]: ").strip()
    epochs = int(epochs) if epochs.isdigit() else 30
    
    print(f"\n🚀 Starting batch addition with {epochs} epochs per stock...")
    print(f"Total stocks to add: {len(top_50_nyse)}")
    print(f"Estimated time: {len(top_50_nyse) * 3} minutes")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Add stocks
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(top_50_nyse, 1):
        print(f"\n[{i}/{len(top_50_nyse)}] Processing {ticker}...")
        
        if add_stock(ticker, epochs):
            successful += 1
        else:
            failed += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BATCH ADDITION SUMMARY")
    print("="*60)
    print(f"✅ Successfully added: {successful} stocks")
    print(f"❌ Failed to add: {failed} stocks")
    print(f"📈 Total processed: {len(top_50_nyse)} stocks")
    
    if successful > 0:
        print(f"\n🎉 You can now predict {successful} new stocks!")
        print("Run: python3 run_prediction.py")
        print("Then select from the expanded menu.")
        
        # Show new portfolio size
        data_dir = "data"
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            total_stocks = len(csv_files)
            print(f"\n📊 Your total portfolio now has {total_stocks} stocks!")
            
            # Show some examples
            print(f"\n📈 Example stocks you can now predict:")
            for i, ticker in enumerate(top_50_nyse[:10], 1):
                print(f"   {i}. {ticker}")

if __name__ == "__main__":
    main() 