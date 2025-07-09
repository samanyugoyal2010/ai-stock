#!/usr/bin/env python3
"""
Major NYSE Stocks Adder
Add the most important and liquid stocks from NYSE to your prediction system.
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
    """Main function to add major NYSE stocks."""
    print("📈 Major NYSE Stocks Adder")
    print("="*60)
    print("Adding the most important and liquid stocks from NYSE")
    print("Focusing on quality stocks with good data availability")
    print("="*60)
    
    # Major NYSE stocks by category
    major_stocks = {
        "Tech Giants": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM"
        ],
        "Financial": [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"
        ],
        "Healthcare": [
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN"
        ],
        "Consumer": [
            "PG", "KO", "PEP", "WMT", "HD", "DIS", "NKE", "MCD", "SBUX", "TGT"
        ],
        "Industrial": [
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "RTX", "LMT", "NOC"
        ],
        "Energy": [
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "KMI"
        ],
        "ETFs": [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "BND", "TLT", "GLD"
        ]
    }
    
    print("\n📊 Available Categories:")
    for i, (category, stocks) in enumerate(major_stocks.items(), 1):
        print(f"{i}. {category} ({len(stocks)} stocks)")
    
    print("\nOptions:")
    print("1. Add all categories (70+ stocks)")
    print("2. Add specific category")
    print("3. Add top 20 stocks only")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        # Add all categories
        all_stocks = []
        for category, stocks in major_stocks.items():
            all_stocks.extend(stocks)
        stocks_to_add = all_stocks
        print(f"\n🎯 Adding all {len(all_stocks)} major stocks...")
        
    elif choice == "2":
        # Add specific category
        print("\nSelect category:")
        for i, (category, stocks) in enumerate(major_stocks.items(), 1):
            print(f"{i}. {category} ({len(stocks)} stocks)")
        
        cat_choice = input("Enter category number: ").strip()
        try:
            cat_index = int(cat_choice) - 1
            categories = list(major_stocks.keys())
            if 0 <= cat_index < len(categories):
                category = categories[cat_index]
                stocks_to_add = major_stocks[category]
                print(f"\n🎯 Adding {category} stocks ({len(stocks_to_add)} stocks)...")
            else:
                print("❌ Invalid category number")
                return
        except ValueError:
            print("❌ Invalid input")
            return
            
    elif choice == "3":
        # Add top 20 stocks only
        top_20 = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "PG",
            "KO", "WMT", "HD", "DIS", "BA", "XOM", "SPY", "QQQ", "VTI", "VOO"
        ]
        stocks_to_add = top_20
        print(f"\n🎯 Adding top 20 stocks...")
        
    else:
        print("❌ Invalid choice")
        return
    
    # Get training parameters
    epochs = input("Enter number of training epochs [default: 30]: ").strip()
    epochs = int(epochs) if epochs.isdigit() else 30
    
    print(f"\n🚀 Starting batch addition with {epochs} epochs per stock...")
    print(f"Total stocks to add: {len(stocks_to_add)}")
    print(f"Estimated time: {len(stocks_to_add) * 3} minutes")
    print("This will take a while. Each stock takes 2-5 minutes to process.")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Add stocks
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(stocks_to_add, 1):
        print(f"\n[{i}/{len(stocks_to_add)}] Processing {ticker}...")
        
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
    print(f"📈 Total processed: {len(stocks_to_add)} stocks")
    
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

if __name__ == "__main__":
    main() 