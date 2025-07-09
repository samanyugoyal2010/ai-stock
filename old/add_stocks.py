#!/usr/bin/env python3
"""
Batch Stock Adder
Add multiple stocks to your prediction system at once.
"""

import subprocess
import sys
import os

def add_stock(ticker, epochs=50):
    """Add a single stock to the system."""
    print(f"\n📈 Adding {ticker}...")
    try:
        result = subprocess.run([
            sys.executable, "train_specific.py", ticker, 
            "--epochs", str(epochs)
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
    """Main function to add multiple stocks."""
    print("📈 Batch Stock Adder")
    print("="*50)
    
    # Popular stocks to add
    popular_stocks = [
        "GOOGL",  # Google
        "AMZN",   # Amazon
        "NVDA",   # NVIDIA
        "META",   # Meta (Facebook)
        "NFLX",   # Netflix
        "AMD",    # Advanced Micro Devices
        "INTC",   # Intel
        "CRM",    # Salesforce
        "ADBE",   # Adobe
        "PYPL",   # PayPal
    ]
    
    # ETFs
    etfs = [
        "QQQ",    # Invesco QQQ Trust
        "IWM",    # iShares Russell 2000 ETF
        "VTI",    # Vanguard Total Stock Market ETF
        "VOO",    # Vanguard S&P 500 ETF
    ]
    
    print("Available stock categories:")
    print("1. Popular Tech Stocks")
    print("2. ETFs")
    print("3. Custom list")
    
    choice = input("\nSelect category (1-3): ").strip()
    
    if choice == "1":
        stocks_to_add = popular_stocks
        print(f"\n🎯 Adding {len(popular_stocks)} popular tech stocks...")
        
    elif choice == "2":
        stocks_to_add = etfs
        print(f"\n🎯 Adding {len(etfs)} ETFs...")
        
    elif choice == "3":
        custom_input = input("Enter stock tickers separated by spaces: ").strip()
        stocks_to_add = [ticker.upper() for ticker in custom_input.split()]
        print(f"\n🎯 Adding {len(stocks_to_add)} custom stocks...")
        
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    # Get training parameters
    epochs = input("Enter number of training epochs [default: 50]: ").strip()
    epochs = int(epochs) if epochs.isdigit() else 50
    
    print(f"\n🚀 Starting batch addition with {epochs} epochs per stock...")
    print("This may take a while. Each stock takes 2-5 minutes to train.")
    
    # Add stocks
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(stocks_to_add, 1):
        print(f"\n[{i}/{len(stocks_to_add)}] Processing {ticker}...")
        
        if add_stock(ticker, epochs):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*50)
    print("📊 BATCH ADDITION SUMMARY")
    print("="*50)
    print(f"✅ Successfully added: {successful} stocks")
    print(f"❌ Failed to add: {failed} stocks")
    print(f"📈 Total processed: {len(stocks_to_add)} stocks")
    
    if successful > 0:
        print(f"\n🎉 You can now predict {successful} new stocks!")
        print("Run: python3 run_prediction.py")
        print("Then select from the expanded menu.")

if __name__ == "__main__":
    main() 