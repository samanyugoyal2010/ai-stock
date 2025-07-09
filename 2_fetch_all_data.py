#!/usr/bin/env python3
"""
NYSE Data Fetcher for All Stocks
Fetches data for all NYSE stocks from the symbols file.
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime
import sys

def load_symbols_from_file(filename="nyse_symbols.txt"):
    """Load symbols from the text file."""
    try:
        if not os.path.exists(filename):
            print(f"❌ Symbols file not found: {filename}")
            print("Please run: python3 1_add_all_nyse.py")
            return []
        
        with open(filename, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Loaded {len(symbols)} symbols from {filename}")
        return symbols
        
    except Exception as e:
        print(f"❌ Error loading symbols: {str(e)}")
        return []

def fetch_stock_data(ticker, period="2y"):
    """Fetch stock data for a given ticker."""
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        data = stock.history(period=period)
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Check what columns we have and select only the ones we need
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = list(data.columns)
        
        # Select only the columns we need, in the right order
        if all(col in available_columns for col in expected_columns):
            data = data[expected_columns]
        else:
            # If some columns are missing, try to handle it gracefully
            print(f"⚠️  Some expected columns missing for {ticker}. Available: {available_columns}")
            # Try to get at least the essential columns
            essential_cols = ['Date', 'Close']
            if all(col in available_columns for col in essential_cols):
                data = data[essential_cols]
                # Add missing columns with default values
                for col in ['Open', 'High', 'Low', 'Volume']:
                    if col not in data.columns:
                        data[col] = data['Close']  # Use Close price as fallback
            else:
                print(f"❌ Essential columns missing for {ticker}. Skipping.")
                return None
        
        # Check if we have enough data
        if len(data) < 30:
            print(f"⚠️  Insufficient data for {ticker} ({len(data)} days). Skipping.")
            return None
        
        return data
    
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {str(e)}")
        return None

def save_stock_data(data, ticker, output_dir="data"):
    """Save stock data to CSV file."""
    if data is None:
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"{ticker.upper()}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Save to CSV
        data.to_csv(filepath, index=False)
        return True
    
    except Exception as e:
        print(f"❌ Error saving data for {ticker}: {str(e)}")
        return False

def fetch_all_stocks_data(symbols, period="2y", delay=1):
    """Fetch data for all stocks."""
    print(f"📊 Fetching data for {len(symbols)} stocks...")
    print(f"Period: {period}")
    print(f"Delay between requests: {delay} seconds")
    print("="*60)
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    for i, ticker in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Fetching {ticker}...")
        
        try:
            # Fetch data
            data = fetch_stock_data(ticker, period)
            
            if data is not None:
                # Save data
                if save_stock_data(data, ticker):
                    print(f"✅ {ticker}: {len(data)} days of data saved")
                    successful += 1
                else:
                    print(f"❌ {ticker}: Failed to save data")
                    failed += 1
            else:
                print(f"⚠️  {ticker}: No data available")
                skipped += 1
            
            # Progress update
            if i % 10 == 0:
                print(f"\n📈 Progress: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")
                print(f"✅ Successful: {successful}, ❌ Failed: {failed}, ⚠️  Skipped: {skipped}")
            
            # Delay to avoid overwhelming the API
            time.sleep(delay)
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user at {ticker}")
            break
        except Exception as e:
            print(f"❌ Unexpected error for {ticker}: {str(e)}")
            failed += 1
            time.sleep(delay)
    
    return successful, failed, skipped

def main():
    """Main function."""
    print("📊 NYSE Data Fetcher for All Stocks")
    print("="*60)
    print("This script fetches data for all NYSE stocks")
    print("from the symbols file created by script 1.")
    print("="*60)
    
    # Load symbols
    symbols = load_symbols_from_file()
    
    if not symbols:
        print("❌ No symbols to process")
        return
    
    # Get parameters
    period = input("Enter time period (1y, 2y, 5y, max) [default: 2y]: ").strip()
    if not period:
        period = "2y"
    
    delay_input = input("Enter delay between requests in seconds [default: 1]: ").strip()
    delay = float(delay_input) if delay_input and delay_input.replace('.', '').isdigit() else 1.0
    
    print(f"\n🚀 Starting data fetch for {len(symbols)} stocks...")
    print(f"Period: {period}")
    print(f"Delay: {delay} seconds")
    print(f"Estimated time: {len(symbols) * delay / 60:.1f} minutes")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Fetch all data
    start_time = time.time()
    successful, failed, skipped = fetch_all_stocks_data(symbols, period, delay)
    end_time = time.time()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DATA FETCH SUMMARY")
    print("="*60)
    print(f"✅ Successfully fetched: {successful} stocks")
    print(f"❌ Failed to fetch: {failed} stocks")
    print(f"⚠️  Skipped (no data): {skipped} stocks")
    print(f"📈 Total processed: {len(symbols)} stocks")
    print(f"⏱️  Time taken: {(end_time - start_time)/60:.1f} minutes")
    
    if successful > 0:
        print(f"\n🎉 Successfully fetched data for {successful} stocks!")
        print("Next step: Run: python3 3_train_all_models.py")
        
        # Show some examples
        data_dir = "data"
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            print(f"\n📊 Data files created: {len(csv_files)}")
            if csv_files:
                print("Example files:")
                for i, file in enumerate(csv_files[:10], 1):
                    print(f"   {i}. {file}")
                if len(csv_files) > 10:
                    print(f"   ... and {len(csv_files) - 10} more")

if __name__ == "__main__":
    main() 