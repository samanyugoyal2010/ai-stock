#!/usr/bin/env python3
"""
Stock Data Fetcher
Fetches daily stock data from Yahoo Finance API and saves it as CSV files.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period="2y"):
    """
    Fetch stock data for a given ticker from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        period (str): Time period to fetch ('1y', '2y', '5y', etc.)
    
    Returns:
        pandas.DataFrame: Stock data with OHLCV columns
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Check what columns we have and select only the ones we need
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = list(data.columns)
        
        print(f"Available columns: {available_columns}")
        
        # Select only the columns we need, in the right order
        if all(col in available_columns for col in expected_columns):
            data = data[expected_columns]
        else:
            # If some columns are missing, try to handle it gracefully
            print(f"Warning: Some expected columns are missing. Available: {available_columns}")
            # Try to get at least the essential columns
            essential_cols = ['Date', 'Close']
            if all(col in available_columns for col in essential_cols):
                data = data[essential_cols]
                # Add missing columns with default values
                for col in ['Open', 'High', 'Low', 'Volume']:
                    if col not in data.columns:
                        data[col] = data['Close']  # Use Close price as fallback
            else:
                print(f"Error: Essential columns missing. Cannot proceed.")
                return None
        
        print(f"Successfully fetched {len(data)} days of data for {ticker}")
        print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def save_stock_data(data, ticker, output_dir="data"):
    """
    Save stock data to CSV file.
    
    Args:
        data (pandas.DataFrame): Stock data to save
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save the CSV file
    """
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
        print(f"Data saved to {filepath}")
        return True
    
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def main():
    """Main function to run the stock data fetcher."""
    print("Stock Data Fetcher")
    print("=" * 50)
    
    while True:
        # Get ticker from user
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() in ['quit', 'q', 'exit']:
            print("Goodbye!")
            break
        
        if not ticker:
            print("Please enter a valid ticker symbol.")
            continue
        
        # Get period from user
        period = input("Enter time period (1y, 2y, 5y, max) [default: 2y]: ").strip()
        if not period:
            period = "2y"
        
        # Fetch and save data
        print(f"\nFetching data for {ticker}...")
        data = fetch_stock_data(ticker, period)
        
        if data is not None:
            success = save_stock_data(data, ticker)
            if success:
                print(f"✅ Successfully saved {ticker} data!")
            else:
                print(f"❌ Failed to save {ticker} data.")
        else:
            print(f"❌ Failed to fetch {ticker} data.")

if __name__ == "__main__":
    main() 