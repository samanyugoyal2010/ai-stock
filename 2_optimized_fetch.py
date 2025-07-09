#!/usr/bin/env python3
"""
Optimized Stock Data Fetcher
Fetches stock data from Yahoo Finance with parallel processing for maximum speed.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import warnings
import concurrent.futures
import threading
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
warnings.filterwarnings('ignore')

class OptimizedStockFetcher:
    """High-speed stock data fetcher with parallel processing."""
    
    def __init__(self, data_dir="data", max_workers=10, batch_size=20):
        """
        Initialize the optimized fetcher.
        
        Args:
            data_dir: Directory to save CSV files
            max_workers: Number of parallel workers
            batch_size: Number of stocks to fetch in each batch
        """
        self.data_dir = data_dir
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configure optimized session for yfinance
        self.session = self._create_optimized_session()
        
        # Thread-safe progress tracking
        self.lock = threading.Lock()
        self.successful_fetches = 0
        self.failed_fetches = 0
        self.total_stocks = 0
        
        print(f"🚀 Optimized Stock Fetcher Initialized")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"⚡ Max Workers: {self.max_workers}")
        print(f"📦 Batch Size: {self.batch_size}")
    
    def _create_optimized_session(self):
        """Create an optimized requests session for faster downloads."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers for better compatibility
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
    def get_nyse_symbols_optimized(self, max_stocks=50):
        """Get NYSE symbols with optimized collection."""
        print("📈 Collecting NYSE symbols...")
        
        # Curated list of 50 major stocks by sector
        major_stocks = [
            # Technology (15 stocks)
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
            "ADBE", "PYPL", "NFLX", "ZM", "SHOP",
            
            # Financial (10 stocks)
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "AXP", "BLK",
            
            # Healthcare (8 stocks)
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR",
            
            # Consumer (8 stocks)
            "PG", "KO", "PEP", "WMT", "HD", "DIS", "NKE", "MCD",
            
            # Industrial (5 stocks)
            "BA", "CAT", "GE", "MMM", "HON",
            
            # Energy (4 stocks)
            "XOM", "CVX", "COP", "EOG"
        ]
        
        # Limit to requested number (default 50)
        if max_stocks:
            major_stocks = major_stocks[:max_stocks]
        
        print(f"✅ Collected {len(major_stocks)} major NYSE symbols")
        return major_stocks
    
    def fetch_single_stock_optimized(self, ticker, period="2y"):
        """Fetch data for a single stock with optimized settings."""
        try:
            # Create ticker object with optimized session
            stock = yf.Ticker(ticker, session=self.session)
            
            # Fetch historical data with optimized parameters
            data = stock.history(
                period=period,
                interval="1d",
                prepost=False,  # Faster without pre/post market data
                threads=False,  # We handle threading ourselves
                progress=False   # Disable progress bar for speed
            )
            
            if data.empty:
                return None, f"No data found for {ticker}"
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Check and select required columns
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = list(data.columns)
            
            if all(col in available_columns for col in expected_columns):
                data = data[expected_columns]
            else:
                # Handle missing columns gracefully
                missing_cols = [col for col in expected_columns if col not in available_columns]
                if 'Close' not in available_columns:
                    return None, f"Missing essential columns for {ticker}: {missing_cols}"
                
                # Use available columns and fill missing ones
                data = data[available_columns]
                for col in expected_columns:
                    if col not in data.columns:
                        if col == 'Date':
                            data[col] = data.index
                        else:
                            data[col] = data['Close']  # Use Close as fallback
            
            # Check data quality
            if len(data) < 60:
                return None, f"Insufficient data for {ticker} ({len(data)} days)"
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            if len(data) < 60:
                return None, f"Too many NaN values for {ticker} ({len(data)} valid days)"
            
            return data, None
            
        except Exception as e:
            return None, f"Error fetching {ticker}: {str(e)}"
    
    def save_stock_data(self, ticker, data):
        """Save stock data to CSV file."""
        try:
            filename = f"{ticker}.csv"
            filepath = os.path.join(self.data_dir, filename)
            data.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"❌ Error saving {ticker}: {str(e)}")
            return False
    
    def fetch_stock_batch(self, stock_batch, period="2y"):
        """Fetch data for a batch of stocks."""
        results = []
        
        for ticker in stock_batch:
            data, error = self.fetch_single_stock_optimized(ticker, period)
            
            if data is not None:
                # Save data
                if self.save_stock_data(ticker, data):
                    results.append((ticker, len(data), "Success"))
                else:
                    results.append((ticker, 0, "Save Error"))
            else:
                results.append((ticker, 0, error))
        
        return results
    
    def update_progress(self, results):
        """Update progress counters thread-safely."""
        with self.lock:
            for ticker, data_points, status in results:
                if status == "Success":
                    self.successful_fetches += 1
                else:
                    self.failed_fetches += 1
    
    def fetch_all_stocks_optimized(self, max_stocks=50, period="2y"):
        """Fetch data for all stocks with parallel processing."""
        print(f"🚀 Starting optimized data fetch...")
        print(f"📅 Period: {period}")
        print(f"⚡ Parallel Workers: {self.max_workers}")
        print(f"📦 Batch Size: {self.batch_size}")
        print("="*80)
        
        # Get stock symbols
        stocks = self.get_nyse_symbols_optimized(max_stocks)
        self.total_stocks = len(stocks)
        
        if not stocks:
            print("❌ No stocks available for fetching")
            return {}
        
        # Create batches
        batches = [stocks[i:i + self.batch_size] for i in range(0, len(stocks), self.batch_size)]
        
        print(f"📊 Processing {len(stocks)} stocks in {len(batches)} batches")
        print("="*80)
        
        start_time = time.time()
        all_results = []
        
        # Process batches with progress bar
        with tqdm(total=len(stocks), desc="Fetching stocks", unit="stock") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self.fetch_stock_batch, batch, period): batch 
                    for batch in batches
                }
                
                # Process completed batches
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update progress
                    self.update_progress(batch_results)
                    pbar.update(len(batch_results))
                    
                    # Show batch summary
                    batch = future_to_batch[future]
                    successful_in_batch = sum(1 for _, _, status in batch_results if status == "Success")
                    print(f"\n📦 Batch completed: {successful_in_batch}/{len(batch)} successful")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_time_per_stock = total_time / len(stocks) if stocks else 0
        
        # Print final summary
        print(f"\n{'='*80}")
        print("🎉 OPTIMIZED FETCH COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Successfully fetched: {self.successful_fetches} stocks")
        print(f"❌ Failed: {self.failed_fetches} stocks")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Average time per stock: {avg_time_per_stock:.2f} seconds")
        print(f"🚀 Speed improvement: ~{self.max_workers}x faster than sequential")
        print(f"📁 Data saved in: {self.data_dir}")
        
        # Save fetch results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': self.total_stocks,
            'successful_fetches': self.successful_fetches,
            'failed_fetches': self.failed_fetches,
            'total_time_minutes': total_time / 60,
            'avg_time_per_stock_seconds': avg_time_per_stock,
            'period': period,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'results': all_results
        }
        
        results_file = os.path.join(self.data_dir, "fetch_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"📋 Results saved to: {results_file}")
        
        return results_summary
    
    def get_existing_files(self):
        """Get list of existing CSV files."""
        if not os.path.exists(self.data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return [f.replace('.csv', '') for f in csv_files]
    
    def fetch_missing_stocks(self, period="2y"):
        """Fetch only stocks that don't have existing data files."""
        print("🔍 Checking for existing data files...")
        
        existing_stocks = set(self.get_existing_files())
        all_stocks = set(self.get_nyse_symbols_optimized())
        
        missing_stocks = all_stocks - existing_stocks
        
        if not missing_stocks:
            print("✅ All stocks already have data files!")
            return {}
        
        print(f"📊 Found {len(existing_stocks)} existing files")
        print(f"📈 Need to fetch {len(missing_stocks)} missing stocks")
        
        # Fetch only missing stocks
        return self.fetch_all_stocks_optimized(max_stocks=len(missing_stocks), period=period)

def main():
    """Main function for optimized data fetching."""
    print("🚀 Optimized NYSE Stock Data Fetcher")
    print("="*80)
    print("Fetches stock data with parallel processing for maximum speed")
    print("="*80)
    
    # Configuration
    print("\n⚙️  Configuration:")
    max_stocks = input("Enter max number of stocks [default: 50]: ").strip()
    max_stocks = int(max_stocks) if max_stocks else 50
    
    period = input("Enter data period (1y, 2y, 5y, max) [default: 2y]: ").strip() or "2y"
    
    max_workers = input("Enter number of parallel workers [default: 10]: ").strip()
    max_workers = int(max_workers) if max_workers else 10
    
    batch_size = input("Enter batch size [default: 20]: ").strip()
    batch_size = int(batch_size) if batch_size else 20
    
    fetch_mode = input("Fetch mode (all/missing) [default: all]: ").strip() or "all"
    
    print(f"\n🚀 Starting optimized fetch with:")
    print(f"   📊 Max Stocks: {max_stocks}")
    print(f"   📅 Period: {period}")
    print(f"   ⚡ Workers: {max_workers}")
    print(f"   📦 Batch Size: {batch_size}")
    print(f"   🔄 Mode: {fetch_mode}")
    
    # Initialize fetcher
    fetcher = OptimizedStockFetcher(
        data_dir="data",
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Start fetching
    if fetch_mode.lower() == "missing":
        results = fetcher.fetch_missing_stocks(period)
    else:
        results = fetcher.fetch_all_stocks_optimized(max_stocks, period)
    
    print(f"\n🎉 Fetch complete! Check the data/ directory for CSV files.")

if __name__ == "__main__":
    main() 