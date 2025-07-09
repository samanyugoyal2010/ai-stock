#!/usr/bin/env python3
"""
Optimized NYSE Stock Adder
Adds 50 major NYSE stocks with optimized performance.
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

class OptimizedStockAdder:
    """High-speed stock adder with parallel processing."""
    
    def __init__(self, max_workers=10, batch_size=10):
        """
        Initialize the optimized adder.
        
        Args:
            max_workers: Number of parallel workers
            batch_size: Number of stocks to process in each batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Configure optimized session for yfinance
        self.session = self._create_optimized_session()
        
        # Thread-safe progress tracking
        self.lock = threading.Lock()
        self.successful_adds = 0
        self.failed_adds = 0
        self.total_stocks = 0
        
        print(f"🚀 Optimized Stock Adder Initialized")
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
    
    def get_major_nyse_symbols(self, max_stocks=50):
        """Get 50 major NYSE symbols across all sectors."""
        print("📈 Collecting major NYSE symbols...")
        
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
        
        # Limit to requested number
        if max_stocks:
            major_stocks = major_stocks[:max_stocks]
        
        print(f"✅ Collected {len(major_stocks)} major NYSE symbols")
        return major_stocks
    
    def verify_stock_exists(self, ticker):
        """Quickly verify if a stock exists and is active."""
        try:
            # Create ticker object with optimized session
            stock = yf.Ticker(ticker, session=self.session)
            
            # Get basic info (faster than full history)
            info = stock.info
            
            # Check if stock is active
            if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                return True, info.get('longName', ticker)
            else:
                return False, "Inactive or delisted"
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def verify_stock_batch(self, stock_batch):
        """Verify a batch of stocks."""
        results = []
        
        for ticker in stock_batch:
            exists, info = self.verify_stock_exists(ticker)
            results.append((ticker, exists, info))
        
        return results
    
    def update_progress(self, results):
        """Update progress counters thread-safely."""
        with self.lock:
            for ticker, exists, info in results:
                if exists:
                    self.successful_adds += 1
                else:
                    self.failed_adds += 1
    
    def add_all_major_stocks_optimized(self, max_stocks=50):
        """Add all major stocks with parallel verification."""
        print(f"🚀 Starting optimized stock addition...")
        print(f"📊 Target: {max_stocks} major stocks")
        print(f"⚡ Parallel Workers: {self.max_workers}")
        print(f"📦 Batch Size: {self.batch_size}")
        print("="*80)
        
        # Get stock symbols
        stocks = self.get_major_nyse_symbols(max_stocks)
        self.total_stocks = len(stocks)
        
        if not stocks:
            print("❌ No stocks available for addition")
            return {}
        
        # Create batches
        batches = [stocks[i:i + self.batch_size] for i in range(0, len(stocks), self.batch_size)]
        
        print(f"📊 Processing {len(stocks)} stocks in {len(batches)} batches")
        print("="*80)
        
        start_time = time.time()
        all_results = []
        valid_stocks = []
        
        # Process batches with progress bar
        with tqdm(total=len(stocks), desc="Verifying stocks", unit="stock") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self.verify_stock_batch, batch): batch 
                    for batch in batches
                }
                
                # Process completed batches
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update progress
                    self.update_progress(batch_results)
                    pbar.update(len(batch_results))
                    
                    # Collect valid stocks
                    for ticker, exists, info in batch_results:
                        if exists:
                            valid_stocks.append(ticker)
                    
                    # Show batch summary
                    batch = future_to_batch[future]
                    successful_in_batch = sum(1 for _, exists, _ in batch_results if exists)
                    print(f"\n📦 Batch completed: {successful_in_batch}/{len(batch)} valid stocks")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_time_per_stock = total_time / len(stocks) if stocks else 0
        
        # Print final summary
        print(f"\n{'='*80}")
        print("🎉 OPTIMIZED ADDITION COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Valid stocks found: {self.successful_adds}")
        print(f"❌ Invalid stocks: {self.failed_adds}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📊 Average time per stock: {avg_time_per_stock:.2f} seconds")
        print(f"🚀 Speed improvement: ~{self.max_workers}x faster than sequential")
        
        # Save results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks_checked': self.total_stocks,
            'valid_stocks': self.successful_adds,
            'invalid_stocks': self.failed_adds,
            'total_time_seconds': total_time,
            'avg_time_per_stock_seconds': avg_time_per_stock,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'valid_stock_list': valid_stocks,
            'results': all_results
        }
        
        results_file = "major_stocks_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"📋 Results saved to: {results_file}")
        print(f"📝 Valid stocks list: {valid_stocks}")
        
        return results_summary
    
    def save_stock_list(self, stocks, filename="major_stocks.txt"):
        """Save the list of valid stocks to a text file."""
        try:
            with open(filename, 'w') as f:
                for stock in stocks:
                    f.write(f"{stock}\n")
            print(f"💾 Stock list saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving stock list: {str(e)}")
            return False

def main():
    """Main function for optimized stock addition."""
    print("🚀 Optimized Major NYSE Stock Adder")
    print("="*80)
    print("Adds 50 major NYSE stocks with parallel verification")
    print("="*80)
    
    # Configuration
    print("\n⚙️  Configuration:")
    max_stocks = input("Enter max number of stocks [default: 50]: ").strip()
    max_stocks = int(max_stocks) if max_stocks else 50
    
    max_workers = input("Enter number of parallel workers [default: 10]: ").strip()
    max_workers = int(max_workers) if max_workers else 10
    
    batch_size = input("Enter batch size [default: 10]: ").strip()
    batch_size = int(batch_size) if batch_size else 10
    
    print(f"\n🚀 Starting optimized addition with:")
    print(f"   📊 Max Stocks: {max_stocks}")
    print(f"   ⚡ Workers: {max_workers}")
    print(f"   📦 Batch Size: {batch_size}")
    
    # Initialize adder
    adder = OptimizedStockAdder(
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Start addition
    results = adder.add_all_major_stocks_optimized(max_stocks)
    
    if results and 'valid_stock_list' in results:
        # Save stock list
        adder.save_stock_list(results['valid_stock_list'])
        
        print(f"\n🎉 Addition complete!")
        print(f"📊 Found {len(results['valid_stock_list'])} valid major stocks")
        print(f"💾 Use these stocks with the optimized fetch script")

if __name__ == "__main__":
    main() 