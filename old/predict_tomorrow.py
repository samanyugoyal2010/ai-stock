#!/usr/bin/env python3
"""
Stock Price Predictor for Tomorrow
Predicts stock prices for tomorrow using OHLCV data and trained model.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from model import StockPredictor, StockDataset, create_model
from fetch_data import fetch_stock_data, save_stock_data

class StockPricePredictor:
    """Predictor class for stock prices."""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the predictor.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.current_ticker = None
        
        print(f"Using device: {self.device}")
    
    def get_available_tickers(self):
        """Get list of available stock tickers from CSV files."""
        if not os.path.exists(self.data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        return sorted(tickers)
    
    def display_available_stocks(self):
        """Display available stocks for prediction."""
        tickers = self.get_available_tickers()
        
        if not tickers:
            print("❌ No trained stocks found!")
            print("Please train some stocks first using:")
            print("   python3 train_specific.py TICKER")
            print("   Example: python3 train_specific.py AAPL")
            return None
        
        print("\n📊 Available Trained Stocks:")
        print("=" * 40)
        for i, ticker in enumerate(tickers, 1):
            # Get some info about the stock data
            filepath = os.path.join(self.data_dir, f"{ticker}.csv")
            if os.path.exists(filepath):
                data = pd.read_csv(filepath)
                data['Date'] = pd.to_datetime(data['Date'])
                latest_date = data['Date'].max().date()
                days_of_data = len(data)
                current_price = data['Close'].iloc[-1]
                print(f"{i:2d}. {ticker:5s} | ${current_price:7.2f} | {days_of_data:3d} days | Updated: {latest_date}")
        
        print("=" * 40)
        return tickers
    
    def select_stock(self):
        """Let user select which stock to predict."""
        tickers = self.display_available_stocks()
        
        if not tickers:
            return None
        
        while True:
            try:
                choice = input(f"\nSelect stock (1-{len(tickers)}) or 'q' to quit: ").strip()
                
                if choice.lower() in ['q', 'quit', 'exit']:
                    print("Goodbye!")
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(tickers):
                    selected_ticker = tickers[choice_num - 1]
                    print(f"\n✅ Selected: {selected_ticker}")
                    return selected_ticker
                else:
                    print(f"❌ Please enter a number between 1 and {len(tickers)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                return None
    
    def ensure_latest_data(self, ticker):
        """Ensure we have the latest stock data."""
        print(f"📊 Fetching latest {ticker} stock data...")
        
        # Fetch latest data (last 2 years)
        data = fetch_stock_data(ticker, period="2y")
        
        if data is not None:
            # Save to data directory
            save_stock_data(data, ticker, self.data_dir)
            print(f"✅ Latest data saved for {ticker}")
            return data
        else:
            print(f"❌ Failed to fetch latest data for {ticker}")
            return None
    
    def load_and_prepare_data(self, ticker):
        """Load and prepare data for prediction."""
        print(f"📈 Loading and preparing {ticker} data...")
        
        # Load data
        filepath = os.path.join(self.data_dir, f"{ticker}.csv")
        if not os.path.exists(filepath):
            print(f"❌ Data file not found: {filepath}")
            return None
        
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"📅 Data range: {data['Date'].min().date()} to {data['Date'].max().date()}")
        print(f"📊 Total days: {len(data)}")
        
        return data
    
    def train_model(self, data, ticker, epochs=50):
        """Train the model on historical data."""
        print(f"🤖 Training prediction model for {ticker}...")
        
        # Create dataset
        dataset = StockDataset(data, sequence_length=30)
        self.scaler = dataset.scaler
        
        # Split data (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create and train model
        self.model = create_model(input_size=5, hidden_size=128, num_layers=3, dropout=0.2)
        self.model = self.model.to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"🔄 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                last_outputs = outputs[:, -1, 0].unsqueeze(1)
                loss = criterion(last_outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss = {train_loss:.6f}")
        
        print("✅ Model training completed!")
    
    def predict_tomorrow(self, ticker):
        """Predict stock price for tomorrow."""
        print(f"🔮 Predicting tomorrow's {ticker} stock price...")
        
        if self.model is None or self.scaler is None:
            print("❌ Model not trained. Please train the model first.")
            return None
        
        # Load latest data
        data = self.load_and_prepare_data(ticker)
        if data is None:
            return None
        
        # Get the last 30 days of data for prediction
        last_30_days = data.tail(30)[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Normalize the data
        normalized_data = self.scaler.transform(last_30_days)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(normalized_data).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor)
            predicted_price = prediction[0, -1, 0].cpu().numpy()
        
        # Inverse transform to get actual price
        predicted_price_actual = self.scaler.inverse_transform(
            np.array([[0, 0, 0, predicted_price, 0]])  # Only transform the Close price
        )[0, 3]
        
        return predicted_price_actual
    
    def get_current_price(self, ticker):
        """Get the current stock price."""
        data = self.load_and_prepare_data(ticker)
        if data is not None:
            current_price = data['Close'].iloc[-1]
            return current_price
        return None
    
    def display_prediction(self, ticker, predicted_price, current_price):
        """Display the prediction results."""
        print("\n" + "="*60)
        print(f"📈 {ticker} STOCK PRICE PREDICTION FOR TOMORROW")
        print("="*60)
        
        print(f"📅 Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Current Price: ${current_price:.2f}")
        print(f"🔮 Predicted Price: ${predicted_price:.2f}")
        
        # Calculate change
        change = predicted_price - current_price
        change_percent = (change / current_price) * 100
        
        print(f"📈 Price Change: ${change:.2f} ({change_percent:+.2f}%)")
        
        # Add some analysis
        if change > 0:
            print("📈 Prediction: BULLISH (Price expected to rise)")
        elif change < 0:
            print("📉 Prediction: BEARISH (Price expected to fall)")
        else:
            print("➡️  Prediction: NEUTRAL (Price expected to remain stable)")
        
        print("="*60)
        print("⚠️  DISCLAIMER: This is a machine learning prediction and should not be used as financial advice.")
        print("   Always do your own research and consult with financial professionals.")
        print("="*60)
    
    def run_prediction(self, ticker=None):
        """Run the complete prediction pipeline."""
        if ticker is None:
            ticker = self.select_stock()
            if ticker is None:
                return
        
        self.current_ticker = ticker
        
        print(f"🚀 Starting {ticker} Stock Price Prediction Pipeline")
        print("="*60)
        
        # Step 1: Ensure latest data
        data = self.ensure_latest_data(ticker)
        if data is None:
            print("❌ Failed to get data. Exiting.")
            return
        
        # Step 2: Train model
        self.train_model(data, ticker, epochs=50)
        
        # Step 3: Get current price
        current_price = self.get_current_price(ticker)
        if current_price is None:
            print("❌ Failed to get current price. Exiting.")
            return
        
        # Step 4: Make prediction
        predicted_price = self.predict_tomorrow(ticker)
        if predicted_price is None:
            print("❌ Failed to make prediction. Exiting.")
            return
        
        # Step 5: Display results
        self.display_prediction(ticker, predicted_price, current_price)

def main():
    """Main function."""
    print("📈 Stock Price Predictor for Tomorrow")
    print("Using OHLCV (Open, High, Low, Close, Volume) Data")
    print("="*60)
    
    # Create predictor
    predictor = StockPricePredictor()
    
    # Run prediction
    predictor.run_prediction()

if __name__ == "__main__":
    main() 