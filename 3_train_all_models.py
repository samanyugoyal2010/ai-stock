#!/usr/bin/env python3
"""
NYSE Model Trainer for All Stocks
Trains the 3-layer DNN model for all NYSE stocks with data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our model
from model import StockPredictor, StockDataset, create_model

class BatchStockTrainer:
    """Batch trainer for multiple stocks."""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the batch trainer.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def get_available_stocks(self):
        """Get list of available stocks from CSV files."""
        if not os.path.exists(self.data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        stocks = [f.replace('.csv', '') for f in csv_files]
        return sorted(stocks)
    
    def load_stock_data(self, ticker):
        """Load stock data for a given ticker."""
        filepath = os.path.join(self.data_dir, f"{ticker}.csv")
        
        if not os.path.exists(filepath):
            print(f"❌ Data file not found: {filepath}")
            return None
        
        try:
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            
            # Check if we have enough data
            if len(data) < 30:
                print(f"⚠️  Insufficient data for {ticker} ({len(data)} days). Skipping.")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data for {ticker}: {str(e)}")
            return None
    
    def prepare_data(self, data, sequence_length=30, train_split=0.8):
        """Prepare data for training."""
        try:
            # Create dataset
            dataset = StockDataset(data, sequence_length=sequence_length)
            
            # Split dataset
            total_size = len(dataset)
            train_size = int(train_split * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            print(f"❌ Error preparing data for training: {str(e)}")
            return None, None, None
    
    def train_model(self, model, train_loader, val_loader, epochs=30, lr=0.001):
        """Train the model."""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                last_outputs = outputs[:, -1, 0].unsqueeze(1)
                loss = criterion(last_outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(sequences)
                    last_outputs = outputs[:, -1, 0].unsqueeze(1)
                    loss = criterion(last_outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:  # Early stopping patience
                break
        
        return train_losses, val_losses, best_val_loss
    
    def evaluate_model(self, model, test_loader, dataset):
        """Evaluate the trained model."""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(sequences)
                last_outputs = outputs[:, -1, 0].unsqueeze(1)
                
                # Convert to numpy for evaluation
                pred_np = last_outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                predictions.extend(pred_np.flatten())
                actuals.extend(target_np.flatten())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform to get actual prices
        predictions_actual = dataset.inverse_transform(predictions)
        actuals_actual = dataset.inverse_transform(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions_actual - actuals_actual) ** 2)
        mae = np.mean(np.abs(predictions_actual - actuals_actual))
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((actuals_actual - predictions_actual) ** 2)
        ss_tot = np.sum((actuals_actual - np.mean(actuals_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate MAPE
        mape = np.mean(np.abs((actuals_actual - predictions_actual) / actuals_actual)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions_actual,
            'actuals': actuals_actual
        }
    
    def train_single_stock(self, ticker, epochs=30, batch_size=32, sequence_length=30):
        """Train model for a single stock."""
        print(f"\n🤖 Training model for {ticker}...")
        
        # Load data
        data = self.load_stock_data(ticker)
        if data is None:
            return False, None
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data(data, sequence_length)
        if train_dataset is None:
            return False, None
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = create_model(input_size=5, hidden_size=128, num_layers=3, dropout=0.2)
        
        # Train model
        train_losses, val_losses, best_val_loss = self.train_model(
            model, train_loader, val_loader, epochs=epochs
        )
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_loader, train_dataset.dataset)
        
        # Print results
        print(f"✅ {ticker} training completed!")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        return True, metrics
    
    def train_all_stocks(self, stocks, epochs=30, batch_size=32, sequence_length=30):
        """Train models for all stocks."""
        print(f"🚀 Starting batch training for {len(stocks)} stocks...")
        print(f"Epochs per stock: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {sequence_length}")
        print("="*60)
        
        successful = 0
        failed = 0
        all_metrics = {}
        
        for i, ticker in enumerate(stocks, 1):
            print(f"\n[{i}/{len(stocks)}] Processing {ticker}...")
            
            try:
                success, metrics = self.train_single_stock(
                    ticker, epochs, batch_size, sequence_length
                )
                
                if success:
                    successful += 1
                    all_metrics[ticker] = metrics
                else:
                    failed += 1
                
                # Progress update
                if i % 5 == 0:
                    print(f"\n📈 Progress: {i}/{len(stocks)} ({i/len(stocks)*100:.1f}%)")
                    print(f"✅ Successful: {successful}, ❌ Failed: {failed}")
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user at {ticker}")
                break
            except Exception as e:
                print(f"❌ Unexpected error for {ticker}: {str(e)}")
                failed += 1
        
        return successful, failed, all_metrics

def main():
    """Main function."""
    print("🤖 NYSE Model Trainer for All Stocks")
    print("="*60)
    print("This script trains 3-layer DNN models for all NYSE stocks")
    print("using the data fetched by script 2.")
    print("="*60)
    
    # Initialize trainer
    trainer = BatchStockTrainer()
    
    # Get available stocks
    stocks = trainer.get_available_stocks()
    
    if not stocks:
        print("❌ No stock data found in data/ directory")
        print("Please run: python3 2_fetch_all_data.py")
        return
    
    print(f"📊 Found {len(stocks)} stocks with data")
    
    # Get training parameters
    epochs_input = input("Enter number of training epochs [default: 30]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 30
    
    batch_size_input = input("Enter batch size [default: 32]: ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 32
    
    sequence_length_input = input("Enter sequence length [default: 30]: ").strip()
    sequence_length = int(sequence_length_input) if sequence_length_input.isdigit() else 30
    
    print(f"\n🚀 Starting batch training...")
    print(f"Stocks to train: {len(stocks)}")
    print(f"Epochs per stock: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Estimated time: {len(stocks) * epochs * 0.1:.1f} minutes")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Train all models
    start_time = time.time()
    successful, failed, all_metrics = trainer.train_all_stocks(
        stocks, epochs, batch_size, sequence_length
    )
    end_time = time.time()
    
    # Summary
    print("\n" + "="*60)
    print("🤖 TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Successfully trained: {successful} models")
    print(f"❌ Failed to train: {failed} models")
    print(f"📈 Total processed: {len(stocks)} stocks")
    print(f"⏱️  Time taken: {(end_time - start_time)/60:.1f} minutes")
    
    if successful > 0:
        print(f"\n🎉 Successfully trained {successful} models!")
        print("Your prediction system is now ready!")
        
        # Show some example metrics
        if all_metrics:
            print(f"\n📊 Example model performance:")
            for i, (ticker, metrics) in enumerate(list(all_metrics.items())[:5], 1):
                print(f"   {i}. {ticker}: R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")

if __name__ == "__main__":
    main() 