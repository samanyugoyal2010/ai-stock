#!/usr/bin/env python3
"""
Example Usage of Stock Predictor
Demonstrates how to use the stock prediction model.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mine.model import StockPredictor, StockDataset, create_model
from png_organizer import save_plot_with_timestamp
from mine.train import StockTrainer

def example_usage():
    """Example of how to use the stock predictor."""
    
    print("Stock Predictor Example")
    print("=" * 50)
    
    # Check if we have any data
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ No data directory found. Please run fetch_data.py first.")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("❌ No CSV files found in data directory. Please run fetch_data.py first.")
        return
    
    # Use the first available ticker
    ticker = csv_files[0].replace('.csv', '')
    print(f"Using ticker: {ticker}")
    
    # Create trainer
    trainer = StockTrainer()
    
    try:
        # Load data
        data = trainer.load_data(ticker)
        print(f"Loaded {len(data)} days of data")
        
        # Create a simple model for demonstration
        model = create_model(
            input_size=5,
            hidden_size=64,  # Smaller for faster demo
            num_layers=2,
            dropout=0.1
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Prepare data with shorter sequence for demo
        train_dataset, val_dataset, test_dataset = trainer.prepare_data(
            data, 
            sequence_length=20,  # Shorter for demo
            train_split=0.7
        )
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Quick training demo (fewer epochs)
        print("\nStarting quick training demo (10 epochs)...")
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train for just a few epochs
        train_losses, val_losses = trainer.train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=10,  # Quick demo
            lr=0.001
        )
        
        print("Demo training completed!")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        
        # Show a simple prediction
        print("\nMaking a sample prediction...")
        model.eval()
        
        # Get a sample from test set
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        sample_sequence, sample_target = next(iter(test_loader))
        
        with torch.no_grad():
            prediction = model(sample_sequence)
            
        # Convert to original scale
        original_pred = train_dataset.dataset.inverse_transform(prediction.numpy())
        original_target = train_dataset.dataset.inverse_transform(sample_target.numpy())
        
        print(f"Predicted price: ${original_pred[0]:.2f}")
        print(f"Actual price: ${original_target[0]:.2f}")
        print(f"Error: ${abs(original_pred[0] - original_target[0]):.2f}")
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'Training Progress for {ticker} (Demo)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot to organized directory
        plot_path = save_plot_with_timestamp(plt.gcf(), ticker, "demo_training")
        plt.show()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 For full training, run: python3 mine/train.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("Make sure you have stock data in the data/ directory.")

def quick_data_fetch():
    """Quick function to fetch some sample data if none exists."""
    print("\nQuick Data Fetch Demo")
    print("=" * 30)
    
    try:
        import yfinance as yf
        
        # Fetch a small amount of data for demo
        print("Fetching sample data for AAPL...")
        stock = yf.Ticker("AAPL")
        data = stock.history(period="6mo")  # 6 months of data
        
        if not data.empty:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Save data
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data.to_csv("data/AAPL.csv", index=False)
            
            print("✅ Sample data saved to data/AAPL.csv")
            print("You can now run the example or fetch more data with fetch_data.py")
        else:
            print("❌ Failed to fetch data")
            
    except ImportError:
        print("❌ yfinance not installed. Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")

if __name__ == "__main__":
    # Check if we need to fetch sample data
    if not os.path.exists("data") or not any(f.endswith('.csv') for f in os.listdir("data") if os.path.exists("data")):
        print("No stock data found. Fetching sample data...")
        quick_data_fetch()
    
    # Run the example
    example_usage() 