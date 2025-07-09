#!/usr/bin/env python3
"""
Quick Training Script
Runs training without interactive prompts for easy testing.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mine.model import StockPredictor, StockDataset, create_model
from png_organizer import save_plot_with_timestamp

def quick_train(ticker="AAPL", epochs=10, lr=0.001, batch_size=16, sequence_length=30):
    """
    Quick training function without interactive prompts.
    
    Args:
        ticker: Stock ticker to train on
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        sequence_length: Length of input sequences
    """
    print(f"Quick Training for {ticker}")
    print("=" * 50)
    
    # Check if data exists
    data_file = f"data/{ticker}.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run fetch_data.py first to download stock data.")
        return False
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(data)} days of data for {ticker}")
        print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
        
        # Create dataset
        dataset = StockDataset(data, sequence_length=sequence_length)
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = create_model(
            input_size=5,
            hidden_size=64,  # Smaller for quick training
            num_layers=2,
            dropout=0.1
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Using device: {device}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                # Use the last prediction from each sequence
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
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(sequences)
                    last_outputs = outputs[:, -1, 0].unsqueeze(1)
                    loss = criterion(last_outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        print("Training completed!")
        
        # Evaluate on test set
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                last_outputs = outputs[:, -1, 0].unsqueeze(1)
                
                predictions.extend(last_outputs.cpu().numpy().flatten())
                actuals.extend(targets.numpy().flatten())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform to original scale
        predictions_original = dataset.inverse_transform(predictions)
        actuals_original = dataset.inverse_transform(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals_original, predictions_original)
        mae = mean_absolute_error(actuals_original, predictions_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals_original, predictions_original)
        mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS FOR {ticker}")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"{'='*50}")
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'Training Progress for {ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot to organized directory
        plot_path = save_plot_with_timestamp(plt.gcf(), ticker, "quick_training")
        plt.show()
        
        # Save model
        model_path = f"{ticker}_quick_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Quick training with default parameters
    success = quick_train(
        ticker="AAPL",
        epochs=10,
        lr=0.001,
        batch_size=16,
        sequence_length=30
    )
    
    if success:
        print("\n🎉 Quick training completed successfully!")
        print("You can now run the full interactive training with: python3 mine/train.py")
    else:
        print("\n❌ Quick training failed. Please check the error messages above.") 