#!/usr/bin/env python3
"""
Stock Prediction Model Trainer
Interactive training script for the 3-layer stock prediction model.
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
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import model from current directory
from model import StockPredictor, StockDataset, create_model
from png_organizer import save_plot_with_timestamp

class StockTrainer:
    """Trainer class for the stock prediction model."""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing CSV files
        """
        # Make data_dir absolute relative to project root
        if not os.path.isabs(data_dir):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Use the script directory as project root
            self.data_dir = os.path.join(script_dir, data_dir)
        else:
            self.data_dir = data_dir
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Data directory: {self.data_dir}")
    
    def get_available_tickers(self):
        """Get list of available stock tickers from CSV files."""
        if not os.path.exists(self.data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        return sorted(tickers)
    
    def load_data(self, ticker):
        """
        Load stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            pandas.DataFrame: Stock data
        """
        filepath = os.path.join(self.data_dir, f"{ticker}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(data)} days of data for {ticker}")
        print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
        
        return data
    
    def prepare_data(self, data, sequence_length=30, train_split=0.8):
        """
        Prepare data for training.
        
        Args:
            data: Stock data DataFrame
            sequence_length: Length of input sequences
            train_split: Fraction of data to use for training
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
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
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, model, train_loader, val_loader, epochs=100, lr=0.001):
        """
        Train the model.
        
        Args:
            model: StockPredictor model
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            tuple: (train_losses, val_losses)
        """
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
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
            
            if patience_counter >= 20:  # Early stopping patience
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        print("Training completed!")
        return train_losses, val_losses
    
    def evaluate_model(self, model, test_loader, dataset):
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            dataset: Original dataset for inverse transformation
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
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
        
        # Calculate percentage error
        mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, predictions_original, actuals_original
    
    def plot_results(self, train_losses, val_losses, predictions, actuals, ticker):
        """Plot training results and predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(train_losses, label='Train Loss')
        axes[0, 0].plot(val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Predictions vs Actual
        axes[0, 1].scatter(actuals, predictions, alpha=0.6)
        axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title('Predictions vs Actual')
        axes[0, 1].grid(True)
        
        # Time series plot
        axes[1, 0].plot(actuals, label='Actual', alpha=0.7)
        axes[1, 0].plot(predictions, label='Predicted', alpha=0.7)
        axes[1, 0].set_title('Stock Price Predictions')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Residuals
        residuals = actuals - predictions
        axes[1, 1].scatter(predictions, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot to organized directory
        plot_path = save_plot_with_timestamp(plt.gcf(), ticker, "training_results")
        plt.show()
    
    def run_training(self, ticker, epochs=100, lr=0.001, batch_size=32, sequence_length=30):
        """
        Run the complete training pipeline.
        
        Args:
            ticker: Stock ticker to train on
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            sequence_length: Length of input sequences
        """
        print(f"\n{'='*60}")
        print(f"Training Stock Prediction Model for {ticker}")
        print(f"{'='*60}")
        
        # Load data
        data = self.load_data(ticker)
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data(data, sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = create_model()
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        train_losses, val_losses = self.train_model(model, train_loader, val_loader, epochs, lr)
        
        # Evaluate model
        metrics, predictions, actuals = self.evaluate_model(model, test_loader, train_dataset.dataset)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS FOR {ticker}")
        print(f"{'='*60}")
        print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"R-squared (R²): {metrics['R2']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
        print(f"{'='*60}")
        
        # Plot results
        self.plot_results(train_losses, val_losses, predictions, actuals, ticker)
        
        # Save model
        model_path = f"{ticker}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return model, metrics

def main():
    """Main function with interactive menu."""
    print("Stock Prediction Model Trainer")
    print("=" * 50)
    
    trainer = StockTrainer()
    
    while True:
        # Get available tickers
        tickers = trainer.get_available_tickers()
        
        if not tickers:
            print("\n❌ No stock data found in the 'data' directory.")
            print("Please run fetch_data.py first to download stock data.")
            break
        
        print(f"\nAvailable stock tickers ({len(tickers)}):")
        for i, ticker in enumerate(tickers, 1):
            print(f"  {i}. {ticker}")
        
        print(f"  {len(tickers) + 1}. Exit")
        
        # Get user selection
        try:
            choice = input(f"\nSelect a ticker to train on (1-{len(tickers) + 1}): ").strip()
            
            if choice == str(len(tickers) + 1) or choice.lower() in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tickers):
                selected_ticker = tickers[choice_idx]
                
                # Get training parameters
                print(f"\nTraining parameters for {selected_ticker}:")
                epochs = input("Number of epochs [default: 100]: ").strip()
                epochs = int(epochs) if epochs else 100
                
                lr = input("Learning rate [default: 0.001]: ").strip()
                lr = float(lr) if lr else 0.001
                
                batch_size = input("Batch size [default: 32]: ").strip()
                batch_size = int(batch_size) if batch_size else 32
                
                sequence_length = input("Sequence length [default: 30]: ").strip()
                sequence_length = int(sequence_length) if sequence_length else 30
                
                # Run training
                try:
                    model, metrics = trainer.run_training(
                        selected_ticker, 
                        epochs=epochs, 
                        lr=lr, 
                        batch_size=batch_size,
                        sequence_length=sequence_length
                    )
                    
                    # Ask if user wants to continue
                    continue_training = input("\nTrain another model? (y/n): ").strip().lower()
                    if continue_training not in ['y', 'yes']:
                        print("Goodbye!")
                        break
                        
                except Exception as e:
                    print(f"❌ Error during training: {str(e)}")
                    continue
            else:
                print("❌ Invalid selection. Please try again.")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 