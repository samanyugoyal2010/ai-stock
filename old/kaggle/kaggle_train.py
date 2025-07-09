#!/usr/bin/env python3
"""
Kaggle-Optimized NYSE Model Trainer
Optimized for Kaggle with GPU acceleration, higher epochs, and better accuracy.
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

class KaggleStockTrainer:
    """Kaggle-optimized batch trainer for multiple stocks."""
    
    def __init__(self, data_dir="data", models_dir="models", png_dir="png"):
        """
        Initialize the Kaggle trainer.
        
        Args:
            data_dir: Directory containing CSV files
            models_dir: Directory to save trained models
            png_dir: Directory to save training plots
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.png_dir = png_dir
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.png_dir, exist_ok=True)
        
        # Use GPU if available (Kaggle provides free GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_available_stocks(self, max_stocks=None):
        """Get list of available stocks from CSV files."""
        if not os.path.exists(self.data_dir):
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        stocks = [f.replace('.csv', '') for f in csv_files]
        stocks = sorted(stocks)
        
        # Limit stocks for Kaggle (optional)
        if max_stocks:
            stocks = stocks[:max_stocks]
            print(f"📊 Limited to {max_stocks} stocks for Kaggle training")
        
        return stocks
    
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
            if len(data) < 60:  # Increased minimum for better training
                print(f"⚠️  Insufficient data for {ticker} ({len(data)} days). Skipping.")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data for {ticker}: {str(e)}")
            return None
    
    def prepare_data(self, data, sequence_length=60, train_split=0.8):  # Increased sequence length
        """Prepare data for training with better splits."""
        try:
            # Create dataset
            dataset = StockDataset(data, sequence_length=sequence_length)
            
            # Split dataset with more validation data
            total_size = len(dataset)
            train_size = int(train_split * total_size)
            val_size = int(0.15 * total_size)  # Increased validation
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            print(f"❌ Error preparing data for training: {str(e)}")
            return None, None, None
    
    def train_model(self, model, train_loader, val_loader, epochs=100, lr=0.001):  # Increased epochs
        """Train the model with Kaggle optimizations."""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"🎯 Training for {epochs} epochs...")
        
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
            
            # Progress update
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping with model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 20:  # Increased patience for higher epochs
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return train_losses, val_losses, best_val_loss
    
    def evaluate_model(self, model, test_loader, dataset):
        """Evaluate the trained model with comprehensive metrics."""
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
        
        # Calculate comprehensive metrics
        mse = np.mean((predictions_actual - actuals_actual) ** 2)
        mae = np.mean(np.abs(predictions_actual - actuals_actual))
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((actuals_actual - predictions_actual) ** 2)
        ss_tot = np.sum((actuals_actual - np.mean(actuals_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate MAPE
        mape = np.mean(np.abs((actuals_actual - predictions_actual) / actuals_actual)) * 100
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(np.diff(predictions_actual)) == np.sign(np.diff(actuals_actual)))
        direction_accuracy = direction_correct / (len(predictions_actual) - 1) * 100 if len(predictions_actual) > 1 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions_actual,
            'actuals': actuals_actual
        }
    
    def save_training_plots(self, ticker, train_losses, val_losses, metrics, predictions, actuals):
        """Save comprehensive training plots."""
        # Create stock-specific directory
        stock_png_dir = os.path.join(self.png_dir, ticker)
        os.makedirs(stock_png_dir, exist_ok=True)
        
        # Plot 1: Training and validation loss
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'{ticker} - Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Predictions vs Actual
        plt.subplot(2, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.6, color='green')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.title(f'{ticker} - Predictions vs Actual')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        
        # Plot 3: Time series comparison
        plt.subplot(2, 2, 3)
        plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title(f'{ticker} - Price Comparison')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Metrics summary
        plt.subplot(2, 2, 4)
        metrics_text = f"""
        MSE: {metrics['mse']:.4f}
        MAE: {metrics['mae']:.4f}
        RMSE: {metrics['rmse']:.4f}
        R²: {metrics['r2']:.4f}
        MAPE: {metrics['mape']:.2f}%
        Direction Acc: {metrics['direction_accuracy']:.2f}%
        """
        plt.text(0.1, 0.5, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title(f'{ticker} - Model Performance')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{ticker}_kaggle_training_{timestamp}.png"
        plot_path = os.path.join(stock_png_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training plots to: {plot_path}")
    
    def train_single_stock(self, ticker, epochs=100, batch_size=32, sequence_length=60):
        """Train model for a single stock with Kaggle optimizations."""
        print(f"\n🚀 Training {ticker} with Kaggle optimizations...")
        print(f"📈 Epochs: {epochs}, Batch Size: {batch_size}, Sequence Length: {sequence_length}")
        
        # Load data
        data = self.load_stock_data(ticker)
        if data is None:
            return None
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data(data, sequence_length)
        if train_dataset is None:
            return None
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = create_model(input_size=5, hidden_size=256, num_layers=3, dropout=0.3)  # Enhanced model
        
        # Train model
        start_time = time.time()
        train_losses, val_losses, best_val_loss = self.train_model(model, train_loader, val_loader, epochs)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_loader, train_dataset)
        
        # Save model
        model_filename = f"{ticker}_kaggle_model.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'training_time': training_time,
            'epochs_trained': len(train_losses),
            'best_val_loss': best_val_loss
        }, model_path)
        
        # Save plots
        self.save_training_plots(ticker, train_losses, val_losses, metrics, 
                               metrics['predictions'], metrics['actuals'])
        
        # Print results
        print(f"\n✅ {ticker} Training Complete!")
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"📊 Final Validation Loss: {best_val_loss:.6f}")
        print(f"🎯 R² Score: {metrics['r2']:.4f}")
        print(f"📈 Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
        print(f"💾 Model saved to: {model_path}")
        
        return metrics
    
    def train_all_stocks(self, stocks, epochs=100, batch_size=32, sequence_length=60):
        """Train models for all stocks with Kaggle optimizations."""
        print(f"🎯 Starting Kaggle training for {len(stocks)} stocks...")
        print(f"📈 Configuration: {epochs} epochs, batch_size={batch_size}, seq_len={sequence_length}")
        print("="*80)
        
        results = {}
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, ticker in enumerate(stocks, 1):
            print(f"\n{'='*60}")
            print(f"📊 Progress: {i}/{len(stocks)} ({i/len(stocks)*100:.1f}%)")
            print(f"🎯 Training: {ticker}")
            print(f"{'='*60}")
            
            try:
                metrics = self.train_single_stock(ticker, epochs, batch_size, sequence_length)
                if metrics:
                    results[ticker] = metrics
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Error training {ticker}: {str(e)}")
                failed += 1
            
            # Progress update
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(stocks) - i)
                print(f"\n📈 Progress Update:")
                print(f"   ✅ Successful: {successful}")
                print(f"   ❌ Failed: {failed}")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f} minutes")
                print(f"   🕐 Estimated remaining: {remaining/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("🎉 KAGGLE TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Successfully trained: {successful} stocks")
        print(f"❌ Failed: {failed} stocks")
        print(f"⏱️  Total time: {total_time/3600:.1f} hours")
        print(f"📊 Models saved in: {self.models_dir}")
        print(f"📈 Plots saved in: {self.png_dir}")
        
        # Save overall results
        results_file = os.path.join(self.models_dir, "kaggle_training_results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📋 Results saved to: {results_file}")
        
        return results

def main():
    """Main function for Kaggle training."""
    print("🚀 Kaggle-Optimized NYSE Model Trainer")
    print("="*80)
    print("Optimized for Kaggle with GPU acceleration and higher accuracy")
    print("="*80)
    
    # Initialize trainer
    trainer = KaggleStockTrainer()
    
    # Get available stocks
    stocks = trainer.get_available_stocks(max_stocks=50)  # Limit for Kaggle
    
    if not stocks:
        print("❌ No stocks available for training")
        return
    
    print(f"📊 Found {len(stocks)} stocks for training")
    
    # Get training parameters
    print("\n🎯 Training Configuration:")
    epochs = int(input("Enter number of epochs [default: 100]: ") or "100")
    batch_size = int(input("Enter batch size [default: 32]: ") or "32")
    sequence_length = int(input("Enter sequence length [default: 60]: ") or "60")
    
    print(f"\n🚀 Starting Kaggle training with:")
    print(f"   📈 Epochs: {epochs}")
    print(f"   📦 Batch Size: {batch_size}")
    print(f"   📏 Sequence Length: {sequence_length}")
    print(f"   🎮 Device: {trainer.device}")
    
    confirm = input("\nContinue with training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Start training
    results = trainer.train_all_stocks(stocks, epochs, batch_size, sequence_length)
    
    print(f"\n🎉 Training complete! Check the models/ and png/ directories for results.")

if __name__ == "__main__":
    main() 