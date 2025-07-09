# 🚀 Kaggle Real-Time NYSE Stock Predictor Notebook
# Copy this entire code block into a Kaggle notebook

# Install required packages
!pip install yfinance pandas numpy torch matplotlib scikit-learn

# Import required libraries
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
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import requests
import json
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# Define the 3-layer model (copy from model.py)
class StockPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, num_layers=3, dropout=0.3):
        super(StockPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer 1: LSTM for learning from raw stock price data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer 2: Monthly average computation and integration
        self.monthly_avg_layer = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer 3: Deep reasoning for final prediction
        self.reasoning_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 8, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_monthly_averages(self, data, dates=None):
        batch_size, seq_len, features = data.shape
        monthly_averages = torch.zeros(batch_size, seq_len, features)
        
        for b in range(batch_size):
            for i in range(seq_len):
                if i < 20:
                    monthly_averages[b, i] = data[b, i]
                else:
                    monthly_averages[b, i] = data[b, i-20:i].mean(dim=0)
        
        return monthly_averages
    
    def forward(self, x, dates=None):
        batch_size, seq_len, features = x.shape
        
        # Layer 1: LSTM processing
        lstm_out, _ = self.lstm(x)
        
        if lstm_out.shape[1] != seq_len:
            if lstm_out.shape[1] < seq_len:
                padding = torch.zeros(batch_size, seq_len - lstm_out.shape[1], lstm_out.shape[2], device=x.device)
                lstm_out = torch.cat([lstm_out, padding], dim=1)
            else:
                lstm_out = lstm_out[:, :seq_len, :]
        
        # Layer 2: Compute monthly averages
        monthly_avgs = self.compute_monthly_averages(x, dates)
        
        if monthly_avgs.shape[:2] != lstm_out.shape[:2]:
            min_seq = min(monthly_avgs.shape[1], lstm_out.shape[1])
            monthly_avgs = monthly_avgs[:, :min_seq]
            lstm_out = lstm_out[:, :min_seq]
        
        combined = torch.cat([lstm_out, monthly_avgs], dim=2)
        monthly_out = self.monthly_avg_layer(combined)
        
        # Layer 3: Deep reasoning
        predictions = self.reasoning_layer(monthly_out)
        
        return predictions

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length
        
        # Normalize the data
        self.scaler = self._fit_scaler()
        self.normalized_data = self._normalize_data()
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _fit_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        return scaler
    
    def _normalize_data(self):
        return self.scaler.transform(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    
    def _create_sequences(self):
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(self.normalized_data)):
            seq = self.normalized_data[i-self.sequence_length:i]
            sequences.append(seq)
            target = self.normalized_data[i, 3]  # Close price
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target
    
    def inverse_transform(self, predictions):
        dummy = np.zeros((len(predictions), 5))
        dummy[:, 3] = predictions.flatten()
        return self.scaler.inverse_transform(dummy)[:, 3]

def create_model(input_size=5, hidden_size=256, num_layers=3, dropout=0.3):
    return StockPredictor(input_size, hidden_size, num_layers, dropout)

# Kaggle Real-Time Trainer Class
class KaggleRealtimeTrainer:
    def __init__(self, models_dir="models", png_dir="png"):
        self.models_dir = models_dir
        self.png_dir = png_dir
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.png_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_nyse_symbols_realtime(self, max_stocks=50):
        print("📈 Fetching NYSE symbols in real-time...")
        
        # Major stocks by sector
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM"]
        financial_stocks = ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"]
        healthcare_stocks = ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN"]
        consumer_stocks = ["PG", "KO", "PEP", "WMT", "HD", "DIS", "NKE", "MCD", "SBUX", "TGT"]
        industrial_stocks = ["BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "RTX", "LMT", "NOC"]
        
        all_stocks = tech_stocks + financial_stocks + healthcare_stocks + consumer_stocks + industrial_stocks
        all_stocks = sorted(list(set(all_stocks)))
        
        if max_stocks:
            all_stocks = all_stocks[:max_stocks]
        
        print(f"✅ Collected {len(all_stocks)} NYSE symbols")
        return all_stocks
    
    def fetch_stock_data_realtime(self, ticker, period="2y"):
        try:
            print(f"📈 Fetching {ticker} data from Yahoo Finance...")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                print(f"❌ No data found for {ticker}")
                return None
            
            data = data.reset_index()
            
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = list(data.columns)
            
            if all(col in available_columns for col in expected_columns):
                data = data[expected_columns]
            else:
                print(f"⚠️  Some columns missing for {ticker}")
                return None
            
            if len(data) < 60:
                print(f"⚠️  Insufficient data for {ticker} ({len(data)} days)")
                return None
            
            print(f"✅ {ticker}: {len(data)} days of data fetched")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {str(e)}")
            return None
    
    def prepare_data(self, data, sequence_length=60, train_split=0.8):
        try:
            dataset = StockDataset(data, sequence_length=sequence_length)
            
            total_size = len(dataset)
            train_size = int(train_split * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            print(f"❌ Error preparing data: {str(e)}")
            return None, None, None
    
    def train_model(self, model, train_loader, val_loader, epochs=100, lr=0.001):
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"🎯 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
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
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return train_losses, val_losses, best_val_loss
    
    def evaluate_model(self, model, test_loader, dataset):
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(sequences)
                last_outputs = outputs[:, -1, 0].unsqueeze(1)
                
                pred_np = last_outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                predictions.extend(pred_np.flatten())
                actuals.extend(target_np.flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        predictions_actual = dataset.inverse_transform(predictions)
        actuals_actual = dataset.inverse_transform(actuals)
        
        mse = np.mean((predictions_actual - actuals_actual) ** 2)
        mae = np.mean(np.abs(predictions_actual - actuals_actual))
        rmse = np.sqrt(mse)
        
        ss_res = np.sum((actuals_actual - predictions_actual) ** 2)
        ss_tot = np.sum((actuals_actual - np.mean(actuals_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        mape = np.mean(np.abs((actuals_actual - predictions_actual) / actuals_actual)) * 100
        
        direction_correct = np.sum(np.sign(np.diff(predictions_actual)) == np.sign(np.diff(actuals_actual)))
        direction_accuracy = direction_correct / (len(predictions_actual) - 1) * 100 if len(predictions_actual) > 1 else 0
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions_actual, 'actuals': actuals_actual
        }
    
    def save_training_plots(self, ticker, train_losses, val_losses, metrics, predictions, actuals):
        stock_png_dir = os.path.join(self.png_dir, ticker)
        os.makedirs(stock_png_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'{ticker} - Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.6, color='green')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.title(f'{ticker} - Predictions vs Actual')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title(f'{ticker} - Price Comparison')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{ticker}_realtime_training_{timestamp}.png"
        plot_path = os.path.join(stock_png_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training plots to: {plot_path}")
    
    def train_single_stock_realtime(self, ticker, epochs=100, batch_size=32, sequence_length=60, period="2y"):
        print(f"\n🚀 Training {ticker} with real-time data...")
        print(f"📈 Epochs: {epochs}, Batch Size: {batch_size}, Sequence Length: {sequence_length}")
        
        # Fetch data in real-time
        data = self.fetch_stock_data_realtime(ticker, period)
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
        model = create_model(input_size=5, hidden_size=256, num_layers=3, dropout=0.3)
        
        # Train model
        start_time = time.time()
        train_losses, val_losses, best_val_loss = self.train_model(model, train_loader, val_loader, epochs)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_loader, train_dataset)
        
        # Save model
        model_filename = f"{ticker}_realtime_model.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'training_time': training_time,
            'epochs_trained': len(train_losses),
            'best_val_loss': best_val_loss,
            'data_period': period,
            'data_points': len(data)
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
    
    def train_all_stocks_realtime(self, max_stocks=50, epochs=100, batch_size=32, sequence_length=60, period="2y"):
        print(f"🎯 Starting real-time training for up to {max_stocks} stocks...")
        print(f"📈 Configuration: {epochs} epochs, batch_size={batch_size}, seq_len={sequence_length}")
        print(f"📅 Data Period: {period}")
        print("="*80)
        
        # Get symbols in real-time
        stocks = self.get_nyse_symbols_realtime(max_stocks)
        
        if not stocks:
            print("❌ No stocks available for training")
            return {}
        
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
                metrics = self.train_single_stock_realtime(ticker, epochs, batch_size, sequence_length, period)
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
        print("🎉 REAL-TIME TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Successfully trained: {successful} stocks")
        print(f"❌ Failed: {failed} stocks")
        print(f"⏱️  Total time: {total_time/3600:.1f} hours")
        print(f"📊 Models saved in: {self.models_dir}")
        print(f"📈 Plots saved in: {self.png_dir}")
        
        # Save overall results
        results_file = os.path.join(self.models_dir, "realtime_training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📋 Results saved to: {results_file}")
        
        return results

# 🚀 MAIN EXECUTION - Copy everything above this line to Kaggle notebook
print("🚀 Kaggle Real-Time NYSE Model Trainer")
print("="*80)
print("Fetches data from Yahoo Finance in real-time and trains models")
print("="*80)

# Initialize trainer
trainer = KaggleRealtimeTrainer()

# Configuration (modify these values as needed)
MAX_STOCKS = 50      # Number of stocks to train
EPOCHS = 200         # Training epochs (higher = better accuracy)
BATCH_SIZE = 32      # Batch size (32 works well with GPU)
SEQUENCE_LENGTH = 60 # Days of historical data
PERIOD = "2y"        # Data period: "1y", "2y", "5y", "max"

print(f"\n🚀 Starting real-time training with:")
print(f"   📊 Max Stocks: {MAX_STOCKS}")
print(f"   📈 Epochs: {EPOCHS}")
print(f"   📦 Batch Size: {BATCH_SIZE}")
print(f"   📏 Sequence Length: {SEQUENCE_LENGTH}")
print(f"   📅 Data Period: {PERIOD}")
print(f"   🎮 Device: {trainer.device}")

# Start training
results = trainer.train_all_stocks_realtime(
    max_stocks=MAX_STOCKS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    period=PERIOD
)

print(f"\n🎉 Real-time training complete!")
print("📊 Check the models/ and png/ directories for results")
print("💾 Download the models/ folder to use trained models locally") 