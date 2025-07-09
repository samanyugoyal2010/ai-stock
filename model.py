#!/usr/bin/env python3
"""
3-Layer Stock Prediction Model
A PyTorch-based neural network for predicting stock prices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class StockPredictor(nn.Module):
    """
    3-Layer Stock Prediction Model
    
    Layer 1: Learns from raw stock price data (OHLCV)
    Layer 2: Computes and integrates monthly OHLCV averages
    Layer 3: Performs deep reasoning to predict next day's closing price
    """
    
    def __init__(self, 
                 input_size: int = 5,  # OHLCV features
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        """
        Initialize the StockPredictor model.
        
        Args:
            input_size: Number of input features (OHLCV = 5)
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
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
            nn.Linear(hidden_size + 5, hidden_size),  # +5 for monthly OHLCV averages
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
            nn.Linear(hidden_size // 8, 1)  # Predict next day's closing price
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_monthly_averages(self, data: torch.Tensor, dates: pd.DatetimeIndex) -> torch.Tensor:
        """
        Compute monthly OHLCV averages from the input data.
        
        Args:
            data: Input tensor of shape (batch_size, seq_len, features)
            dates: Corresponding dates for the data
            
        Returns:
            Monthly averages tensor
        """
        batch_size, seq_len, features = data.shape
        monthly_averages = torch.zeros(batch_size, seq_len, features)
        
        for b in range(batch_size):
            for i in range(seq_len):
                if i < 20:  # Need at least 20 days for monthly average
                    monthly_averages[b, i] = data[b, i]  # Use current value
                else:
                    # Compute average of last 20 trading days (approximately one month)
                    monthly_averages[b, i] = data[b, i-20:i].mean(dim=0)
        
        return monthly_averages
    
    def forward(self, x: torch.Tensor, dates: Optional[pd.DatetimeIndex] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            dates: Optional dates for computing monthly averages
            
        Returns:
            Predicted closing prices of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, features = x.shape
        
        # Layer 1: LSTM processing of raw stock data
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Ensure LSTM output has the same sequence length as input
        if lstm_out.shape[1] != seq_len:
            # Pad or truncate LSTM output to match input sequence length
            if lstm_out.shape[1] < seq_len:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len - lstm_out.shape[1], lstm_out.shape[2], device=x.device)
                lstm_out = torch.cat([lstm_out, padding], dim=1)
            else:
                # Truncate
                lstm_out = lstm_out[:, :seq_len, :]
        
        # Layer 2: Compute monthly averages and integrate
        if dates is not None:
            monthly_avgs = self.compute_monthly_averages(x, dates)
        else:
            # If no dates provided, use simple moving average
            monthly_avgs = torch.zeros_like(x)
            window = 20
            for i in range(seq_len):
                if i < window:
                    monthly_avgs[:, i] = x[:, i]
                else:
                    # mean over the window for each batch, keep shape
                    monthly_avgs[:, i] = x[:, i-window:i].mean(dim=1)
        
        # Ensure monthly_avgs and lstm_out have the same batch and seq_len dimensions
        if monthly_avgs.shape[:2] != lstm_out.shape[:2]:
            min_seq = min(monthly_avgs.shape[1], lstm_out.shape[1])
            monthly_avgs = monthly_avgs[:, :min_seq]
            lstm_out = lstm_out[:, :min_seq]
        
        # Debug: Print shapes before concatenation
        # print(f"LSTM output shape: {lstm_out.shape}")
        # print(f"Monthly averages shape: {monthly_avgs.shape}")
        
        # Concatenate LSTM output with monthly averages
        combined = torch.cat([lstm_out, monthly_avgs], dim=2)
        
        # Process through monthly average layer
        monthly_out = self.monthly_avg_layer(combined)
        
        # Layer 3: Deep reasoning for final prediction
        predictions = self.reasoning_layer(monthly_out)
        
        return predictions

class StockDataset(torch.utils.data.Dataset):
    """
    Custom dataset for stock data.
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 30):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with OHLCV columns
            sequence_length: Length of input sequences
        """
        self.data = data
        self.sequence_length = sequence_length
        
        # Normalize the data
        self.scaler = self._fit_scaler()
        self.normalized_data = self._normalize_data()
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _fit_scaler(self):
        """Fit a scaler to normalize the data."""
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        # Fit on OHLCV columns
        scaler.fit(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        return scaler
    
    def _normalize_data(self) -> np.ndarray:
        """Normalize the data using the fitted scaler."""
        return self.scaler.transform(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and target values."""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(self.normalized_data)):
            # Input sequence
            seq = self.normalized_data[i-self.sequence_length:i]
            sequences.append(seq)
            
            # Target: next day's closing price
            target = self.normalized_data[i, 3]  # Close price is at index 3
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target
    
    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        # Create dummy array with same shape as original data
        dummy = np.zeros((len(predictions), 5))
        dummy[:, 3] = predictions.flatten()  # Put predictions in Close column
        return self.scaler.inverse_transform(dummy)[:, 3]

def create_model(input_size: int = 5, 
                hidden_size: int = 128, 
                num_layers: int = 3, 
                dropout: float = 0.2) -> StockPredictor:
    """
    Create and return a StockPredictor model.
    
    Args:
        input_size: Number of input features
        hidden_size: Size of hidden layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        
    Returns:
        Initialized StockPredictor model
    """
    model = StockPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    return model 