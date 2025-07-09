"""
Stock Prediction Model Package
A PyTorch-based neural network for predicting stock prices.
"""

from .model import StockPredictor, StockDataset, create_model
from .train import StockTrainer

__version__ = "1.0.0"
__author__ = "Stock Predictor Team"

__all__ = [
    "StockPredictor",
    "StockDataset", 
    "create_model",
    "StockTrainer"
] 