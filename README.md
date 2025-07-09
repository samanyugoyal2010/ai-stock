# Stock Price Predictor with PyTorch

A sophisticated 3-layer neural network model for predicting stock prices using PyTorch. This project implements a deep learning approach that combines raw stock data analysis with monthly trend integration and deep reasoning capabilities.

## 🏗️ Project Structure

```
t-dnns/
├── data/                   # Stock data CSV files
├── mine/                   # Core model and training code
│   ├── model.py           # 3-layer PyTorch model
│   └── train.py           # Interactive training script
├── fetch_data.py          # Yahoo Finance data fetcher
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 Model Architecture

The stock predictor uses a sophisticated 3-layer architecture:

### Layer 1: Raw Data Learning
- **LSTM Network**: Processes raw OHLCV (Open, High, Low, Close, Volume) data
- **Purpose**: Learns temporal patterns and price movements from historical data
- **Features**: Captures short-term and long-term dependencies in stock prices

### Layer 2: Monthly Trend Integration
- **Monthly Average Computation**: Calculates 20-day moving averages for OHLCV
- **Integration Layer**: Combines LSTM outputs with monthly averages
- **Purpose**: Incorporates medium-term trends and seasonal patterns

### Layer 3: Deep Reasoning
- **Multi-layer Perceptron**: Performs deep analysis on combined features
- **Output**: Predicts the next day's closing price
- **Purpose**: Final decision-making based on all available information

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd t-dnns

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch Stock Data

```bash
# Run the data fetcher
python3 fetch_data.py

# Example usage:
# Enter stock ticker: AAPL
# Enter time period: 2y
```

### 3. Train the Model

```bash
# Run the training script
python3 mine/train.py

# Select from available tickers and configure training parameters
```

## 📊 Features

### Data Management
- **Yahoo Finance Integration**: Automatic data fetching using `yfinance`
- **Flexible Time Periods**: Support for 1y, 2y, 5y, or max historical data
- **CSV Storage**: Organized data storage in `data/` directory

### Model Training
- **Interactive Menu**: User-friendly interface for model selection
- **Configurable Parameters**: Customizable epochs, learning rate, batch size
- **Early Stopping**: Prevents overfitting with validation-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate optimization

### Performance Metrics
- **MSE (Mean Squared Error)**: Overall prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Error in original price units
- **R² Score**: Model's explanatory power
- **MAPE (Mean Absolute Percentage Error)**: Percentage prediction error

### Visualization
- **Training Progress**: Loss curves for training and validation
- **Prediction Analysis**: Scatter plots of predicted vs actual prices
- **Time Series**: Historical vs predicted price trends
- **Residual Analysis**: Error distribution and patterns

## 🔧 Usage Examples

### Fetching Data for Multiple Stocks

```bash
python3 fetch_data.py
# Enter: AAPL
# Enter: 2y
# Enter: TSLA
# Enter: 1y
# Enter: MSFT
# Enter: 5y
```

### Training with Custom Parameters

```bash
python3 mine/train.py
# Select: 1 (AAPL)
# Epochs: 200
# Learning rate: 0.0005
# Batch size: 64
# Sequence length: 50
```

## 📈 Model Performance

The model typically achieves:
- **R² Score**: 0.7-0.9 (depending on stock volatility)
- **MAPE**: 2-8% (percentage prediction error)
- **RMSE**: Varies by stock price range

*Note: Performance varies significantly based on market conditions, stock volatility, and training data quality.*

## 🛠️ Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **yfinance**: Yahoo Finance API wrapper
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Data preprocessing and metrics
- **matplotlib/seaborn**: Visualization

### Model Parameters
- **Input Features**: 5 (OHLCV)
- **Hidden Size**: 128 (configurable)
- **LSTM Layers**: 3 (configurable)
- **Dropout Rate**: 0.2 (configurable)
- **Sequence Length**: 30 days (configurable)

### Training Features
- **Adam Optimizer**: Adaptive learning rate optimization
- **MSE Loss**: Mean squared error loss function
- **ReduceLROnPlateau**: Learning rate scheduling
- **Early Stopping**: Validation-based stopping criterion
- **Data Normalization**: MinMaxScaler for feature scaling

## 📁 File Descriptions

### `fetch_data.py`
- Interactive script for downloading stock data
- Supports multiple time periods
- Automatic CSV file organization
- Error handling and validation

### `mine/model.py`
- `StockPredictor`: Main 3-layer neural network
- `StockDataset`: Custom PyTorch dataset
- Data preprocessing and normalization
- Model creation utilities

### `mine/train.py`
- `StockTrainer`: Complete training pipeline
- Interactive menu system
- Performance evaluation and metrics
- Visualization and model saving

## 🔍 Advanced Usage

### Custom Model Architecture

```python
from mine.model import create_model

# Create custom model
model = create_model(
    input_size=5,
    hidden_size=256,
    num_layers=4,
    dropout=0.3
)
```

### Batch Training Multiple Stocks

```python
import os
from mine.train import StockTrainer

trainer = StockTrainer()
tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']

for ticker in tickers:
    if os.path.exists(f"data/{ticker}.csv"):
        model, metrics = trainer.run_training(ticker, epochs=50)
        print(f"{ticker}: R² = {metrics['R2']:.4f}")
```

## ⚠️ Important Notes

1. **Data Quality**: Model performance depends heavily on data quality and market conditions
2. **Overfitting**: Use validation data and early stopping to prevent overfitting
3. **Market Volatility**: High volatility periods may reduce prediction accuracy
4. **Not Financial Advice**: This is a research tool, not investment advice
5. **GPU Usage**: Training automatically uses GPU if available, falls back to CPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with Yahoo Finance's terms of service when using their data.

## 🆘 Troubleshooting

### Common Issues

1. **No data found**: Run `python3 fetch_data.py` first to download stock data
2. **CUDA out of memory**: Reduce batch size or sequence length
3. **Poor performance**: Try different hyperparameters or more training data
4. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Performance Tips

- Use GPU for faster training (automatically detected)
- Increase sequence length for better temporal modeling
- Experiment with different learning rates and batch sizes
- Use more historical data for better pattern recognition 