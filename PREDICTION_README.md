# 🍎 Apple Stock Price Predictor for Tomorrow

This project predicts Apple's stock price for tomorrow using a 3-layer Deep Neural Network (DNN) trained on OHLCV (Open, High, Low, Close, Volume) data.

## 🚀 Quick Start

To predict Apple's stock price for tomorrow, simply run:

```bash
python run_prediction.py
```

This will:
1. ✅ Check and install required dependencies
2. 📊 Fetch the latest Apple stock data
3. 🤖 Train the neural network model
4. 🔮 Predict tomorrow's stock price
5. 📈 Display the results with analysis

## 📋 What You'll Get

The prediction will show you:
- **Current Apple Stock Price**: Today's closing price
- **Predicted Price for Tomorrow**: AI-predicted closing price
- **Price Change**: Dollar amount and percentage change
- **Market Sentiment**: Bullish, Bearish, or Neutral prediction

## 🏗️ How It Works

### Model Architecture (3-Layer DNN)
1. **Layer 1**: LSTM processes raw OHLCV data
2. **Layer 2**: Computes and integrates monthly averages
3. **Layer 3**: Deep reasoning for final price prediction

### Data Processing
- **Input**: Last 30 days of OHLCV data
- **Features**: Open, High, Low, Close, Volume
- **Output**: Predicted closing price for tomorrow

## 📁 Project Structure

```
t-dnns/
├── predict_tomorrow.py      # Main prediction script
├── run_prediction.py        # Easy execution script
├── model.py                 # 3-layer DNN model
├── fetch_data.py           # Data fetching utilities
├── data/                   # Stock data storage
│   └── AAPL.csv           # Apple stock data
├── png/                    # Training plots (organized)
└── requirements.txt        # Dependencies
```

## 🔧 Manual Execution

If you prefer to run the prediction manually:

```bash
# Option 1: Run the main prediction script
python predict_tomorrow.py

# Option 2: Run with specific parameters
python -c "
from predict_tomorrow import ApplePricePredictor
predictor = ApplePricePredictor()
predictor.run_prediction()
"
```

## 📊 Understanding the Results

### Example Output:
```
🍎 APPLE STOCK PRICE PREDICTION FOR TOMORROW
============================================================
📅 Prediction Date: 2024-01-15 14:30:25
📊 Current Price: $185.64
🔮 Predicted Price: $187.23
📈 Price Change: $1.59 (+0.86%)
📈 Prediction: BULLISH (Price expected to rise)
============================================================
```

### Interpretation:
- **Bullish**: Model predicts price will increase
- **Bearish**: Model predicts price will decrease  
- **Neutral**: Model predicts minimal change

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This is a machine learning prediction tool, not financial advice
2. **Past Performance**: Historical data doesn't guarantee future results
3. **Market Volatility**: Stock prices are influenced by many unpredictable factors
4. **Do Your Research**: Always consult with financial professionals before making investment decisions

## 🛠️ Technical Details

### Dependencies
- `torch`: PyTorch for deep learning
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `yfinance`: Stock data fetching
- `scikit-learn`: Data preprocessing
- `matplotlib`: Plotting (optional)

### Model Parameters
- **Input Size**: 5 (OHLCV features)
- **Hidden Size**: 128 neurons
- **Layers**: 3-layer architecture
- **Sequence Length**: 30 days
- **Training Epochs**: 50 (configurable)

## 🔍 Troubleshooting

### Common Issues:

1. **"No data found for ticker AAPL"**
   - Check internet connection
   - Verify Yahoo Finance API is accessible

2. **"Missing required packages"**
   - Run: `pip install -r requirements.txt`
   - Or let `run_prediction.py` install them automatically

3. **"CUDA out of memory"**
   - The model will automatically use CPU if GPU memory is insufficient
   - Reduce batch size in the code if needed

4. **"Data file not found"**
   - Ensure you're in the correct directory
   - Check if `data/AAPL.csv` exists

## 📈 Model Performance

The model uses:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Regularization**: Dropout (20%) to prevent overfitting
- **Early Stopping**: Prevents overtraining

## 🔄 Updating Predictions

To get fresh predictions:
1. Run the script again (it fetches latest data automatically)
2. The model retrains on the most recent data
3. Predictions are based on the latest market conditions

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all files are in the same directory
3. Verify Python 3.7+ is installed
4. Check internet connection for data fetching

---

**Remember**: This tool is for educational and research purposes. Always do your own due diligence before making any investment decisions. 