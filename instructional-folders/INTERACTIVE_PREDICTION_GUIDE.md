# 📈 Interactive Stock Price Prediction System

## 🎯 What's New

Your prediction system now allows you to:
- **Choose any stock** from your trained stocks
- **See current prices** and data freshness
- **Predict any stock** you've trained
- **Interactive selection** with numbered menu

## 🚀 How to Use

### Step 1: Run the Interactive Predictor
```bash
python3 run_prediction.py
```

### Step 2: Select Your Stock
The system will show you all available trained stocks:

```
📊 Available Trained Stocks:
========================================
 1. AAPL  | $ 210.01 | 501 days | Updated: 2025-07-08
 2. COST  | $ 985.84 | 501 days | Updated: 2025-07-08
 3. MSFT  | $ 496.62 | 9906 days | Updated: 2025-07-08
 4. PANW  | $ 203.99 | 501 days | Updated: 2025-07-08
 5. SPY   | $ 620.34 | 501 days | Updated: 2025-07-08
 6. TSLA  | $ 297.81 | 501 days | Updated: 2025-07-08
========================================

Select stock (1-6) or 'q' to quit: 
```

### Step 3: Get Your Prediction
Enter the number of the stock you want to predict, and the system will:
1. Fetch latest data for that stock
2. Train the 3-layer DNN model
3. Predict tomorrow's price
4. Show detailed analysis

## 📊 Example Output

```
============================================================
📈 AAPL STOCK PRICE PREDICTION FOR TOMORROW
============================================================
📅 Prediction Date: 2025-07-08 17:49:20
📊 Current Price: $210.01
🔮 Predicted Price: $209.89
📈 Price Change: $-0.12 (-0.06%)
📉 Prediction: BEARISH (Price expected to fall)
============================================================
⚠️  DISCLAIMER: This is a machine learning prediction and should not be used as financial advice.
   Always do your own research and consult with financial professionals.
============================================================
```

## 🏗️ Adding New Stocks

To add more stocks to your prediction system:

### Method 1: Train New Stock
```bash
python3 train_specific.py TICKER --epochs 50
```

Examples:
```bash
python3 train_specific.py GOOGL --epochs 50
python3 train_specific.py AMZN --epochs 50
python3 train_specific.py NVDA --epochs 50
```

### Method 2: Fetch Data First
```bash
python3 fetch_data.py
# Then follow prompts to add new ticker
```

## 📈 Available Stocks

Currently trained stocks:
- **AAPL** (Apple) - $210.01
- **COST** (Costco) - $985.84
- **MSFT** (Microsoft) - $496.62
- **PANW** (Palo Alto Networks) - $203.99
- **SPY** (S&P 500 ETF) - $620.34
- **TSLA** (Tesla) - $297.81

## 🔧 Advanced Usage

### Direct Prediction (No Menu)
```bash
python3 predict_tomorrow.py
# Then select from the interactive menu
```

### Programmatic Usage
```python
from predict_tomorrow import StockPricePredictor

predictor = StockPricePredictor()
predictor.run_prediction("AAPL")  # Direct stock selection
```

## 📊 Understanding the Display

The stock list shows:
- **Number**: Selection number
- **Ticker**: Stock symbol
- **Current Price**: Latest closing price
- **Days of Data**: How much historical data is available
- **Updated**: When the data was last refreshed

## ⚠️ Important Notes

### Only Trained Stocks
- You can only predict stocks that have been trained
- Training creates the necessary data files
- Untrained stocks won't appear in the selection menu

### Data Freshness
- The system automatically fetches the latest data
- Shows when data was last updated
- Ensures predictions use current market information

### Model Training
- Each prediction retrains the model on fresh data
- Takes 2-5 minutes per prediction
- Ensures accuracy with latest market conditions

## 🎯 Quick Commands

```bash
# Interactive prediction (recommended)
python3 run_prediction.py

# Train new stock
python3 train_specific.py TICKER --epochs 50

# Direct prediction
python3 predict_tomorrow.py

# Fetch new data
python3 fetch_data.py
```

## 📞 Troubleshooting

### "No trained stocks found"
- Train some stocks first: `python3 train_specific.py AAPL`
- Check if `data/` directory exists
- Verify CSV files are present

### "Stock not in selection"
- Train the stock first
- Check if data file exists in `data/TICKER.csv`
- Ensure the ticker symbol is correct

### "Prediction failed"
- Check internet connection for data fetching
- Verify all dependencies are installed
- Try running with a different stock

---

**🎯 You now have a fully interactive stock prediction system! Choose any trained stock and get tomorrow's price prediction!** 