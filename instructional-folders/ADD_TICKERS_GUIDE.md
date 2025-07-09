# 📈 How to Add More Tickers to Your Prediction System

## 🚀 **3 Easy Methods to Add Tickers**

### **Method 1: Interactive Data Fetcher (Easiest)**
```bash
python3 fetch_data.py
```
**Then follow prompts:**
- Enter ticker (e.g., GOOGL, AMZN, NVDA)
- Choose time period (default: 2y)
- Data gets saved to `data/TICKER.csv`

### **Method 2: Direct Training (Recommended)**
```bash
python3 train_specific.py TICKER --epochs 50
```

**Popular Examples:**
```bash
# Tech Giants
python3 train_specific.py GOOGL --epochs 50
python3 train_specific.py AMZN --epochs 50
python3 train_specific.py NVDA --epochs 50
python3 train_specific.py META --epochs 50

# Other Popular Stocks
python3 train_specific.py NFLX --epochs 50
python3 train_specific.py AMD --epochs 50
python3 train_specific.py INTC --epochs 50
python3 train_specific.py CRM --epochs 50

# ETFs
python3 train_specific.py QQQ --epochs 50
python3 train_specific.py IWM --epochs 50
python3 train_specific.py VTI --epochs 50
```

### **Method 3: Batch Add Multiple Stocks (Fastest)**
```bash
python3 add_stocks.py
```
**Choose from:**
1. Popular Tech Stocks (10 stocks)
2. ETFs (4 ETFs)
3. Custom list (your choice)

## 📊 **Popular Stock Tickers to Add**

### **Tech Stocks**
- **GOOGL** - Google (Alphabet)
- **AMZN** - Amazon
- **NVDA** - NVIDIA
- **META** - Meta (Facebook)
- **NFLX** - Netflix
- **AMD** - Advanced Micro Devices
- **INTC** - Intel
- **CRM** - Salesforce
- **ADBE** - Adobe
- **PYPL** - PayPal

### **ETFs**
- **QQQ** - Invesco QQQ Trust (Nasdaq-100)
- **IWM** - iShares Russell 2000 ETF
- **VTI** - Vanguard Total Stock Market ETF
- **VOO** - Vanguard S&P 500 ETF

### **Other Popular Stocks**
- **JPM** - JPMorgan Chase
- **JNJ** - Johnson & Johnson
- **PG** - Procter & Gamble
- **KO** - Coca-Cola
- **PEP** - PepsiCo
- **WMT** - Walmart
- **HD** - Home Depot
- **DIS** - Disney

## 🎯 **Step-by-Step Process**

### **Step 1: Add the Stock**
```bash
# Example: Add Google
python3 train_specific.py GOOGL --epochs 50
```

### **Step 2: Verify It's Added**
```bash
# Check your data directory
ls data/
# Should see: GOOGL.csv
```

### **Step 3: Use in Predictions**
```bash
python3 run_prediction.py
# GOOGL will now appear in the selection menu
```

## 📈 **What Happens When You Add a Stock**

1. **Data Fetching**: Gets 2 years of OHLCV data from Yahoo Finance
2. **Data Saving**: Saves to `data/TICKER.csv`
3. **Model Training**: Trains 3-layer DNN for 50 epochs
4. **Metrics Generation**: Creates training plots and performance metrics
5. **System Integration**: Makes stock available for prediction

## ⚡ **Quick Commands Reference**

```bash
# Add single stock
python3 train_specific.py TICKER --epochs 50

# Add multiple stocks (batch)
python3 add_stocks.py

# Interactive data fetching
python3 fetch_data.py

# Make predictions (after adding stocks)
python3 run_prediction.py

# Check current stocks
ls data/
```

## 🔍 **Troubleshooting**

### **"Ticker not found"**
- Check if ticker symbol is correct
- Try different variations (e.g., GOOG vs GOOGL)
- Verify ticker exists on Yahoo Finance

### **"No data found"**
- Check internet connection
- Try a different time period
- Some stocks may have limited data

### **"Training failed"**
- Reduce epochs: `--epochs 30`
- Check available memory
- Try with a different stock first

## 📊 **Current vs. Expanded Portfolio**

### **Currently Available (6 stocks):**
- AAPL, COST, MSFT, PANW, SPY, TSLA

### **After Adding Popular Stocks (16+ stocks):**
- AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- COST, CRM, ADBE, PYPL, AMD, INTC
- SPY, QQQ, IWM, VTI, VOO, PANW

## 🎉 **Benefits of More Stocks**

1. **Diversification**: Predict different sectors
2. **Comparison**: Compare predictions across stocks
3. **Portfolio Analysis**: Predict multiple holdings
4. **Learning**: See how different stocks behave
5. **Flexibility**: Choose from many options

---

**🚀 Ready to expand your prediction system? Start with:**
```bash
python3 add_stocks.py
``` 