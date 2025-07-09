# 📈 NYSE Stock Prediction System

A comprehensive 3-script system to add, fetch data for, and train models for all NYSE stocks using a 3-layer Deep Neural Network.

## 🚀 Quick Start

Run these 3 scripts in order:

```bash
# Step 1: Collect all NYSE stock symbols
python3 1_add_all_nyse.py

# Step 2: Fetch data for all stocks
python3 2_fetch_all_data.py

# Step 3: Train models for all stocks
python3 3_train_all_models.py
```

## 📋 Script Details

### **Script 1: `1_add_all_nyse.py`**
- **Purpose**: Collects all NYSE stock symbols
- **Output**: Creates `nyse_symbols.txt` with ~160 stock symbols
- **Time**: ~1 minute
- **What it does**: 
  - Collects major stocks from all sectors (Tech, Financial, Healthcare, etc.)
  - Saves symbols to text file for processing

### **Script 2: `2_fetch_all_data.py`**
- **Purpose**: Fetches OHLCV data for all stocks
- **Input**: Reads from `nyse_symbols.txt`
- **Output**: Creates CSV files in `data/` directory
- **Time**: ~2-3 hours (depending on delay settings)
- **What it does**:
  - Fetches 2 years of historical data from Yahoo Finance
  - Saves OHLCV data for each stock
  - Handles API rate limits with delays

### **Script 3: `3_train_all_models.py`**
- **Purpose**: Trains 3-layer DNN models for all stocks
- **Input**: Uses CSV files from `data/` directory
- **Output**: Trained models ready for prediction
- **Time**: ~4-6 hours (depending on epochs and stock count)
- **What it does**:
  - Trains 3-layer neural network for each stock
  - Uses OHLCV data (Open, High, Low, Close, Volume)
  - Provides performance metrics for each model

## 🏗️ Model Architecture

The system uses a **3-layer Deep Neural Network**:

1. **Layer 1**: LSTM processes raw OHLCV data
2. **Layer 2**: Integrates monthly averages with short-term trends
3. **Layer 3**: Deep reasoning for final price prediction

## 📊 Stock Coverage

The system includes stocks from all major sectors:

- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, CRM
- **Financial**: JPM, BAC, WFC, GS, MS, C, USB, PNC, AXP, BLK
- **Healthcare**: JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, DHR, BMY, AMGN
- **Consumer**: PG, KO, PEP, WMT, HD, DIS, NKE, MCD, SBUX, TGT
- **Industrial**: BA, CAT, GE, MMM, HON, UPS, FDX, RTX, LMT, NOC
- **Energy**: XOM, CVX, COP, EOG, SLB, PSX, VLO, MPC, OXY, KMI
- **Materials**: LIN, APD, FCX, NEM, NUE, AA, DOW, DD, CTVA, BLL
- **Utilities**: NEE, DUK, SO, D, AEP, EXC, XEL, SRE, WEC, DTE
- **Real Estate**: AMT, CCI, PLD, EQIX, DLR, PSA, O, SPG, WELL, VICI
- **ETFs**: SPY, QQQ, IWM, VTI, VOO, VEA, VWO, BND, TLT, GLD

## ⚙️ Configuration Options

### **Script 2 (Data Fetching)**
- **Time Period**: 1y, 2y, 5y, max (default: 2y)
- **Delay**: Seconds between API requests (default: 1s)

### **Script 3 (Training)**
- **Epochs**: Training iterations per stock (default: 30)
- **Batch Size**: Samples per training batch (default: 32)
- **Sequence Length**: Days of historical data (default: 30)

## 📁 Project Structure

```
t-dnns/
├── 1_add_all_nyse.py          # Collect NYSE symbols
├── 2_fetch_all_data.py        # Fetch stock data
├── 3_train_all_models.py      # Train models
├── model.py                   # 3-layer DNN model
├── old/                       # Previous scripts
├── data/                      # Stock data (CSV files)
├── nyse_symbols.txt          # List of stock symbols
└── README.md                 # This file
```

## ⏱️ Time Estimates

| Script | Time | Description |
|--------|------|-------------|
| 1 | ~1 min | Collect symbols |
| 2 | ~2-3 hours | Fetch data (160 stocks × 1s delay) |
| 3 | ~4-6 hours | Train models (160 stocks × 30 epochs) |
| **Total** | **~6-9 hours** | Complete setup |

## 🎯 Usage Example

```bash
# Start the complete process
python3 1_add_all_nyse.py
# Follow prompts, creates nyse_symbols.txt

python3 2_fetch_all_data.py
# Follow prompts, fetches data for all stocks

python3 3_train_all_models.py
# Follow prompts, trains models for all stocks
```

## 📊 Expected Results

After running all 3 scripts, you'll have:

- **~160 stock symbols** in `nyse_symbols.txt`
- **~160 CSV files** in `data/` directory
- **~160 trained models** ready for prediction
- **Performance metrics** for each model (R², MAPE, etc.)

## ⚠️ Important Notes

- **Internet Required**: Script 2 needs internet for data fetching
- **Time Investment**: Total process takes 6-9 hours
- **Storage**: ~50MB for all stock data
- **API Limits**: Script 2 includes delays to respect rate limits
- **Interruption**: You can stop and resume at any time

## 🔧 Troubleshooting

### **"No symbols found"**
- Run script 1 first: `python3 1_add_all_nyse.py`

### **"No data found"**
- Run script 2 first: `python3 2_fetch_all_data.py`
- Check internet connection

### **"Training failed"**
- Reduce epochs in script 3
- Check available memory
- Try with fewer stocks first

---

**🚀 Ready to process all NYSE stocks? Start with:**
```bash
python3 1_add_all_nyse.py
``` 