# 🚀 Kaggle Setup Guide for NYSE Stock Predictor

This guide will help you run the NYSE stock predictor on Kaggle with GPU acceleration for faster training and better accuracy.

## 📋 Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Basic Python Knowledge**: Familiarity with Python and machine learning concepts
3. **Patience**: Training multiple stocks can take several hours

## 🎯 Two Training Options

### Option 1: Real-Time Training (Recommended) 🆕
**Use `kaggle_notebook_template.py`** - Fetches data directly from Yahoo Finance in real-time

**Advantages:**
- ✅ No need to upload data files
- ✅ Always uses the latest stock data
- ✅ Works immediately without setup
- ✅ Automatically handles data fetching errors
- ✅ More stocks available (up to 50 major stocks)

### Option 2: Local Data Training
**Use `kaggle_train.py`** - Uses your local CSV data files

**Advantages:**
- ✅ Faster training (no API calls)
- ✅ Works offline
- ✅ Consistent data across runs
- ✅ Can train on any stocks you have data for

## 🚀 Quick Start: Real-Time Training (Recommended)

### Step 1: Create New Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com)
2. Click "Create" → "New Notebook"
3. Choose "Python" as the language
4. Enable GPU: Click "Settings" → "Accelerator" → "GPU"

### Step 2: Copy the Real-Time Code
1. Open `kaggle_notebook_template.py` from your local project
2. Copy the **entire content** (everything from the first line to the last line)
3. Paste it into your Kaggle notebook
4. The code will automatically:
   - Install required packages
   - Define the 3-layer model
   - Set up the real-time trainer
   - Start training on 50 major stocks

### Step 3: Configure Training Parameters
Modify these values in the code as needed:

```python
MAX_STOCKS = 50      # Number of stocks to train (1-50)
EPOCHS = 200         # Training epochs (100-500 recommended)
BATCH_SIZE = 32      # Batch size (32 works well with GPU)
SEQUENCE_LENGTH = 60 # Days of historical data
PERIOD = "2y"        # Data period: "1y", "2y", "5y", "max"
```

### Step 4: Run the Notebook
1. Click "Run All" or press `Ctrl+Enter` to run each cell
2. The system will:
   - Detect your GPU and show specs
   - Fetch stock symbols in real-time
   - Download data from Yahoo Finance for each stock
   - Train models with GPU acceleration
   - Save models and plots automatically

## 📊 Expected Results

### Training Time Estimates
- **1 stock**: ~10-15 minutes
- **10 stocks**: ~2-3 hours
- **50 stocks**: ~8-12 hours

### Output Files
After training completes, you'll have:

```
📁 models/
├── AAPL_realtime_model.pth
├── MSFT_realtime_model.pth
├── GOOGL_realtime_model.pth
├── ... (one file per stock)
└── realtime_training_results.json

📁 png/
├── AAPL/
│   └── AAPL_realtime_training_20241201_143022.png
├── MSFT/
│   └── MSFT_realtime_training_20241201_143045.png
└── ... (one folder per stock)
```

### Model Performance
With 200 epochs and GPU acceleration, expect:
- **R² Score**: 0.70-0.90 (higher is better)
- **Direction Accuracy**: 55-75% (predicting up/down movement)
- **MAPE**: 2-8% (mean absolute percentage error)

## 🔧 Alternative: Local Data Training

If you prefer to use your local data files:

### Step 1: Upload Data
1. In your Kaggle notebook, click "Add data"
2. Upload your `data/` folder with CSV files
3. Or upload individual CSV files

### Step 2: Use Local Training Script
1. Copy the content of `kaggle_train.py` to your notebook
2. Modify the stock list to match your CSV files
3. Run the training

## ⚙️ Advanced Configuration

### GPU Optimization
The code automatically detects and uses GPU. For best performance:

```python
# In the trainer initialization
trainer = KaggleRealtimeTrainer()

# Check GPU info
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Memory Management
If you run out of GPU memory:

```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Reduce hidden size
model = create_model(hidden_size=128)  # Instead of 256

# Reduce sequence length
SEQUENCE_LENGTH = 30  # Instead of 60
```

### Training Parameters
For different accuracy vs speed trade-offs:

```python
# Fast training (lower accuracy)
EPOCHS = 50
BATCH_SIZE = 64
MAX_STOCKS = 10

# High accuracy (slower)
EPOCHS = 500
BATCH_SIZE = 16
MAX_STOCKS = 25
```

## 📈 Monitoring Training

### Real-Time Progress
The system shows:
- ✅ Current stock being trained
- 📊 Progress percentage
- ⏱️ Elapsed time
- 🕐 Estimated remaining time
- 📈 Training/validation loss

### Early Stopping
Models automatically stop training when:
- Validation loss stops improving for 20 epochs
- Maximum epochs reached
- Best model state is saved

## 🚨 Troubleshooting

### Common Issues

**1. "No data found for [TICKER]"**
- Some stocks may be delisted or have data issues
- The system automatically skips these and continues

**2. "CUDA out of memory"**
- Reduce `BATCH_SIZE` to 16 or 8
- Reduce `MAX_STOCKS` to train fewer stocks
- Reduce `HIDDEN_SIZE` in model creation

**3. "Connection timeout"**
- Yahoo Finance API may be slow
- The system retries automatically
- Consider using shorter `PERIOD` like "1y"

**4. "Package not found"**
- The code automatically installs required packages
- If issues persist, manually run:
```python
!pip install yfinance pandas numpy torch matplotlib scikit-learn
```

### Performance Tips

**For Faster Training:**
- Use fewer stocks (`MAX_STOCKS = 10`)
- Use fewer epochs (`EPOCHS = 100`)
- Use larger batch size (`BATCH_SIZE = 64`)
- Use shorter data period (`PERIOD = "1y"`)

**For Better Accuracy:**
- Use more stocks (`MAX_STOCKS = 50`)
- Use more epochs (`EPOCHS = 300-500`)
- Use smaller batch size (`BATCH_SIZE = 16`)
- Use longer data period (`PERIOD = "5y"`)

## 📥 Downloading Results

### After Training Completes
1. **Download Models**: Click "Output" → "models/" → Download
2. **Download Plots**: Click "Output" → "png/" → Download
3. **Download Results**: Click "Output" → "realtime_training_results.json" → Download

### Using Models Locally
```python
# Load a trained model
import torch
from model import create_model

# Load the model
checkpoint = torch.load('AAPL_realtime_model.pth', map_location='cpu')
model = create_model()
model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"R² Score: {checkpoint['metrics']['r2']:.4f}")
print(f"Direction Accuracy: {checkpoint['metrics']['direction_accuracy']:.2f}%")
```

## 🎯 Next Steps

1. **Analyze Results**: Check the training plots for each stock
2. **Compare Performance**: Look at the `realtime_training_results.json` file
3. **Download Models**: Save the trained models for local use
4. **Make Predictions**: Use the models to predict future stock prices

## 💡 Pro Tips

1. **Start Small**: Begin with 5-10 stocks to test the system
2. **Monitor GPU**: Watch GPU memory usage in Kaggle's resource monitor
3. **Save Progress**: Download results periodically during long training runs
4. **Experiment**: Try different parameters to find optimal settings
5. **Backup**: Always download your trained models before the session ends

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Look at the error messages in the notebook output
3. Try reducing the number of stocks or epochs
4. Ensure GPU is enabled in notebook settings

---

**Happy Training! 🚀📈**

The real-time training option is the easiest way to get started and will automatically fetch the latest data for you. 