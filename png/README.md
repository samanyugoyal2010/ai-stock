# PNG Organization System

This directory automatically organizes all training plots and visualizations from the stock prediction model.

## 📁 Directory Structure

```
png/
├── AAPL/          # Apple Inc. training plots
├── TSLA/          # Tesla Inc. training plots  
├── MSFT/          # Microsoft Corp. training plots
├── SPY/           # SPDR S&P 500 ETF training plots
├── COST/          # Costco Wholesale Corp. training plots
├── PANW/          # Palo Alto Networks training plots
└── README.md      # This file
```

## 📊 Plot Types

Each ticker folder contains different types of training plots:

- **`{TICKER}_quick_training_{TIMESTAMP}.png`** - Quick training progress plots
- **`{TICKER}_training_results_{TIMESTAMP}.png`** - Full training results with 4-panel analysis
- **`{TICKER}_demo_training_{TIMESTAMP}.png`** - Demo training plots

## 🕒 Timestamp Format

All files include timestamps in the format: `YYYYMMDD_HHMMSS`

Example: `AAPL_training_results_20250708_171039.png`

## 🔧 Automatic Organization

All training scripts now automatically save plots to this organized structure:

- **`quick_train.py`** → `png/{TICKER}/quick_training_{TIMESTAMP}.png`
- **`mine/train.py`** → `png/{TICKER}/training_results_{TIMESTAMP}.png`
- **`example.py`** → `png/{TICKER}/demo_training_{TIMESTAMP}.png`
- **`train_specific.py`** → `png/{TICKER}/training_results_{TIMESTAMP}.png`

## 📈 Plot Contents

### Quick Training Plots
- Training loss vs validation loss over epochs
- Simple line chart showing convergence

### Training Results Plots (4-panel)
1. **Training Progress** - Loss curves over epochs
2. **Predictions vs Actual** - Scatter plot of predictions
3. **Time Series** - Historical vs predicted prices
4. **Residuals** - Error distribution analysis

### Demo Training Plots
- Simplified training progress for demonstration purposes

## 🛠️ Utilities

### View Current Structure
```bash
python3 png_organizer.py
```

### Organize Existing Files
```bash
python3 organize_existing_pngs.py
```

## 📋 Benefits

1. **Automatic Organization** - No manual file management needed
2. **Timestamped Files** - Never lose track of when plots were generated
3. **Stock-Specific Folders** - Easy to find plots for specific stocks
4. **Consistent Naming** - Standardized file naming across all scripts
5. **Easy Comparison** - Compare different training runs for the same stock

## 🎯 Usage

The system works automatically with all training scripts. Just run any training script and plots will be automatically organized:

```bash
# Quick training - saves to png/AAPL/
python3 quick_train.py

# Interactive training - saves to png/{SELECTED_TICKER}/
python3 mine/train.py

# Specific stock training - saves to png/{TICKER}/
python3 train_specific.py TSLA

# Demo - saves to png/AAPL/
python3 example.py
``` 