#!/usr/bin/env python3
"""
Train on a specific stock without interactive prompts.
Usage: python3 train_specific.py TICKER [EPOCHS] [LEARNING_RATE] [BATCH_SIZE]
Example: python3 train_specific.py AAPL 50 0.001 32
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mine.train import StockTrainer

def main():
    parser = argparse.ArgumentParser(description='Train stock prediction model on specific ticker')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length (default: 30)')
    
    args = parser.parse_args()
    
    print(f"Training {args.ticker} with parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print("=" * 50)
    
    # Create trainer
    trainer = StockTrainer()
    
    # Check if ticker exists
    available_tickers = trainer.get_available_tickers()
    if args.ticker.upper() not in available_tickers:
        print(f"❌ Ticker {args.ticker} not found in data directory.")
        print(f"Available tickers: {available_tickers}")
        return False
    
    try:
        # Run training
        model, metrics = trainer.run_training(
            ticker=args.ticker.upper(),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )
        
        print(f"\n🎉 Training completed successfully for {args.ticker}!")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 