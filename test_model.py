#!/usr/bin/env python3
"""
Simple test script to verify the model works correctly.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mine.model import StockPredictor, StockDataset, create_model

def test_model():
    """Test the model with a simple forward pass."""
    print("Testing Stock Predictor Model")
    print("=" * 40)
    
    # Check if we have data
    if not os.path.exists("data/AAPL.csv"):
        print("❌ No AAPL.csv found. Please run fetch_data.py first.")
        return False
    
    try:
        # Load data
        data = pd.read_csv("data/AAPL.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"Loaded {len(data)} days of AAPL data")
        
        # Create dataset
        dataset = StockDataset(data, sequence_length=30)
        print(f"Created dataset with {len(dataset)} sequences")
        
        # Create model
        model = create_model(
            input_size=5,
            hidden_size=64,  # Smaller for testing
            num_layers=2,
            dropout=0.1
        )
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        model.eval()
        
        # Get a sample sequence
        sample_sequence, sample_target = dataset[0]
        sample_sequence = sample_sequence.unsqueeze(0)  # Add batch dimension
        
        print(f"Input sequence shape: {sample_sequence.shape}")
        print(f"Target shape: {sample_target.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(sample_sequence)
        
        print(f"Model output shape: {output.shape}")
        print(f"Model output (last prediction): {output[0, -1, 0].item():.6f}")
        print(f"Target: {sample_target.item():.6f}")
        
        # Test with a batch
        batch_size = 4
        batch_sequences = []
        batch_targets = []
        
        for i in range(batch_size):
            seq, target = dataset[i]
            batch_sequences.append(seq)
            batch_targets.append(target)
        
        batch_sequences = torch.stack(batch_sequences)
        batch_targets = torch.stack(batch_targets)
        
        print(f"Batch sequences shape: {batch_sequences.shape}")
        print(f"Batch targets shape: {batch_targets.shape}")
        
        # Forward pass with batch
        with torch.no_grad():
            batch_output = model(batch_sequences)
        
        print(f"Batch output shape: {batch_output.shape}")
        print(f"Batch output: {batch_output.flatten().numpy()}")
        print(f"Batch targets: {batch_targets.flatten().numpy()}")
        
        print("✅ Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model is working correctly! You can now run the training script.")
        print("Run: python3 mine/train.py")
    else:
        print("\n❌ Model test failed. Please check the error messages above.") 