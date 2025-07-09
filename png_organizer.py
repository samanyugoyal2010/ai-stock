#!/usr/bin/env python3
"""
PNG Organizer Utility
Automatically organizes training plots into png/ directory structure.
"""

import os
import shutil
from datetime import datetime

def get_project_root():
    """
    Get the project root directory (where the main scripts are located).
    
    Returns:
        str: Absolute path to project root
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in the mine/ directory, go up one level
    if os.path.basename(script_dir) == 'mine':
        return os.path.dirname(script_dir)
    
    # Otherwise, assume we're already in the project root
    return script_dir

def ensure_png_directory(ticker):
    """
    Ensure the PNG directory exists for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        str: Path to the ticker's PNG directory
    """
    # Get project root and create absolute path to png directory
    project_root = get_project_root()
    png_dir = os.path.join(project_root, "png")
    
    # Create main png directory if it doesn't exist
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    
    # Create ticker subdirectory
    ticker_dir = os.path.join(png_dir, ticker.upper())
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)
    
    return ticker_dir

def save_plot_with_timestamp(fig, ticker, plot_type="training", base_dir="png"):
    """
    Save a matplotlib figure to the organized PNG directory with timestamp.
    
    Args:
        fig: Matplotlib figure object
        ticker (str): Stock ticker symbol
        plot_type (str): Type of plot (e.g., "training", "quick_training", "results")
        base_dir (str): Base directory for PNG files
        
    Returns:
        str: Path to the saved PNG file
    """
    # Ensure directory exists
    ticker_dir = ensure_png_directory(ticker)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker.upper()}_{plot_type}_{timestamp}.png"
    filepath = os.path.join(ticker_dir, filename)
    
    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"📊 Plot saved: {filepath}")
    return filepath

def move_existing_png(ticker, old_filename, plot_type="training"):
    """
    Move an existing PNG file to the organized directory structure.
    
    Args:
        ticker (str): Stock ticker symbol
        old_filename (str): Current filename
        plot_type (str): Type of plot
        
    Returns:
        str: New filepath if moved, None if file doesn't exist
    """
    if not os.path.exists(old_filename):
        return None
    
    # Ensure directory exists
    ticker_dir = ensure_png_directory(ticker)
    
    # Create new filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{ticker.upper()}_{plot_type}_{timestamp}.png"
    new_filepath = os.path.join(ticker_dir, new_filename)
    
    # Move the file
    shutil.move(old_filename, new_filepath)
    print(f"📁 Moved {old_filename} to {new_filepath}")
    
    return new_filepath

def get_png_directory_structure():
    """
    Get the current PNG directory structure.
    
    Returns:
        dict: Dictionary with ticker folders and their contents
    """
    structure = {}
    
    # Get project root and create absolute path to png directory
    project_root = get_project_root()
    png_dir = os.path.join(project_root, "png")
    
    if not os.path.exists(png_dir):
        return structure
    
    for item in os.listdir(png_dir):
        item_path = os.path.join(png_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.endswith('.png')]
            structure[item] = files
    
    return structure

def print_png_structure():
    """Print the current PNG directory structure."""
    structure = get_png_directory_structure()
    
    if not structure:
        print("📁 No PNG files organized yet.")
        return
    
    print("📁 PNG Directory Structure:")
    print("=" * 50)
    
    for ticker, files in structure.items():
        print(f"\n📂 {ticker}/")
        for file in sorted(files):
            print(f"  📄 {file}")
    
    print(f"\nTotal tickers: {len(structure)}")
    total_files = sum(len(files) for files in structure.values())
    print(f"Total PNG files: {total_files}")

if __name__ == "__main__":
    # Print current structure
    print_png_structure() 