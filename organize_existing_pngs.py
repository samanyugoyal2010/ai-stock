#!/usr/bin/env python3
"""
Organize existing PNG files into the png/ directory structure.
This script will find any PNG files in the root directory and move them to the appropriate ticker folders.
"""

import os
import glob
from png_organizer import move_existing_png

def organize_existing_pngs():
    """Find and organize any PNG files in the root directory."""
    print("🔍 Searching for PNG files to organize...")
    
    # Find all PNG files in the root directory
    png_files = glob.glob("*.png")
    
    if not png_files:
        print("✅ No PNG files found in root directory.")
        return
    
    print(f"📁 Found {len(png_files)} PNG files to organize:")
    
    organized_count = 0
    
    for png_file in png_files:
        print(f"\n📄 Processing: {png_file}")
        
        # Extract ticker from filename
        filename = os.path.basename(png_file)
        parts = filename.split('_')
        
        if len(parts) >= 2:
            ticker = parts[0].upper()
            plot_type = "_".join(parts[1:]).replace('.png', '')
            
            print(f"  Ticker: {ticker}")
            print(f"  Plot type: {plot_type}")
            
            # Move the file
            new_path = move_existing_png(ticker, png_file, plot_type)
            if new_path:
                organized_count += 1
                print(f"  ✅ Organized: {new_path}")
            else:
                print(f"  ❌ Failed to organize: {png_file}")
        else:
            print(f"  ⚠️  Could not parse filename: {png_file}")
    
    print(f"\n🎉 Organization complete!")
    print(f"📊 Organized {organized_count} out of {len(png_files)} files.")

if __name__ == "__main__":
    organize_existing_pngs()
    
    # Show final structure
    print("\n" + "="*50)
    from png_organizer import print_png_structure
    print_png_structure() 