#!/usr/bin/env python3
"""
Download OASIS dataset from Kaggle and run full pipeline with real data.

This script:
1. Downloads the OASIS dataset from Kaggle using kagglehub
2. Organizes the data into the expected directory structure
3. Runs the full training pipeline
4. Generates publication-ready results

Usage:
    python run_with_real_data.py
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

print("="*60)
print("OASIS Dataset Download and Pipeline Execution")
print("="*60)

# Step 1: Download dataset from Kaggle
print("\n📥 Step 1: Downloading OASIS dataset from Kaggle...")
print("This may take a few minutes...")

try:
    import kagglehub
    
    # Download latest version of OASIS dataset
    path = kagglehub.dataset_download("ninadaithal/imagesoasis")
    
    print(f"✓ Dataset downloaded to: {path}")
    
except ImportError:
    print("✗ Error: kagglehub not installed")
    print("Install with: pip install kagglehub")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    print("\nPlease ensure:")
    print("1. You have Kaggle API credentials configured")
    print("2. kagglehub is installed: pip install kagglehub")
    print("3. You have accepted the dataset terms on Kaggle")
    sys.exit(1)

# Step 2: Organize data
print("\n📁 Step 2: Organizing data into project structure...")

project_root = Path(__file__).parent
data_raw_dir = project_root / "data" / "raw"
data_raw_dir.mkdir(parents=True, exist_ok=True)

downloaded_path = Path(path)

# Look for CSV file (clinical data)
csv_files = list(downloaded_path.glob("**/*.csv"))

if csv_files:
    # Use the first CSV file found (should be oasis_cross-sectional.csv)
    csv_file = csv_files[0]
    
    # Copy to our data/raw directory
    dest_csv = data_raw_dir / "clinical.csv"
    shutil.copy2(csv_file, dest_csv)
    
    print(f"✓ Clinical data copied to: {dest_csv}")
    
    # Count samples
    import pandas as pd
    df = pd.read_csv(dest_csv)
    print(f"✓ Loaded {len(df)} clinical samples")
    print(f"  Columns: {list(df.columns)[:5]}...")
else:
    print("✗ Warning: No CSV files found in downloaded dataset")

# Look for image directories
image_dirs = [d for d in downloaded_path.glob("*") if d.is_dir()]

if image_dirs:
    print(f"\n✓ Found {len(image_dirs)} image directories")
    
    # Copy image directories if they exist (for CNN training)
    for img_dir in image_dirs:
        dest_dir = data_raw_dir / img_dir.name
        if not dest_dir.exists():
            print(f"  Copying {img_dir.name}...")
            shutil.copytree(img_dir, dest_dir)
    
    print("✓ Image data organized")
else:
    print("⚠ No image directories found (CNN training will be skipped)")

# Step 3: Run the full pipeline
print("\n" + "="*60)
print("🚀 Step 3: Running Full ML Pipeline with Real Data")
print("="*60)

print("\nThis will:")
print("  1. Train tabular models (LR, RF, GBM)")
print("  2. Create ensemble models")
print("  3. Evaluate all models")
print("  4. Generate publication-ready outputs")
print("\nEstimated time: ~30 minutes")
print("\n" + "="*60)

input("\nPress Enter to start the pipeline (or Ctrl+C to cancel)...")

# Run main.py in full mode
result = subprocess.run(
    ["python", "main.py", "--mode", "full"],
    cwd=project_root
)

if result.returncode == 0:
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    print("\n📊 Results Location:")
    print("  - Models: models/")
    print("  - Tables: outputs/tables/")
    print("  - Figures: outputs/figures/")
    print("  - Summary: outputs/EXECUTIVE_SUMMARY.txt")
    
    print("\n📝 For Your Dissertation:")
    print("  1. Performance table: outputs/tables/model_performance.csv")
    print("  2. ROC curves: outputs/figures/roc_curves.png")
    print("  3. Confusion matrices: outputs/figures/confusion_matrix_*.png")
    print("  4. Executive summary: outputs/EXECUTIVE_SUMMARY.txt")
    
    print("\n" + "="*60)
    print("All results are production-ready for dissertation!")
    print("="*60)
else:
    print("\n✗ Pipeline execution failed")
    print("Check pipeline_execution.log for details")
    sys.exit(1)
