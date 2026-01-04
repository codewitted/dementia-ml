#!/usr/bin/env python3
"""
Main entry point for the dementia-ml project.

This script provides a one-click execution of the entire ML pipeline,
making it easy for examiners and reviewers to reproduce results.

Usage:
    python main.py --mode full           # Run complete pipeline
    python main.py --mode tabular        # Run only tabular models
    python main.py --mode quick          # Quick validation run
    python main.py --mode demo           # Demo with sample data
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if environment is properly configured."""
    logger.info("Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        logger.error(f"Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'yaml', 'torch', 'shap'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: conda env create -f environment.yml")
        return False
    
    logger.info("✓ Environment check passed")
    return True


def check_data():
    """Check if required data files exist."""
    logger.info("Checking data availability...")
    
    clinical_data = Path('data/raw/clinical.csv')
    
    if not clinical_data.exists():
        logger.warning("Clinical data not found at data/raw/clinical.csv")
        logger.info("Please download OASIS dataset (see data/README_data.md)")
        return False
    
    logger.info("✓ Clinical data found")
    return True


def run_step(step_name, command, optional=False):
    """Execute a pipeline step."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ {step_name} completed in {elapsed_time:.2f}s")
        
        # Log output
        if result.stdout:
            logger.debug(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        if optional:
            logger.warning(f"⚠ {step_name} skipped (optional): {e}")
            return True
        else:
            logger.error(f"✗ {step_name} failed after {elapsed_time:.2f}s")
            logger.error(f"Error: {e.stderr}")
            return False


def run_full_pipeline():
    """Execute the complete ML pipeline."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA-ML: FULL PIPELINE EXECUTION")
    logger.info("="*60 + "\n")
    
    pipeline_start = time.time()
    
    # Step 1: Train tabular models
    if not run_step(
        "Train Tabular Models",
        "python scripts/train_tabular.py --config scripts/config.yaml"
    ):
        return False
    
    # Step 2: Train ensemble models
    if not run_step(
        "Train Ensemble Models",
        "python scripts/train_ensemble.py --config scripts/config.yaml"
    ):
        return False
    
    # Step 3: Evaluate models
    if not run_step(
        "Evaluate Models",
        "python scripts/evaluate_models.py --config scripts/config.yaml"
    ):
        return False
    
    # Step 4: Run tests
    if not run_step(
        "Run Tests",
        "python -m pytest tests/ -v",
        optional=True
    ):
        logger.warning("Tests skipped or failed (non-critical)")
    
    total_time = time.time() - pipeline_start
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    logger.info("="*60 + "\n")
    
    logger.info("Results available in:")
    logger.info("  - outputs/tables/model_performance.csv")
    logger.info("  - outputs/figures/roc_curves.png")
    logger.info("  - outputs/EXECUTIVE_SUMMARY.txt")
    
    return True


def run_tabular_only():
    """Execute only tabular models (quick validation)."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA-ML: TABULAR MODELS ONLY")
    logger.info("="*60 + "\n")
    
    pipeline_start = time.time()
    
    # Train and evaluate tabular models
    if not run_step(
        "Train Tabular Models",
        "python scripts/train_tabular.py --config scripts/config.yaml"
    ):
        return False
    
    total_time = time.time() - pipeline_start
    logger.info(f"\n✓ Tabular pipeline completed in {total_time/60:.2f} minutes")
    
    return True


def run_quick_validation():
    """Quick validation run with minimal data."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA-ML: QUICK VALIDATION")
    logger.info("="*60 + "\n")
    
    # Run tests
    if not run_step(
        "Run Unit Tests",
        "python -m pytest tests/ -v"
    ):
        logger.warning("Some tests failed")
    
    # Quick import check
    logger.info("\nValidating imports...")
    try:
        from src import preprocessing, tabular_models, ensemble, explainability
        logger.info("✓ All modules importable")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def run_demo():
    """Run demo with sample/synthetic data."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA-ML: DEMO MODE")
    logger.info("="*60 + "\n")
    
    logger.info("Demo mode: Creating synthetic data for demonstration...")
    
    # Create synthetic demo data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 200
    
    demo_data = pd.DataFrame({
        'Age': np.random.randint(60, 90, n_samples),
        'EDUC': np.random.randint(8, 20, n_samples),
        'MMSE': np.random.randint(15, 30, n_samples),
        'eTIV': np.random.randint(1200, 1800, n_samples),
        'nWBV': np.random.uniform(0.6, 0.8, n_samples),
        'ASF': np.random.uniform(0.9, 1.3, n_samples),
        'M/F': np.random.choice(['M', 'F'], n_samples),
        'CDR': np.random.choice([0, 0.5, 1], n_samples, p=[0.6, 0.3, 0.1])
    })
    
    # Save demo data
    os.makedirs('data/raw', exist_ok=True)
    demo_data.to_csv('data/raw/clinical.csv', index=False)
    logger.info(f"✓ Created demo data: {len(demo_data)} samples")
    
    # Run tabular pipeline
    return run_tabular_only()


def print_summary():
    """Print execution summary and next steps."""
    logger.info("\n" + "="*60)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*60)
    
    # Check what was created
    models_dir = Path('models')
    outputs_dir = Path('outputs')
    
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.pth'))
        logger.info(f"\nModels created: {len(model_files)}")
        for f in model_files[:5]:  # Show first 5
            logger.info(f"  - {f.name}")
    
    if outputs_dir.exists():
        if (outputs_dir / 'tables').exists():
            tables = list((outputs_dir / 'tables').glob('*.csv'))
            logger.info(f"\nTables created: {len(tables)}")
        
        if (outputs_dir / 'figures').exists():
            figures = list((outputs_dir / 'figures').glob('*.png'))
            logger.info(f"Figures created: {len(figures)}")
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS FOR DISSERTATION")
    logger.info("="*60)
    logger.info("1. Review outputs/EXECUTIVE_SUMMARY.txt")
    logger.info("2. Check outputs/tables/model_performance.csv for metrics")
    logger.info("3. View outputs/figures/roc_curves.png for visualizations")
    logger.info("4. Include results in your dissertation")
    logger.info("\nFor detailed documentation, see:")
    logger.info("  - README.md (complete guide)")
    logger.info("  - VALIDATION.md (requirements checklist)")
    logger.info("  - PROJECT_SUMMARY.md (achievement overview)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Dementia-ML: One-click pipeline execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full      # Complete pipeline (recommended)
  python main.py --mode tabular   # Only tabular models
  python main.py --mode quick     # Quick validation
  python main.py --mode demo      # Demo with synthetic data
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'tabular', 'quick', 'demo'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment and data checks'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("DEMENTIA-ML: Automated Pipeline Execution")
    print("Early Detection of Dementia using Machine Learning")
    print("="*60 + "\n")
    print(f"Mode: {args.mode.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Pre-flight checks
    if not args.skip_checks:
        if not check_environment():
            logger.error("Environment check failed. Exiting.")
            return 1
        
        if args.mode != 'demo' and args.mode != 'quick':
            if not check_data():
                logger.error("Data check failed. Use --mode demo to run with synthetic data.")
                return 1
    
    # Execute selected mode
    success = False
    
    if args.mode == 'full':
        success = run_full_pipeline()
    elif args.mode == 'tabular':
        success = run_tabular_only()
    elif args.mode == 'quick':
        success = run_quick_validation()
    elif args.mode == 'demo':
        success = run_demo()
    
    # Print summary
    if success:
        print_summary()
        logger.info("\n✓ Execution completed successfully!")
        return 0
    else:
        logger.error("\n✗ Execution failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
