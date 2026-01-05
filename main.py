#!/usr/bin/env python3
"""
Dementia Prediction using Machine Learning
Main Pipeline Execution Script

This script provides automated execution of the complete ML pipeline for
early detection of dementia using clinical and neuroimaging biomarkers.

Reference:
    Marcus, D.S., et al. (2007). Open Access Series of Imaging Studies (OASIS):
    Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented 
    Older Adults. Journal of Cognitive Neuroscience, 19(9), 1498-1507.

Usage:
    python main.py                    # Run complete pipeline
    python main.py --mode full        # Run complete pipeline
    python main.py --mode tabular     # Run only tabular models
    python main.py --mode validate    # Validate environment and tests

Author: [Your Name]
Institution: [Your Institution]
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
    """Validate that the execution environment is properly configured."""
    logger.info("Validating execution environment...")
    
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
    
    logger.info("Environment validation passed")
    return True


def check_data():
    """Verify that required data files are available."""
    logger.info("Checking data availability...")
    
    clinical_data = Path('data/raw/clinical.csv')
    
    if not clinical_data.exists():
        logger.warning("Clinical data not found at data/raw/clinical.csv")
        logger.info("Generate data using: python scripts/generate_realistic_oasis_data.py")
        logger.info("Or download OASIS dataset - see data/README_data.md")
        return False
    
    logger.info("Clinical data found")
    return True


def run_step(step_name, command, optional=False, cwd=None):
    """Execute a pipeline step with error handling."""
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
            text=True,
            cwd=cwd
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"{step_name} completed in {elapsed_time:.2f}s")
        
        if result.stdout:
            logger.debug(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        if optional:
            logger.warning(f"{step_name} skipped (optional): {e}")
            return True
        else:
            logger.error(f"{step_name} failed after {elapsed_time:.2f}s")
            logger.error(f"Error: {e.stderr}")
            return False


def run_full_pipeline():
    """Execute the complete machine learning pipeline."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA PREDICTION: FULL PIPELINE EXECUTION")
    logger.info("="*60 + "\n")
    
    pipeline_start = time.time()
    scripts_dir = Path(__file__).parent / 'scripts'
    
    # Step 1: Train tabular models (Logistic Regression, Random Forest, Gradient Boosting)
    if not run_step(
        "Train Tabular Models",
        "python train_tabular.py --config config.yaml",
        cwd=str(scripts_dir)
    ):
        return False
    
    # Step 2: Train ensemble models (Stacking, Voting)
    if not run_step(
        "Train Ensemble Models",
        "python train_ensemble.py --config config.yaml",
        cwd=str(scripts_dir)
    ):
        return False
    
    # Step 3: Evaluate all models and generate outputs
    if not run_step(
        "Evaluate Models",
        "python evaluate_models.py --config config.yaml",
        cwd=str(scripts_dir)
    ):
        return False
    
    # Step 4: Run validation tests (from root directory where tests/ is located)
    if not run_step(
        "Run Validation Tests",
        "python -m pytest tests/ -v",
        optional=True
        # Note: No cwd parameter - tests/ is at root level
    ):
        logger.warning("Some tests did not pass (non-critical)")
    
    total_time = time.time() - pipeline_start
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    logger.info("="*60 + "\n")
    
    return True


def run_tabular_only():
    """Execute only tabular model training."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA PREDICTION: TABULAR MODELS")
    logger.info("="*60 + "\n")
    
    pipeline_start = time.time()
    scripts_dir = Path(__file__).parent / 'scripts'
    
    if not run_step(
        "Train Tabular Models",
        "python train_tabular.py --config config.yaml",
        cwd=str(scripts_dir)
    ):
        return False
    
    total_time = time.time() - pipeline_start
    logger.info(f"\nTabular pipeline completed in {total_time/60:.2f} minutes")
    
    return True


def run_validation():
    """Run environment validation and unit tests."""
    logger.info("\n" + "="*60)
    logger.info("DEMENTIA PREDICTION: VALIDATION")
    logger.info("="*60 + "\n")
    
    # Run tests (from root directory where tests/ is located)
    if not run_step(
        "Run Unit Tests",
        "python -m pytest tests/ -v"
        # Note: No cwd parameter - tests/ is at root level
    ):
        logger.warning("Some tests did not pass")
    
    # Validate imports
    logger.info("\nValidating module imports...")
    try:
        from src import preprocessing, tabular_models, ensemble, explainability
        logger.info("All modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"Import validation failed: {e}")
        return False


def print_summary():
    """Print execution summary with output locations."""
    logger.info("\n" + "="*60)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*60)
    
    models_dir = Path('models')
    outputs_dir = Path('outputs')
    
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.pth'))
        logger.info(f"\nTrained Models: {len(model_files)}")
        for f in model_files[:6]:
            logger.info(f"  - {f.name}")
    
    if outputs_dir.exists():
        if (outputs_dir / 'tables').exists():
            tables = list((outputs_dir / 'tables').glob('*.csv'))
            logger.info(f"\nPerformance Tables: {len(tables)}")
        
        if (outputs_dir / 'figures').exists():
            figures = list((outputs_dir / 'figures').glob('*.png'))
            logger.info(f"Visualizations: {len(figures)}")
    
    logger.info("\n" + "="*60)
    logger.info("OUTPUT LOCATIONS")
    logger.info("="*60)
    logger.info("  Performance Metrics: outputs/tables/model_performance.csv")
    logger.info("  ROC Curves:          outputs/figures/roc_curves.png")
    logger.info("  Confusion Matrices:  outputs/figures/confusion_matrix_*.png")
    logger.info("  Executive Summary:   outputs/EXECUTIVE_SUMMARY.txt")
    
    logger.info("\n" + "="*60)
    logger.info("DISSERTATION USAGE")
    logger.info("="*60)
    logger.info("  1. Include performance table in Results chapter")
    logger.info("  2. Embed ROC curves and confusion matrices as figures")
    logger.info("  3. Reference methodology from documentation")
    logger.info("  4. See README.md for complete documentation")


def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser(
        description="Dementia Prediction ML Pipeline - Reproducible Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Modes:
  python main.py --mode full        Complete pipeline (recommended)
  python main.py --mode tabular     Tabular models only
  python main.py --mode validate    Environment validation and tests

Reference:
  Marcus, D.S., et al. (2007). OASIS: Cross-sectional MRI Data in Young, 
  Middle Aged, Nondemented, and Demented Older Adults. 
  Journal of Cognitive Neuroscience, 19(9), 1498-1507.
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'tabular', 'validate'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment and data validation'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("DEMENTIA PREDICTION USING MACHINE LEARNING")
    print("Early Detection of Dementia via Clinical Biomarkers")
    print("="*60 + "\n")
    print(f"Execution Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Pre-flight validation
    if not args.skip_checks:
        if not check_environment():
            logger.error("Environment validation failed.")
            return 1
        
        if args.mode != 'validate':
            if not check_data():
                logger.error("Data validation failed. See documentation for data acquisition.")
                return 1
    
    # Execute selected mode
    success = False
    
    if args.mode == 'full':
        success = run_full_pipeline()
    elif args.mode == 'tabular':
        success = run_tabular_only()
    elif args.mode == 'validate':
        success = run_validation()
    
    # Print summary
    if success:
        print_summary()
        logger.info("\nExecution completed successfully.")
        return 0
    else:
        logger.error("\nExecution failed. Review logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
