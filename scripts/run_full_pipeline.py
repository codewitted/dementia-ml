#!/usr/bin/env python3
"""
Automated Full Pipeline Runner for Dementia-ML Project.

This script runs the complete ML pipeline including:
1. Data generation (realistic OASIS-1 style data)
2. Tabular model training (LR, RF, GBM)
3. Ensemble model training (Stacking, Voting)
4. Model evaluation and report generation

Usage:
    python scripts/run_full_pipeline.py

Output:
    - models/: Trained model files (.pkl)
    - outputs/tables/: Performance metrics (CSV)
    - outputs/figures/: Visualizations (PNG)
    - outputs/EXECUTIVE_SUMMARY.txt: Full results summary
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_step(name: str, command: str, cwd: Path) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n📍 Running: {name}")
    print(f"   Command: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✓ Completed in {elapsed:.2f}s")
            return True
        else:
            print(f"   ✗ Failed after {elapsed:.2f}s")
            print(f"   Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ✗ Timed out after 600s")
        return False
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False


def main():
    """Run complete ML pipeline."""
    print_header("DEMENTIA-ML: FULL AUTOMATED PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Track results
    results = {
        'steps_completed': [],
        'steps_failed': [],
        'start_time': datetime.now()
    }
    
    # Use scripts directory for commands that use config.yaml
    scripts_dir = project_root / "scripts"
    
    # Step 1: Generate data
    print_header("STEP 1: DATA GENERATION")
    if run_step(
        "Generate OASIS-style clinical data",
        "python generate_realistic_oasis_data.py",
        scripts_dir
    ):
        results['steps_completed'].append("Data Generation")
    else:
        print("\n❌ Data generation failed. Cannot proceed.")
        return 1
    
    # Step 2: Train tabular models
    print_header("STEP 2: TABULAR MODEL TRAINING")
    if run_step(
        "Train Logistic Regression, Random Forest, Gradient Boosting",
        "python train_tabular.py --config config.yaml",
        scripts_dir
    ):
        results['steps_completed'].append("Tabular Models")
    else:
        results['steps_failed'].append("Tabular Models")
        print("\n⚠️ Tabular training failed. Skipping ensembles.")
        return 1
    
    # Step 3: Train ensemble models
    print_header("STEP 3: ENSEMBLE MODEL TRAINING")
    if run_step(
        "Train Stacking and Voting Ensembles",
        "python train_ensemble.py --config config.yaml",
        scripts_dir
    ):
        results['steps_completed'].append("Ensemble Models")
    else:
        results['steps_failed'].append("Ensemble Models")
        print("\n⚠️ Ensemble training failed. Continuing with evaluation.")
    
    # Step 4: Evaluate models
    print_header("STEP 4: MODEL EVALUATION")
    if run_step(
        "Evaluate all models and generate reports",
        "python evaluate_models.py --config config.yaml",
        scripts_dir
    ):
        results['steps_completed'].append("Model Evaluation")
    else:
        results['steps_failed'].append("Model Evaluation")
    
    # Step 5: Run tests
    print_header("STEP 5: VALIDATION TESTS")
    if run_step(
        "Run unit tests",
        "python -m pytest tests/ -v",
        project_root
    ):
        results['steps_completed'].append("Unit Tests")
    else:
        results['steps_failed'].append("Unit Tests")
        print("\n⚠️ Some tests failed (non-critical)")
    
    # Summary
    print_header("PIPELINE EXECUTION COMPLETE")
    
    end_time = datetime.now()
    elapsed = (end_time - results['start_time']).total_seconds()
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Elapsed time: {elapsed/60:.1f} minutes")
    print(f"   Steps completed: {len(results['steps_completed'])}")
    print(f"   Steps failed: {len(results['steps_failed'])}")
    
    if results['steps_completed']:
        print(f"\n   ✓ Completed:")
        for step in results['steps_completed']:
            print(f"     - {step}")
    
    if results['steps_failed']:
        print(f"\n   ✗ Failed:")
        for step in results['steps_failed']:
            print(f"     - {step}")
    
    # Output locations
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   Models: {project_root / 'models'}/")
    print(f"   Tables: {project_root / 'outputs' / 'tables'}/")
    print(f"   Figures: {project_root / 'outputs' / 'figures'}/")
    print(f"   Summary: {project_root / 'outputs' / 'EXECUTIVE_SUMMARY.txt'}")
    
    # Final status
    if not results['steps_failed']:
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("   Results are ready for dissertation use.")
        return 0
    else:
        print("\n⚠️ PIPELINE COMPLETED WITH WARNINGS")
        print("   Some steps failed - check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
