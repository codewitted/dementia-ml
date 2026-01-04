#!/usr/bin/env python3
"""
AI Agent Automation Script for Dementia-ML

This script enables AI agents (Claude, ChatGPT, etc.) to automatically
execute the ML pipeline and generate results.

Usage:
    python scripts/ai_agent_run.py --auto          # Full automated run
    python scripts/ai_agent_run.py --validate      # Validate only
    python scripts/ai_agent_run.py --report        # Generate report
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class AIAgentRunner:
    """Automated execution manager for AI agents."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.manifest_path = self.project_root / 'PROJECT_MANIFEST.json'
        self.results = {
            'started_at': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'outputs_generated': [],
            'status': 'in_progress'
        }
    
    def load_manifest(self):
        """Load project manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError("PROJECT_MANIFEST.json not found")
        
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def validate_environment(self):
        """Validate that environment is ready."""
        print("🔍 Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ required")
        
        # Check critical packages
        try:
            import pandas
            import sklearn
            import yaml
            print("✓ Core packages installed")
        except ImportError as e:
            raise RuntimeError(f"Missing package: {e}")
        
        return True
    
    def check_data_availability(self):
        """Check if data is available."""
        print("🔍 Checking data availability...")
        
        clinical_data = self.project_root / 'data' / 'raw' / 'clinical.csv'
        
        if clinical_data.exists():
            print(f"✓ Clinical data found: {clinical_data}")
            return True
        else:
            print("⚠ Clinical data not found - will create synthetic demo data")
            return False
    
    def create_demo_data(self):
        """Create synthetic demo data for testing."""
        print("📊 Creating synthetic demo data...")
        
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 300
        
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
        
        data_dir = self.project_root / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        demo_data.to_csv(data_dir / 'clinical.csv', index=False)
        print(f"✓ Created {len(demo_data)} synthetic samples")
    
    def execute_step(self, step_name, command):
        """Execute a pipeline step."""
        print(f"\n{'='*60}")
        print(f"📍 Executing: {step_name}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✓ {step_name} completed")
                self.results['steps_completed'].append(step_name)
                return True
            else:
                print(f"✗ {step_name} failed")
                print(f"Error: {result.stderr[:500]}")
                self.results['steps_failed'].append({
                    'step': step_name,
                    'error': result.stderr[:500]
                })
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ {step_name} timed out")
            self.results['steps_failed'].append({
                'step': step_name,
                'error': 'Timeout'
            })
            return False
        except Exception as e:
            print(f"✗ {step_name} error: {e}")
            self.results['steps_failed'].append({
                'step': step_name,
                'error': str(e)
            })
            return False
    
    def run_pipeline(self):
        """Execute the complete ML pipeline."""
        print("\n🚀 Starting automated ML pipeline...\n")
        
        # Step 1: Train tabular models
        if not self.execute_step(
            "Train Tabular Models",
            "python scripts/train_tabular.py --config scripts/config.yaml"
        ):
            print("⚠ Pipeline stopped due to failure in tabular training")
            return False
        
        # Step 2: Train ensembles
        if not self.execute_step(
            "Train Ensemble Models",
            "python scripts/train_ensemble.py --config scripts/config.yaml"
        ):
            print("⚠ Ensemble training failed, continuing...")
        
        # Step 3: Evaluate
        if not self.execute_step(
            "Evaluate Models",
            "python scripts/evaluate_models.py --config scripts/config.yaml"
        ):
            print("⚠ Evaluation failed")
        
        return True
    
    def collect_outputs(self):
        """Collect and catalog generated outputs."""
        print("\n📁 Collecting outputs...")
        
        outputs_dir = self.project_root / 'outputs'
        models_dir = self.project_root / 'models'
        
        # Collect model files
        if models_dir.exists():
            models = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.pth'))
            self.results['outputs_generated'].extend([
                {'type': 'model', 'path': str(m.relative_to(self.project_root))}
                for m in models
            ])
        
        # Collect result files
        if outputs_dir.exists():
            tables = list((outputs_dir / 'tables').glob('*.csv')) if (outputs_dir / 'tables').exists() else []
            figures = list((outputs_dir / 'figures').glob('*.png')) if (outputs_dir / 'figures').exists() else []
            
            self.results['outputs_generated'].extend([
                {'type': 'table', 'path': str(t.relative_to(self.project_root))}
                for t in tables
            ])
            self.results['outputs_generated'].extend([
                {'type': 'figure', 'path': str(f.relative_to(self.project_root))}
                for f in figures
            ])
        
        print(f"✓ Found {len(self.results['outputs_generated'])} output files")
    
    def generate_report(self):
        """Generate AI-friendly execution report."""
        self.results['completed_at'] = datetime.now().isoformat()
        self.results['status'] = 'completed' if not self.results['steps_failed'] else 'partial'
        
        report_path = self.project_root / 'AI_EXECUTION_REPORT.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Steps completed: {len(self.results['steps_completed'])}")
        print(f"Steps failed: {len(self.results['steps_failed'])}")
        print(f"Outputs generated: {len(self.results['outputs_generated'])}")
        print(f"Status: {self.results['status']}")
        
        if self.results['steps_failed']:
            print("\nFailed steps:")
            for failure in self.results['steps_failed']:
                print(f"  - {failure['step']}: {failure['error'][:100]}")
        
        return report_path
    
    def auto_run(self):
        """Fully automated execution."""
        print("🤖 AI Agent Automated Execution")
        print("="*60 + "\n")
        
        try:
            # Validate
            self.validate_environment()
            
            # Check/create data
            if not self.check_data_availability():
                self.create_demo_data()
            
            # Run pipeline
            self.run_pipeline()
            
            # Collect outputs
            self.collect_outputs()
            
            # Generate report
            report_path = self.generate_report()
            
            print("\n✓ Automated execution completed!")
            print(f"\nResults available at: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Execution failed: {e}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Agent automation for dementia-ml"
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Full automated execution'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate environment only'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate report from existing outputs'
    )
    
    args = parser.parse_args()
    
    runner = AIAgentRunner()
    
    if args.validate:
        runner.validate_environment()
        runner.check_data_availability()
        print("✓ Validation complete")
    elif args.report:
        runner.collect_outputs()
        runner.generate_report()
    elif args.auto:
        success = runner.auto_run()
        sys.exit(0 if success else 1)
    else:
        # Default: auto run
        success = runner.auto_run()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
