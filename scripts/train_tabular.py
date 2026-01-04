#!/usr/bin/env python3
"""
Train tabular models for dementia prediction.

This script trains Logistic Regression, Random Forest, and Gradient Boosting
models on clinical data and saves them for later use.

Usage:
    python train_tabular.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loading import load_clinical_data
from preprocessing import get_fit_transform
from tabular_models import train_logistic_regression, train_random_forest, train_gbm


def setup_logging(config):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("Starting tabular model training pipeline")
    
    # Load data
    data_config = config['data']
    clinical_path = os.path.join(data_config['raw_dir'], data_config['clinical_file'])
    
    logger.info(f"Loading data from {clinical_path}")
    df = load_clinical_data(clinical_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Prepare features and target
    feature_config = config['features']
    numeric_features = feature_config['numeric']
    categorical_features = feature_config['categorical']
    target_column = feature_config['target']
    
    # Clean data
    df_clean = df.dropna(subset=[target_column])
    logger.info(f"After removing missing targets: {len(df_clean)} samples")
    
    # Select features
    all_features = numeric_features + categorical_features
    available_features = [f for f in all_features if f in df_clean.columns]
    X = df_clean[available_features]
    y = df_clean[target_column]
    
    # Convert to binary classification (non-demented vs demented)
    y_binary = (y > 0).astype(int)
    logger.info(f"Class distribution: {y_binary.value_counts().to_dict()}")
    
    # Train-test split
    train_config = config['training']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=train_config['test_size'],
        random_state=train_config['random_state'],
        stratify=y_binary if train_config['stratify'] else None
    )
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Preprocess data
    actual_numeric = [f for f in numeric_features if f in X.columns]
    actual_categorical = [f for f in categorical_features if f in X.columns]
    
    logger.info("Preprocessing data...")
    X_train_processed, preprocessor = get_fit_transform(
        X_train, actual_numeric, actual_categorical
    )
    X_test_processed = preprocessor.transform(X_test)
    
    # Create output directory
    models_dir = config['outputs']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    # Train models
    models = {}
    model_configs = config['models']
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_config = model_configs['logistic_regression']
    models['logistic_regression'] = train_logistic_regression(
        X_train_processed, y_train, **lr_config
    )
    logger.info("Logistic Regression training complete")
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_config = model_configs['random_forest']
    models['random_forest'] = train_random_forest(
        X_train_processed, y_train, **rf_config
    )
    logger.info("Random Forest training complete")
    
    # Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gbm_config = model_configs['gradient_boosting']
    models['gradient_boosting'] = train_gbm(
        X_train_processed, y_train, **gbm_config
    )
    logger.info("Gradient Boosting training complete")
    
    # Save models
    for name, model in models.items():
        model_path = os.path.join(models_dir, f'{name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} to {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Evaluate models
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    logger.info("\nModel Evaluation:")
    logger.info("=" * 60)
    for name, model in models.items():
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"{name.replace('_', ' ').title()}: Accuracy={acc:.4f}, AUC={auc:.4f}")
    
    logger.info("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular models for dementia prediction")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
