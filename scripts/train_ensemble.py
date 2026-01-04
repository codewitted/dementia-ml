#!/usr/bin/env python3
"""
Train ensemble models combining tabular and CNN predictions.

This script creates stacking and voting ensembles from pre-trained base models.

Usage:
    python train_ensemble.py [--config CONFIG_PATH]
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
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loading import load_clinical_data
from ensemble import LateFusionStacker


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
    """Main ensemble training function."""
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("Starting ensemble training pipeline")
    
    # Load base models
    models_dir = config['outputs']['models_dir']
    base_models = {}
    
    try:
        model_names = ['logistic_regression', 'random_forest', 'gradient_boosting']
        for name in model_names:
            model_path = os.path.join(models_dir, f'{name}.pkl')
            with open(model_path, 'rb') as f:
                base_models[name] = pickle.load(f)
            logger.info(f"Loaded {name}")
        
        # Load preprocessor
        with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Loaded preprocessor")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Please run train_tabular.py first to train base models.")
        return
    
    # Load and prepare data
    data_config = config['data']
    clinical_path = os.path.join(data_config['raw_dir'], data_config['clinical_file'])
    
    logger.info(f"Loading data from {clinical_path}")
    df = load_clinical_data(clinical_path)
    
    # Prepare features and target
    feature_config = config['features']
    numeric_features = feature_config['numeric']
    categorical_features = feature_config['categorical']
    target_column = feature_config['target']
    
    # Clean data
    df_clean = df.dropna(subset=[target_column])
    all_features = numeric_features + categorical_features
    available_features = [f for f in all_features if f in df_clean.columns]
    X = df_clean[available_features]
    y = df_clean[target_column]
    y_binary = (y > 0).astype(int)
    
    # Train-test split
    train_config = config['training']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=train_config['test_size'],
        random_state=train_config['random_state'],
        stratify=y_binary if train_config['stratify'] else None
    )
    
    # Preprocess
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names matching the training pipeline format
    actual_numeric = [f for f in feature_config['numeric'] if f in available_features]
    actual_categorical = [f for f in feature_config['categorical'] if f in available_features]
    
    num_features_out = actual_numeric
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    if hasattr(cat_encoder, "get_feature_names_out"):
        cat_features_out = cat_encoder.get_feature_names_out(actual_categorical)
    else:
        cat_features_out = cat_encoder.get_feature_names(actual_categorical)
    
    feature_names = num_features_out + list(cat_features_out)
    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test.index
    )
    
    logger.info(f"Train set: {len(X_train_processed)}, Test set: {len(X_test_processed)}")
    
    # Create base learners list
    base_learners = [
        ('lr', base_models['logistic_regression']),
        ('rf', base_models['random_forest']),
        ('gbm', base_models['gradient_boosting'])
    ]
    
    # Train stacking ensemble
    logger.info("\nTraining stacking ensemble...")
    ensemble_config = config['models']['ensemble']
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking_model = LateFusionStacker(
        base_learners=base_learners,
        meta_learner=meta_learner,
        cv=ensemble_config['cv_folds'],
        n_jobs=ensemble_config['n_jobs']
    )
    
    stacking_model.fit(X_train_processed, y_train)
    logger.info("Stacking ensemble training complete")
    
    # Train voting ensemble
    logger.info("\nTraining voting ensemble...")
    voting_model = VotingClassifier(
        estimators=base_learners,
        voting='soft',
        n_jobs=ensemble_config['n_jobs']
    )
    
    voting_model.fit(X_train_processed, y_train)
    logger.info("Voting ensemble training complete")
    
    # Save ensemble models
    stacking_path = os.path.join(models_dir, 'stacking_ensemble.pkl')
    with open(stacking_path, 'wb') as f:
        pickle.dump(stacking_model, f)
    logger.info(f"Saved stacking ensemble to {stacking_path}")
    
    voting_path = os.path.join(models_dir, 'voting_ensemble.pkl')
    with open(voting_path, 'wb') as f:
        pickle.dump(voting_model, f)
    logger.info(f"Saved voting ensemble to {voting_path}")
    
    # Evaluate ensembles
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    logger.info("\nEnsemble Evaluation:")
    logger.info("=" * 60)
    
    # Stacking
    stacking_pred = stacking_model.predict(X_test_processed)
    stacking_proba = stacking_model.predict_proba(X_test_processed)[:, 1]
    stacking_acc = accuracy_score(y_test, stacking_pred)
    stacking_auc = roc_auc_score(y_test, stacking_proba)
    logger.info(f"Stacking Ensemble: Accuracy={stacking_acc:.4f}, AUC={stacking_auc:.4f}")
    
    # Voting
    voting_pred = voting_model.predict(X_test_processed)
    voting_proba = voting_model.predict_proba(X_test_processed)[:, 1]
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_auc = roc_auc_score(y_test, voting_proba)
    logger.info(f"Voting Ensemble: Accuracy={voting_acc:.4f}, AUC={voting_auc:.4f}")
    
    # Save results
    results_dir = config['outputs']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_df = pd.DataFrame([
        {'Model': 'Stacking Ensemble', 'Accuracy': stacking_acc, 'AUC-ROC': stacking_auc, 'Type': 'Ensemble'},
        {'Model': 'Voting Ensemble', 'Accuracy': voting_acc, 'AUC-ROC': voting_auc, 'Type': 'Ensemble'}
    ])
    
    results_path = os.path.join(results_dir, 'ensemble_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    logger.info("\nEnsemble training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble models for dementia prediction")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
