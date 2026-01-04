#!/usr/bin/env python3
"""
Evaluate trained models and generate comprehensive performance reports.

This script loads trained models, evaluates them on test data,
and generates performance metrics and visualizations.

Usage:
    python evaluate_models.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loading import load_clinical_data


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


def evaluate_model(model, X_test, y_test, model_name, logger):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    }
    
    logger.info(f"{model_name}:")
    logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
    logger.info(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")
    logger.info(f"  Recall: {metrics['Recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    return metrics, y_pred, y_pred_proba


def plot_roc_curves(models_dict, X_test, y_test, save_path, logger):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        linestyle = '-' if 'Ensemble' in name else '--'
        linewidth = 2.5 if 'Ensemble' in name else 1.5
        
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', 
                linestyle=linestyle, linewidth=linewidth)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_confusion_matrix(y_test, y_pred, model_name, save_path, logger):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Demented', 'Demented'],
                yticklabels=['Non-Demented', 'Demented'])
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def main(config_path):
    """Main evaluation function."""
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("Starting model evaluation pipeline")
    
    # Load models
    models_dir = config['outputs']['models_dir']
    models = {}
    
    try:
        # Load tabular models
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
            'Stacking Ensemble': 'stacking_ensemble.pkl',
            'Voting Ensemble': 'voting_ensemble.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                logger.info(f"Loaded {name}")
        
        # Load preprocessor
        with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Loaded preprocessor")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Please train models first using train_*.py scripts.")
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
    
    # Train-test split (same as training)
    train_config = config['training']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=train_config['test_size'],
        random_state=train_config['random_state'],
        stratify=y_binary if train_config['stratify'] else None
    )
    
    # Preprocess
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test.index
    )
    
    logger.info(f"Test set size: {len(X_test_processed)}")
    
    # Evaluate all models
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*60 + "\n")
    
    all_metrics = []
    predictions = {}
    
    for name, model in models.items():
        metrics, y_pred, y_pred_proba = evaluate_model(
            model, X_test_processed, y_test, name, logger
        )
        all_metrics.append(metrics)
        predictions[name] = (y_pred, y_pred_proba)
        logger.info("")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)
    results_df = results_df.sort_values('AUC-ROC', ascending=False)
    
    # Save results
    results_dir = config['outputs']['results_dir']
    tables_dir = config['outputs']['tables_dir']
    figures_dir = config['outputs']['figures_dir']
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save comprehensive results table
    results_path = os.path.join(tables_dir, 'model_performance.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results table saved to {results_path}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # ROC curves
    roc_path = os.path.join(figures_dir, 'roc_curves.png')
    plot_roc_curves(models, X_test_processed, y_test, roc_path, logger)
    
    # Confusion matrices for top 3 models
    top_models = results_df.head(3)['Model'].tolist()
    for model_name in top_models:
        y_pred, _ = predictions[model_name]
        cm_path = os.path.join(figures_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"\nBest Model: {results_df.iloc[0]['Model']}")
    logger.info(f"Best AUC-ROC: {results_df.iloc[0]['AUC-ROC']:.4f}")
    logger.info(f"Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    logger.info("\nAll results saved to:")
    logger.info(f"  Tables: {tables_dir}")
    logger.info(f"  Figures: {figures_dir}")
    logger.info("\nEvaluation pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
