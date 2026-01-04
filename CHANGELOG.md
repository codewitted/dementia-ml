# Changelog

All notable changes to the dementia-ml project are documented here.

## [1.0.0] - 2026-01-04

### Initial Release

#### Core Functionality
- Complete machine learning pipeline for dementia prediction
- Tabular models: Logistic Regression, Random Forest, Gradient Boosting
- Ensemble methods: Stacking and Voting classifiers
- Comprehensive model evaluation with multiple metrics
- Publication-ready output generation

#### Source Modules
- `src/data_loading.py` - Data loading utilities for CSV and MRI formats
- `src/preprocessing.py` - Preprocessing pipeline with imputation and scaling
- `src/tabular_models.py` - Tabular model training functions
- `src/cnn_model.py` - CNN architecture for MRI classification
- `src/ensemble.py` - Ensemble learning methods
- `src/explainability.py` - SHAP values and feature importance

#### Scripts
- `scripts/config.yaml` - Central configuration
- `scripts/train_tabular.py` - Tabular model training
- `scripts/train_ensemble.py` - Ensemble creation
- `scripts/evaluate_models.py` - Model evaluation and reporting
- `scripts/generate_realistic_oasis_data.py` - Synthetic data generation
- `scripts/run_full_pipeline.py` - Complete automation

#### Tests
- `tests/test_preprocessing.py` - Preprocessing pipeline tests
- `tests/test_models.py` - Model training tests

#### Documentation
- Comprehensive README with methodology
- Quick start guide
- Reproducibility instructions
- Contributing guidelines

#### Features
- **Multi-modal learning**: Tabular and imaging data support
- **Ensemble methods**: Stacking and voting for improved predictions
- **Explainability**: SHAP values for clinical interpretation
- **Reproducibility**: Fixed seeds, configuration files
- **Publication-ready**: Tables, figures, benchmarks

---

## Version History

This project uses [Semantic Versioning](https://semver.org/).

### Future Roadmap
- Cross-validation analysis
- Hyperparameter optimization
- Additional explainability methods (Grad-CAM)
- Extended benchmark comparisons

---

**Last Updated**: January 2026
