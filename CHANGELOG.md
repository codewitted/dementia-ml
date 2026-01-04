# Changelog

All notable changes to the dementia-ml project will be documented in this file.

## [Unreleased]

### Added
- Complete workflow notebooks (00-06) covering entire ML pipeline
- Executable training scripts for tabular, CNN, and ensemble models
- Comprehensive evaluation script with visualization generation
- Configuration management via YAML files
- Unit tests for preprocessing and model training
- Documentation: README, QUICKSTART, CONTRIBUTING guides
- Directory structure for organized outputs
- .gitignore for data, models, and output files

### Changed
- Enhanced README with detailed setup and usage instructions
- Restructured project for academic/examiner standards

## [1.0.0] - 2024-01-04

### Initial Release

#### Notebooks
- `00_Data_Provenance_And_Access.ipynb` - Dataset documentation and rationale
- `01_EDA_and_Preprocessing.ipynb` - Exploratory data analysis
- `02_Tabular_Models.ipynb` - Logistic Regression, Random Forest, GBM training
- `03_CNN_Models.ipynb` - CNN for MRI image classification
- `04_Ensemble_Fusion.ipynb` - Stacking and voting ensembles
- `05_Explainability.ipynb` - SHAP values and feature importance
- `06_Results_and_Reporting.ipynb` - Publication-ready outputs

#### Source Code
- `src/data_loading.py` - Data loading utilities
- `src/preprocessing.py` - Preprocessing pipeline
- `src/tabular_models.py` - Tabular model training functions
- `src/cnn_model.py` - CNN architecture definition
- `src/ensemble.py` - Ensemble learning methods
- `src/explainability.py` - SHAP and explainability tools

#### Scripts
- `scripts/config.yaml` - Central configuration
- `scripts/train_tabular.py` - Tabular model training
- `scripts/train_cnn.py` - CNN training
- `scripts/train_ensemble.py` - Ensemble creation
- `scripts/evaluate_models.py` - Model evaluation and reporting

#### Tests
- `tests/test_preprocessing.py` - Preprocessing tests
- `tests/test_models.py` - Model training tests

#### Documentation
- Comprehensive README with installation and usage
- Quick start guide
- Contributing guidelines
- Dataset information and download instructions
- MIT License

#### Features
- **Multi-modal learning**: Tabular and imaging data
- **Ensemble methods**: Stacking and voting
- **Explainability**: SHAP values and visualizations
- **Reproducibility**: Fixed seeds, configuration files
- **Publication-ready**: Tables, figures, benchmarks
- **Academic standards**: Proper citations, documentation

---

## Version History

### Version Numbering
This project uses [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Future Plans
- [ ] Add cross-validation analysis
- [ ] Implement additional explainability methods (Grad-CAM)
- [ ] Add hyperparameter optimization
- [ ] Expand benchmark comparisons
- [ ] Add 3D CNN for volumetric MRI data
- [ ] Integrate additional datasets (ADNI when available)

---

Last updated: 2024-01-04
