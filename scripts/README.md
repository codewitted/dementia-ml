# Scripts Directory

This directory contains executable Python scripts for automated model training and evaluation.

## Overview

The scripts provide a command-line interface to the complete machine learning pipeline, allowing for:
- Reproducible experiments
- Automated training workflows
- Batch processing
- Configuration-based customization

## Files

### `config.yaml`
Central configuration file containing all parameters for:
- Data paths
- Model hyperparameters
- Training settings
- Output directories

### `train_tabular.py`
Trains tabular machine learning models on clinical data.

**Models trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting

**Usage:**
```bash
python scripts/train_tabular.py [--config scripts/config.yaml]
```

**Outputs:**
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`
- `models/preprocessor.pkl`

### `train_cnn.py`
Trains a convolutional neural network on MRI images.

**Model:** SimpleMRI2DCNN (2D CNN for grayscale images)

**Usage:**
```bash
python scripts/train_cnn.py [--config scripts/config.yaml]
```

**Requirements:**
- MRI images in `data/raw/Non Demented/` and `data/raw/Mild Dementia/`

**Outputs:**
- `models/cnn_model.pth`

### `train_ensemble.py`
Creates ensemble models combining multiple base learners.

**Ensembles created:**
- Stacking Ensemble (meta-learning)
- Voting Ensemble (soft voting)

**Usage:**
```bash
python scripts/train_ensemble.py [--config scripts/config.yaml]
```

**Prerequisites:**
- Run `train_tabular.py` first

**Outputs:**
- `models/stacking_ensemble.pkl`
- `models/voting_ensemble.pkl`
- `outputs/ensemble_results.csv`

### `evaluate_models.py`
Evaluates all trained models and generates comprehensive reports.

**Usage:**
```bash
python scripts/evaluate_models.py [--config scripts/config.yaml]
```

**Prerequisites:**
- Trained models in `models/` directory

**Outputs:**
- `outputs/tables/model_performance.csv` - Performance metrics
- `outputs/figures/roc_curves.png` - ROC curve comparison
- `outputs/figures/confusion_matrix_*.png` - Confusion matrices

## Complete Workflow

Run scripts in this order:

```bash
# 1. Train base tabular models
python scripts/train_tabular.py

# 2. (Optional) Train CNN on MRI images
python scripts/train_cnn.py

# 3. Create ensemble models
python scripts/train_ensemble.py

# 4. Evaluate all models and generate reports
python scripts/evaluate_models.py
```

## Configuration

### Customizing Parameters

Edit `config.yaml` to modify:

**Data paths:**
```yaml
data:
  raw_dir: "../data/raw"
  clinical_file: "clinical.csv"
```

**Model hyperparameters:**
```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: null
    random_state: 42
```

**Training settings:**
```yaml
training:
  test_size: 0.2
  random_state: 42
  stratify: true
```

## Logging

All scripts log to console by default. Configure logging in `config.yaml`:

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Error Handling

### Common Issues

**"Data file not found"**
- Ensure data is downloaded and placed in correct directory
- Check paths in `config.yaml`

**"Model file not found"**
- Run prerequisite training scripts first
- Check `models/` directory exists

**"Out of memory" (for CNN)**
- Reduce `batch_size` in config.yaml
- Use smaller `image_size`
- Close other applications

## Performance Tips

1. **Use GPU for CNN training**: Much faster than CPU
2. **Adjust n_jobs**: Set to -1 to use all CPU cores
3. **Reduce data size**: For testing, use a subset of data
4. **Batch processing**: Train models sequentially if memory limited

## Integration with Notebooks

Scripts complement the Jupyter notebooks:
- **Notebooks**: Interactive exploration, visualization, teaching
- **Scripts**: Automation, reproducibility, batch processing

Both use the same `src/` modules for consistency.

## Testing

Test scripts work correctly:

```bash
# Dry run with small dataset
python scripts/train_tabular.py --config scripts/config.yaml
```

## Further Reading

- Main README: [../README.md](../README.md)
- Quick Start: [../QUICKSTART.md](../QUICKSTART.md)
- Source code documentation: [../src/](../src/)

---

For questions or issues, please open a GitHub issue.
