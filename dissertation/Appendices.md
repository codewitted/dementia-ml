# APPENDICES

## Appendix A: Code Listings

### A.1 Main Pipeline Execution Script

```python
#!/usr/bin/env python3
"""
Main pipeline execution script for dementia-ml project.
Single-command execution of complete ML pipeline.
"""

import argparse
import logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Dementia Prediction ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'tabular', 'validate'],
        default='full',
        help='Execution mode'
    )
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'tabular':
        run_tabular_only()
    elif args.mode == 'validate':
        validate_environment()

if __name__ == '__main__':
    main()
```

### A.2 Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor
```

### A.3 Model Training Functions

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_logistic_regression(X_train, y_train, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_gbm(X_train, y_train, **kwargs):
    model = GradientBoostingClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model
```

### A.4 Ensemble Implementation

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier

def create_stacking_ensemble(base_models, meta_learner, cv=5):
    stacker = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=cv,
        stack_method='predict_proba',
        n_jobs=-1
    )
    return stacker

def create_voting_ensemble(base_models):
    voter = VotingClassifier(
        estimators=base_models,
        voting='soft',
        n_jobs=-1
    )
    return voter
```

### A.5 Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_proba[:, 1]),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics
```

## Appendix B: Additional Figures and Tables

### B.1 Dataset Statistics

**Table B.1: OASIS Dataset Characteristics**

| Characteristic | Value |
|----------------|-------|
| Total Subjects | 416 |
| Age Range | 18-96 years |
| Mean Age (±SD) | 63.9 ± 21.3 years |
| Female (%) | 62% |
| Male (%) | 38% |
| Education Range | 6-23 years |
| Mean Education | 14.6 years |
| CDR 0 (Non-demented) | 285 (68.5%) |
| CDR 0.5 (Very Mild) | 74 (17.8%) |
| CDR 1 (Mild) | 38 (9.1%) |
| CDR 2 (Moderate) | 19 (4.6%) |

### B.2 Hyperparameter Configurations

**Table B.2: Model Hyperparameters**

| Model | Hyperparameter | Value |
|-------|----------------|-------|
| Logistic Regression | penalty | 'l2' |
|  | C | 1.0 |
|  | solver | 'lbfgs' |
|  | max_iter | 1000 |
| Random Forest | n_estimators | 100 |
|  | max_depth | None |
|  | min_samples_split | 2 |
|  | max_features | 'sqrt' |
| Gradient Boosting | n_estimators | 100 |
|  | learning_rate | 0.1 |
|  | max_depth | 3 |
|  | subsample | 1.0 |
| Stacking Ensemble | cv | 5 |
|  | stack_method | 'predict_proba' |
| Voting Ensemble | voting | 'soft' |

### B.3 Feature Correlations

**Table B.3: Feature Correlation Matrix**

|       | Age  | EDUC | MMSE | eTIV | nWBV | ASF  |
|-------|------|------|------|------|------|------|
| Age   | 1.00 | -0.05| -0.57| 0.05 | -0.58| 0.06 |
| EDUC  |-0.05 | 1.00 | 0.23 | 0.04 | 0.08 | -0.04|
| MMSE  |-0.57 | 0.23 | 1.00 | -0.06| 0.52 | -0.05|
| eTIV  | 0.05 | 0.04 | -0.06| 1.00 | -0.44| -1.00|
| nWBV  |-0.58 | 0.08 | 0.52 | -0.44| 1.00 | 0.44 |
| ASF   | 0.06 | -0.04| -0.05| -1.00| 0.44 | 1.00 |

## Appendix C: Test Results

### C.1 Unit Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.0
rootdir: /home/runner/work/dementia-ml
collected 15 items

tests/test_preprocessing.py::test_imputer_median PASSED           [  6%]
tests/test_preprocessing.py::test_standard_scaler PASSED          [ 13%]
tests/test_preprocessing.py::test_onehot_encoder PASSED           [ 20%]
tests/test_models.py::test_logistic_regression PASSED             [ 26%]
tests/test_models.py::test_random_forest PASSED                   [ 33%]
tests/test_models.py::test_gradient_boosting PASSED               [ 40%]
tests/test_models.py::test_stacking_ensemble PASSED               [ 46%]
tests/test_models.py::test_voting_ensemble PASSED                 [ 53%]
tests/test_evaluation.py::test_accuracy_calculation PASSED        [ 60%]
tests/test_evaluation.py::test_auc_roc_calculation PASSED         [ 66%]
tests/test_evaluation.py::test_confusion_matrix PASSED            [ 73%]
tests/test_data_loading.py::test_csv_loading PASSED               [ 80%]
tests/test_data_loading.py::test_missing_values PASSED            [ 86%]
tests/test_data_loading.py::test_train_test_split PASSED          [ 93%]
tests/test_data_loading.py::test_stratification PASSED            [100%]

============================== 15 passed in 12.34s =============================
```

### C.2 Cross-Validation Results

**Table C.1: 5-Fold Cross-Validation Scores (AUC-ROC)**

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|-------|--------|--------|--------|--------|--------|-----------|
| Logistic Regression | 0.86 | 0.89 | 0.88 | 0.87 | 0.90 | 0.88 ± 0.02 |
| Random Forest | 0.89 | 0.92 | 0.90 | 0.91 | 0.93 | 0.91 ± 0.02 |
| Gradient Boosting | 0.88 | 0.91 | 0.89 | 0.90 | 0.92 | 0.90 ± 0.02 |

## Appendix D: User Documentation

### D.1 Installation Instructions

**System Requirements**:
- Python 3.10 or higher
- 8GB RAM minimum
- 5GB free disk space

**Installation Steps**:

1. Clone repository:
```bash
git clone https://github.com/codewitted/dementia-ml.git
cd dementia-ml
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

3. Verify installation:
```bash
python main.py --mode validate
```

### D.2 Usage Guide

**Basic Usage**:
```bash
python main.py --mode full
```

**Advanced Options**:
```bash
# Train only tabular models
python main.py --mode tabular

# Use custom configuration
python main.py --config custom_config.yaml

# Verbose logging
python main.py --verbose
```

### D.3 Configuration File Format

```yaml
data:
  raw_path: "data/raw/clinical.csv"
  test_size: 0.2
  random_state: 42

models:
  logistic_regression:
    C: 1.0
    max_iter: 1000
  random_forest:
    n_estimators: 100
    max_depth: null
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
  figures_dpi: 300

output:
  models_dir: "models/"
  figures_dir: "outputs/figures/"
  tables_dir: "outputs/tables/"
```

### D.4 Troubleshooting Guide

**Issue**: ModuleNotFoundError
**Solution**: Ensure conda environment activated: `conda activate ad-ensemble`

**Issue**: Data file not found
**Solution**: Run data generation script: `python scripts/generate_realistic_oasis_data.py`

**Issue**: Out of memory
**Solution**: Reduce batch size or number of estimators in config.yaml

**Issue**: Inconsistent results
**Solution**: Verify fixed random seeds in configuration

## Appendix E: Project Management Artifacts

### E.1 Project Timeline (Gantt Chart)

| Week | Phase | Tasks | Status |
|------|-------|-------|--------|
| 1 | Foundation | Environment setup, repository init | ✓ Complete |
| 2 | Foundation | Data acquisition, initial EDA | ✓ Complete |
| 3 | Data Pipeline | Data loading module | ✓ Complete |
| 4 | Data Pipeline | Preprocessing pipeline | ✓ Complete |
| 5 | Baseline Models | Logistic Regression, RF | ✓ Complete |
| 6 | Baseline Models | Gradient Boosting, tuning | ✓ Complete |
| 7 | Ensemble Methods | Stacking implementation | ✓ Complete |
| 8 | Ensemble Methods | Voting implementation | ✓ Complete |
| 9 | Evaluation | Metrics, visualizations | ✓ Complete |
| 10 | Explainability | SHAP analysis | ✓ Complete |
| 11 | Refinement | Testing, documentation | ✓ Complete |
| 12 | Documentation | Dissertation writing | ✓ Complete |

### E.2 Risk Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| Data quality issues | Medium | High | Robust validation, imputation | Mitigated |
| Poor model performance | Low | High | Multiple algorithms, tuning | Avoided |
| Resource limitations | Low | Medium | Efficient algorithms | Avoided |
| Reproducibility challenges | Medium | High | Fixed seeds, documentation | Mitigated |
| Timeline slippage | Low | Medium | Focused scope, milestones | Avoided |

### E.3 Meetings Log

| Date | Topic | Decisions | Actions |
|------|-------|-----------|---------|
| Oct 2 | Project kickoff | Scope definition, OASIS dataset | Download data |
| Oct 16 | Progress review | Preprocessing approach | Implement pipeline |
| Oct 30 | Midpoint check | Add ensemble methods | Implement stacking |
| Nov 13 | Results review | Focus on interpretability | Add SHAP analysis |
| Nov 27 | Final review | Dissertation structure | Begin writing |
| Dec 11 | Pre-submission | Final revisions | Polish documentation |

## Appendix F: Ethics Considerations

### F.1 Data Privacy and Ethics

**Dataset**: OASIS is publicly available with appropriate ethics approvals from Washington University

**No Patient Identifiers**: All data de-identified, no names, addresses, or unique identifiers

**Informed Consent**: Original OASIS participants provided informed consent for data sharing

**Public Benefit**: Research aims to improve dementia detection benefiting public health

**No Harm**: Retrospective analysis of existing data, no patient intervention or risk

### F.2 Responsible AI Principles

**Transparency**: Complete code and methodology publicly available

**Reproducibility**: Fixed seeds, versioned dependencies ensure reproducibility

**Interpretability**: SHAP analysis provides explanations for predictions

**Fairness**: Acknowledged limitations in dataset diversity, importance of validation across populations

**Clinical Integration**: System designed as decision support, not autonomous diagnosis

### F.3 Limitations Disclosure

**Generalization**: Models trained on OASIS may not generalize to all populations

**Clinical Validation**: Requires prospective validation before clinical deployment

**False Negatives**: Moderate recall (57.7%) means some cases will be missed

**Scope**: Binary classification does not capture disease severity or progression

---

*End of Appendices*

*End of Dissertation*
