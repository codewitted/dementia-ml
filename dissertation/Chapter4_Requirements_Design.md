# CHAPTER 4: REQUIREMENTS AND DESIGN

## 4.1 Requirements Gathering

Requirements were gathered through multiple sources following established software engineering practices (Pressman and Maxim, 2014):

**Literature Analysis**: Reviewing published dementia prediction systems to understand common features, evaluation metrics, and performance targets (Pellegrini et al., 2018)

**Dataset Examination**: Analyzing OASIS dataset characteristics to determine feasible features and classification tasks (Marcus et al., 2007)

**Supervisor Consultation**: Regular discussions identifying academic and technical requirements (IEEE, 2014)

**Dissertation Guidelines**: Aligning with MSc project evaluation criteria for system development projects

**Best Practices Research**: Studying software engineering and ML research best practices for reproducible systems (Haibe-Kains et al., 2020)

## 4.2 Functional Requirements

### FR1: Data Management

**FR1.1**: System shall load clinical data from CSV files in OASIS format
**FR1.2**: System shall validate data integrity (correct columns, valid ranges, proper types)
**FR1.3**: System shall handle missing values through appropriate imputation strategies
**FR1.4**: System shall perform stratified train-test splitting (80/20 ratio)
**FR1.5**: System shall support both real OASIS data and synthetic data generation

### FR2: Data Preprocessing

**FR2.1**: System shall encode categorical variables (e.g., M/F gender encoding)
**FR2.2**: System shall standardize numerical features (zero mean, unit variance)
**FR2.3**: System shall create derived features as needed (e.g., binary CDR classification)
**FR2.4**: System shall persist preprocessing pipeline for consistent test set transformation
**FR2.5**: System shall validate feature distributions pre and post-transformation

### FR3: Model Training

**FR3.1**: System shall train Logistic Regression with regularization
**FR3.2**: System shall train Random Forest with configurable number of trees
**FR3.3**: System shall train Gradient Boosting Machine with tunable learning rate
**FR3.4**: System shall train Stacking Ensemble using cross-validated base predictions
**FR3.5**: System shall train Voting Ensemble combining base model predictions
**FR3.6**: System shall save trained models to disk for reuse
**FR3.7**: System shall support hyperparameter configuration via YAML files

### FR4: Model Evaluation

**FR4.1**: System shall calculate accuracy, precision, recall, F1-score for all models
**FR4.2**: System shall compute AUC-ROC and plot ROC curves
**FR4.3**: System shall calculate specificity and sensitivity
**FR4.4**: System shall generate confusion matrices for each model
**FR4.5**: System shall perform statistical significance testing between models
**FR4.6**: System shall compare results against published benchmarks

### FR5: Explainability

**FR5.1**: System shall compute feature importance for tree-based models
**FR5.2**: System shall calculate SHAP values for model explanations
**FR5.3**: System shall visualize feature importance rankings
**FR5.4**: System shall generate SHAP summary plots
**FR5.5**: System shall identify top predictive features

### FR6: Visualization and Reporting

**FR6.1**: System shall generate publication-ready ROC curves (300 DPI)
**FR6.2**: System shall create confusion matrix heatmaps
**FR6.3**: System shall produce performance comparison tables
**FR6.4**: System shall export results in CSV and LaTeX formats
**FR6.5**: System shall generate executive summary report
**FR6.6**: System shall save all outputs to organized directory structure

### FR7: Automation and Usability

**FR7.1**: System shall provide command-line interface for pipeline execution
**FR7.2**: System shall support multiple execution modes (full, tabular, validate)
**FR7.3**: System shall provide comprehensive logging of execution progress
**FR7.4**: System shall validate environment and dependencies before execution
**FR7.5**: System shall generate execution logs for debugging

## 4.3 Non-Functional Requirements

### NFR1: Performance

**NFR1.1**: Complete pipeline execution shall complete within 30 minutes on standard hardware
**NFR1.2**: Model training shall utilize available CPU cores for parallelization
**NFR1.3**: Memory usage shall not exceed 8GB RAM
**NFR1.4**: System shall handle datasets up to 10,000 samples efficiently

### NFR2: Reproducibility

**NFR2.1**: System shall use fixed random seeds (seed=42) for all stochastic operations
**NFR2.2**: System shall produce identical results across multiple executions
**NFR2.3**: System shall specify exact dependency versions in environment files
**NFR2.4**: System shall document all configuration parameters
**NFR2.5**: System shall provide verification mechanism for reproducibility

### NFR3: Reliability

**NFR3.1**: System shall validate inputs and provide clear error messages
**NFR3.2**: System shall handle edge cases (empty data, single class, etc.)
**NFR3.3**: System shall not crash on missing or malformed data
**NFR3.4**: System shall pass comprehensive test suite
**NFR3.5**: System shall include exception handling for anticipated failures

### NFR4: Maintainability

**NFR4.1**: Code shall follow PEP 8 Python style guidelines
**NFR4.2**: Functions shall include docstrings describing purpose, parameters, returns
**NFR4.3**: Code shall be modular with clear separation of concerns
**NFR4.4**: Configuration shall be externalized, not hard-coded
**NFR4.5**: System shall include inline comments for complex logic

### NFR5: Portability

**NFR5.1**: System shall run on Windows, macOS, and Linux
**NFR5.2**: System shall support Python 3.10 and higher
**NFR5.3**: System shall use cross-platform file path handling
**NFR5.4**: System shall not depend on proprietary software
**NFR5.5**: System shall provide both conda and pip installation options

### NFR6: Usability

**NFR6.1**: System shall provide comprehensive README with installation and usage instructions
**NFR6.2**: System shall include quick start guide for new users
**NFR6.3**: System shall provide example commands and expected outputs
**NFR6.4**: Error messages shall be clear and actionable
**NFR6.5**: System shall include troubleshooting guide

### NFR7: Security

**NFR7.1**: System shall not store or transmit sensitive patient data
**NFR7.2**: System shall use public datasets with appropriate licenses
**NFR7.3**: System shall not include credentials or API keys in code
**NFR7.4**: System shall validate file paths to prevent directory traversal
**NFR7.5**: System shall use secure dependencies without known vulnerabilities

## 4.4 System Architecture

### 4.4.1 High-Level Architecture

The system follows a layered architecture with clear separation of concerns (Gamma et al., 1994; Bass et al., 2012):

**Layer 1: Data Access Layer**
- Data loading from CSV files (Wickham and Grolemund, 2016)
- Data validation and quality checks (Rahm and Do, 2000)
- Data persistence and caching

**Layer 2: Preprocessing Layer**
- Missing value imputation (Little and Rubin, 2019)
- Feature encoding and scaling (Géron, 2019)
- Feature engineering (Zheng and Casari, 2018)
- Train-test splitting (Hastie et al., 2009)

**Layer 3: Model Layer**
- Individual model training (LR, RF, GBM) (James et al., 2013)
- Ensemble model training (Stacking, Voting) (Dietterich, 2000)
- Model serialization and loading (Pedregosa et al., 2011)
- Hyperparameter management (Bergstra and Bengio, 2012)

**Layer 4: Evaluation Layer**
- Metric calculation
- Statistical testing
- Benchmark comparison
- Result aggregation

**Layer 5: Explainability Layer**
- Feature importance extraction
- SHAP value calculation
- Visualization generation

**Layer 6: Presentation Layer**
- Report generation
- Figure creation
- Table formatting
- Executive summary

**Layer 7: Orchestration Layer**
- Pipeline coordination
- Configuration management
- Logging and monitoring
- Error handling

### 4.4.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Pipeline (main.py)                  │
│                   Orchestrates execution                     │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐   ┌─────▼──────┐   ┌────▼─────┐
   │   Data   │   │   Model    │   │ Evaluate │
   │ Loading  │──>│  Training  │──>│  Models  │
   └──────────┘   └────────────┘   └────┬─────┘
        │                │               │
   ┌────▼─────┐   ┌─────▼──────┐   ┌────▼─────┐
   │  Prepro- │   │  Ensemble  │   │Explain & │
   │  cessing │   │  Methods   │   │Visualize │
   └──────────┘   └────────────┘   └──────────┘
```

### 4.4.3 Module Organization

```
src/
├── data_loading.py      # Data access functions
├── preprocessing.py     # Preprocessing pipeline
├── tabular_models.py    # Individual model training
├── ensemble.py          # Ensemble methods
└── explainability.py    # SHAP and feature importance

scripts/
├── train_tabular.py     # Train individual models
├── train_ensemble.py    # Train ensemble models
├── evaluate_models.py   # Evaluation and visualization
└── run_full_pipeline.py # Complete automation

main.py                  # Entry point and orchestration
```

## 4.5 Data Pipeline Design

### 4.5.1 Data Flow

```
Raw CSV ──> Load ──> Validate ──> Encode ──> Scale ──> Split ──> Models
   │          │          │           │         │        │
   │          ├─> Error  ├─> Error   ├─> Log   ├─> Log ├─> Train/Test
   │          │          │           │         │        │
   └────> Log ───────────────────────────────────────────┘
```

### 4.5.2 Preprocessing Steps

**Step 1: Data Loading**
- Read CSV with pandas
- Validate column names and types
- Check for unexpected values
- Log dataset statistics

**Step 2: Missing Value Handling**
- Identify missing values (NaN, empty strings)
- Impute numerical features with median
- Impute categorical features with mode
- Document imputation decisions

**Step 3: Feature Engineering**
- Create binary dementia label (CDR > 0)
- Encode gender (M=1, F=0)
- Normalize education years
- Log transformations if needed

**Step 4: Feature Scaling**
- StandardScaler for numerical features
- Zero mean, unit variance transformation
- Fit on training set only
- Apply to both train and test

**Step 5: Data Splitting**
- Stratified 80/20 split
- Preserve class distribution
- Fixed random seed for reproducibility
- Validate split proportions

### 4.5.3 Feature Descriptions

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| Age | Numerical | Age in years | 18-96 |
| Gender | Categorical | M or F | Binary |
| EDUC | Numerical | Education years | 6-23 |
| MMSE | Numerical | Cognitive score | 0-30 |
| CDR | Categorical | Dementia rating | 0, 0.5, 1, 2, 3 |
| eTIV | Numerical | Intracranial volume | mm³ |
| nWBV | Numerical | Brain volume ratio | 0-1 |
| ASF | Numerical | Atlas scaling factor | Real |

## 4.6 Model Architecture Design

### 4.6.1 Logistic Regression Design

```python
LogisticRegression(
    penalty='l2',          # L2 regularization
    C=1.0,                 # Inverse regularization strength
    solver='lbfgs',        # Optimization algorithm
    max_iter=1000,         # Maximum iterations
    random_state=42        # Reproducibility
)
```

**Rationale**: Baseline model providing interpretable coefficients and probabilistic outputs

### 4.6.2 Random Forest Design

```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Unlimited tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples per leaf
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

**Rationale**: Robust ensemble handling non-linear relationships with built-in feature importance

### 4.6.3 Gradient Boosting Design

```python
GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinkage parameter
    max_depth=3,           # Tree complexity
    subsample=1.0,         # Sample fraction
    random_state=42        # Reproducibility
)
```

**Rationale**: Sequential boosting correcting predecessor errors, often achieving highest accuracy

### 4.6.4 Stacking Ensemble Design

```python
StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression(),
    cv=5,                  # Cross-validation folds
    stack_method='predict_proba',
    n_jobs=-1
)
```

**Rationale**: Meta-learning combining diverse base models for improved generalization

### 4.6.5 Voting Ensemble Design

```python
VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    voting='soft',         # Use probability averages
    n_jobs=-1
)
```

**Rationale**: Simple but effective ensemble through probability averaging

## 4.7 User Interface Design

### 4.7.1 Command-Line Interface

**Primary Command**:
```bash
python main.py --mode full
```

**Execution Modes**:
- `--mode full`: Complete pipeline (default)
- `--mode tabular`: Train only tabular models
- `--mode validate`: Validate environment only

**Optional Parameters**:
- `--config PATH`: Custom configuration file
- `--verbose`: Detailed logging
- `--help`: Display usage information

### 4.7.2 Configuration Interface

YAML configuration file (`scripts/config.yaml`):

```yaml
data:
  raw_path: "data/raw/clinical.csv"
  processed_path: "data/processed/"
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

### 4.7.3 Output Interface

**Console Output**:
```
================================================================================
DEMENTIA PREDICTION ML PIPELINE
================================================================================
[INFO] Loading data from data/raw/clinical.csv...
[INFO] Dataset: 416 samples, 8 features
[INFO] Training Logistic Regression...
[INFO] Training Random Forest...
[INFO] Training Gradient Boosting...
[INFO] Training Stacking Ensemble...
[INFO] Training Voting Ensemble...
[INFO] Evaluating models...
[INFO] Generating visualizations...
[INFO] Pipeline complete!
================================================================================
```

**File Outputs**:
```
outputs/
├── figures/
│   ├── roc_curves.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_stacking_ensemble.png
│   └── confusion_matrix_voting_ensemble.png
├── tables/
│   └── model_performance.csv
└── EXECUTIVE_SUMMARY.txt

models/
├── logistic_regression.pkl
├── random_forest.pkl
├── gradient_boosting.pkl
├── stacking_ensemble.pkl
├── voting_ensemble.pkl
└── preprocessor.pkl
```

### Summary

This chapter detailed comprehensive functional and non-functional requirements derived from literature, dataset analysis, and best practices. The system architecture follows layered design with clear separation of concerns, enabling modularity, testability, and maintainability. Data pipeline design ensures robust preprocessing with proper validation and error handling. Model architectures balance performance and interpretability, while the user interface provides simplicity through command-line execution and configuration files.

---

*End of Chapter 4*
