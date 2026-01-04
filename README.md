# Alzheimer’s Early Dementia Detection: Interpretable Ensemble ML Pipeline

## Project Overview

This project develops an interpretable ensemble machine learning pipeline combining MRI and clinical tabular data to predict early dementia/Alzheimer’s (CDR > 0) using the OASIS-1 dataset.  
It benchmarks tabular (logistic regression, random forest, gradient boosting) and MRI slice CNNs, and fuses their predictions with late-stage stacking/ensembles. Explainability is delivered via SHAP (tabular) and GradCAM (imaging), delivering robust, clinician-auditable models.

## Workflow

1. **Data acquisition** (MRI slices + clinical/demographic spreadsheet)
2. **EDA & preprocessing** (data cleaning, one-hot, scaling, image preprocessing)
3. **Model training**  
   - Tabular: LR, RF, GBM  
   - Imaging: 2D-CNN (per slice, with subject-level pooling/fusion)
   - Ensemble: Meta-learner stacking  
4. **Evaluation** (Accuracy, ROC-AUC, F1, confusion, calibration)
5. **Explainability** (SHAP, GradCAM, joint feature importances)
6. **Reporting** (tables, plots, figures auto-generated)

## Getting Started

### Prerequisites
- Python ≥ 3.10, conda (recommended for GPU/Torch)
- OASIS dataset access ([Kaggle link](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) or original OASIS [site](https://oasis-brains.org))
- (Optional: CUDA 11/12-enabled GPU for CNNs)

### Setup
```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

### Data Preparation
- **Clinical CSV:** Place in `data/raw/`  
- **MRI images:** Place in `data/raw/Mild Dementia`, `data/raw/Non Demented`, etc (see Data README for structure).

### Pipeline Automation (Example)
```bash
# Run EDA and preprocessing
jupyter notebook notebooks/01_EDA_and_Preprocessing.ipynb

# Train tabular models
python scripts/train_tabular.py

# Train MRI CNN model (GPU recommended)
python scripts/train_cnn.py

# Run ensemble/stacking
python scripts/run_ensemble.py

# Run explainability/XAI notebooks
jupyter notebook notebooks/05_Explainability_And_XAI.ipynb
```

All main outputs (tables, plots, trained models) are saved in `/outputs/`.

### Reproducibility
- Fixed seeds, cross-validation splits, config-driven paths and parameters.
- Full pipeline runs in Docker via:
```bash
docker build -t ad-ensemble .
docker run --gpus all -v $(pwd)/data:/app/data ad-ensemble
```

### Structure

- `data/` → Dataset input (not pushed to GitHub for privacy, scripts provided)
- `notebooks/` → EDA, model, XAI (ready for examiner review)
- `src/` → Source code modules and utilities
- `scripts/` → Training/automation scripts
- `outputs/` → Results, plots, tables, model weights
- `tests/` → Unit/integration test files for code sanity

### Citation/Acknowledgement

Please cite OASIS:  
Marcus, D.S. et al. (2007) ‘Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults’, Journal of Cognitive Neuroscience, 19(9), pp. 1498–1507.  
**Code released for academic and examiner reproducibility only.**

---