# Reproducibility Guide

This document provides complete instructions for reproducing the results of this dementia prediction study.

## One-Click Execution

The simplest way to reproduce all results:

```bash
python main.py --mode full
```

This single command will:
1. Validate environment
2. Verify data availability
3. Train all models
4. Generate evaluations
5. Create publication-ready outputs

**Expected runtime**: ~15-30 minutes

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: Optional (for CNN training only)

### Software Dependencies

**Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

**pip Alternative**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Acquisition

### OASIS Dataset

1. **Visit**: https://www.oasis-brains.org/ or https://www.kaggle.com/datasets/ninadaithal/imagesoasis

2. **Download** clinical data (CSV file) and optionally MRI images

3. **Place files**:
   ```
   data/raw/clinical.csv
   data/raw/Non Demented/     # Optional
   data/raw/Mild Dementia/    # Optional
   ```

### Alternative: Generate Synthetic Data

If OASIS data is unavailable, generate statistically-matched synthetic data:

```bash
python scripts/generate_realistic_oasis_data.py
```

This creates data matching published OASIS-1 distributions (Marcus et al., 2007).

## Execution Modes

### Full Pipeline (Recommended)
```bash
python main.py --mode full
```

### Tabular Models Only
```bash
python main.py --mode tabular
```

### Environment Validation
```bash
python main.py --mode validate
```

## Step-by-Step Manual Execution

### Step 1: Train Tabular Models
```bash
python scripts/train_tabular.py --config scripts/config.yaml
```

**Outputs**:
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`
- `models/preprocessor.pkl`

### Step 2: Train Ensemble Models
```bash
python scripts/train_ensemble.py --config scripts/config.yaml
```

**Outputs**:
- `models/stacking_ensemble.pkl`
- `models/voting_ensemble.pkl`

### Step 3: Evaluate Models
```bash
python scripts/evaluate_models.py --config scripts/config.yaml
```

**Outputs**:
- `outputs/tables/model_performance.csv`
- `outputs/figures/roc_curves.png`
- `outputs/figures/confusion_matrix_*.png`

## Verification

### Verify Installation
```bash
python main.py --mode validate
```

### Run Tests
```bash
pytest tests/ -v
```

### Check Outputs
```bash
ls models/*.pkl                              # Trained models
ls outputs/tables/model_performance.csv      # Performance metrics
ls outputs/figures/roc_curves.png           # ROC curves
cat outputs/EXECUTIVE_SUMMARY.txt           # Summary report
```

## Expected Results

### Performance Ranges (OASIS Dataset)

| Model | Expected AUC-ROC | Expected Accuracy |
|-------|------------------|-------------------|
| Logistic Regression | 0.85 - 0.90 | 80% - 85% |
| Random Forest | 0.88 - 0.92 | 84% - 88% |
| Gradient Boosting | 0.87 - 0.91 | 83% - 88% |
| **Stacking Ensemble** | **0.89 - 0.93** | **85% - 90%** |
| Voting Ensemble | 0.88 - 0.92 | 84% - 89% |

### Validation Criteria

Results are valid if:
- All models achieve AUC-ROC > 0.80
- Ensemble models equal or outperform individual models
- No critical errors during execution
- Output files are generated successfully

## Using Results in Dissertation

### Tables
```bash
cat outputs/tables/model_performance.csv
```

### Figures
```
outputs/figures/roc_curves.png         # 300 DPI
outputs/figures/confusion_matrix_*.png # Publication-ready
```

### Summary
```bash
cat outputs/EXECUTIVE_SUMMARY.txt
```

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
conda activate ad-ensemble
conda env update -f environment.yml --prune
```

**2. Data file not found**
```bash
python scripts/generate_realistic_oasis_data.py
```

**3. Out of memory errors**
- Close other applications
- Reduce batch size in `scripts/config.yaml`

### Getting Help

1. Check `pipeline_execution.log` for detailed logs
2. Review `QUICKSTART.md` for setup issues
3. See `README.md` for comprehensive documentation

## Verification Checklist

Before submission, verify:

- [ ] `main.py --mode full` runs without errors
- [ ] All output files are generated
- [ ] Performance metrics are within expected ranges
- [ ] Figures are publication-quality
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Documentation is complete

## Citation

If using this repository for research:

```bibtex
@software{dementia_ml_2026,
  author = {Codewitted},
  title = {Dementia Prediction using Multi-Modal Machine Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/codewitted/dementia-ml}
}
```

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Reproducible
