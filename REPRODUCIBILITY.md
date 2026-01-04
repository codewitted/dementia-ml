# Reproducibility Guide

This document provides complete instructions for reproducing the results of this dementia prediction study.

## 🎯 One-Click Execution

The simplest way to reproduce all results:

```bash
python main.py --mode full
```

This single command will:
1. ✅ Validate environment
2. ✅ Check data availability
3. ✅ Train all models
4. ✅ Generate evaluations
5. ✅ Create publication-ready outputs

**Expected runtime**: ~30 minutes (without GPU)

## 📋 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: Optional (for CNN training only)

### Software Dependencies

**Option 1: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

**Option 2: pip + venv**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Data Acquisition

### OASIS Dataset

1. **Visit**: https://www.oasis-brains.org/ or https://www.kaggle.com/datasets/ninadaithal/imagesoasis

2. **Download**:
   - Clinical data (CSV file)
   - MRI images (optional, for CNN models)

3. **Place files**:
   ```
   data/raw/clinical.csv
   data/raw/Non Demented/     # Optional
   data/raw/Mild Dementia/    # Optional
   ```

### Demo Mode (No Download Required)

If you don't have the dataset, use demo mode with synthetic data:

```bash
python main.py --mode demo
```

This creates synthetic data that mimics real clinical patterns for testing.

## 🔄 Execution Modes

### Full Pipeline (Recommended)
```bash
python main.py --mode full
```
Runs complete workflow: tabular models → ensembles → evaluation

### Tabular Only
```bash
python main.py --mode tabular
```
Faster execution, skips CNN and focuses on tabular models

### Quick Validation
```bash
python main.py --mode quick
```
Runs tests and validates imports (1-2 minutes)

### Demo Mode
```bash
python main.py --mode demo
```
Uses synthetic data, perfect for testing setup

## 📝 Step-by-Step Manual Execution

If you prefer to run each step individually:

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

### Step 4 (Optional): Train CNN
```bash
python scripts/train_cnn.py --config scripts/config.yaml
```

**Requirements**: MRI image data must be available

## 🤖 AI Agent Execution

For AI agents (Claude, ChatGPT, etc.):

```bash
python scripts/ai_agent_run.py --auto
```

This generates a machine-readable report: `AI_EXECUTION_REPORT.json`

## 🔍 Verification

### Verify Installation
```bash
python main.py --mode quick
```

### Run Tests
```bash
pytest tests/ -v
```

### Check Outputs
After execution, verify these files exist:
```bash
ls models/*.pkl                              # Trained models
ls outputs/tables/model_performance.csv      # Performance metrics
ls outputs/figures/roc_curves.png           # ROC curves
cat outputs/EXECUTIVE_SUMMARY.txt           # Summary report
```

## 📈 Expected Results

### Performance Ranges (OASIS Dataset)

| Model | Expected AUC-ROC | Expected Accuracy |
|-------|------------------|-------------------|
| Logistic Regression | 0.82 - 0.88 | 80% - 85% |
| Random Forest | 0.85 - 0.90 | 82% - 88% |
| Gradient Boosting | 0.86 - 0.91 | 83% - 89% |
| **Stacking Ensemble** | **0.88 - 0.93** | **85% - 92%** |
| Voting Ensemble | 0.87 - 0.92 | 84% - 90% |

**Note**: Exact values depend on:
- Dataset split (fixed seed: 42)
- Data quality
- Specific OASIS subset used

### Validation Criteria

✅ **Results are valid if**:
- All models achieve AUC-ROC > 0.80
- Ensemble models outperform individual models
- No significant warnings or errors during execution
- Output files are generated successfully

⚠️ **Investigate if**:
- Any model has AUC-ROC < 0.75
- Training fails with errors
- Output files are missing

## 🎨 Using Results in Dissertation

### Tables
```bash
# LaTeX format
cat outputs/tables/model_performance.tex

# CSV for Excel/Python
cat outputs/tables/model_performance.csv
```

### Figures
```bash
# High-resolution PNG (300 DPI)
outputs/figures/roc_curves.png

# PDF for LaTeX documents
outputs/figures/roc_curves.pdf
```

### Summary
```bash
cat outputs/EXECUTIVE_SUMMARY.txt
```

## 🌐 Web API (Optional)

Start prediction API server:

```bash
python api/app.py
```

Access:
- **Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Make predictions:
```bash
curl -X POST "http://localhost:8000/predict/stacking_ensemble" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 75,
       "EDUC": 12,
       "MMSE": 28,
       "eTIV": 1500,
       "nWBV": 0.7,
       "ASF": 1.2,
       "gender": "F"
     }'
```

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Verify environment is activated
conda activate ad-ensemble

# Reinstall dependencies
conda env update -f environment.yml --prune
```

**2. "Data file not found"**
```bash
# Use demo mode instead
python main.py --mode demo

# Or download OASIS dataset (see Data Acquisition section)
```

**3. "Out of memory" errors**
```bash
# Reduce batch size in config.yaml
# Or close other applications
```

**4. "CUDA not available" (for CNN)**
```bash
# This is OK - CNN will use CPU
# To use GPU, install pytorch with CUDA support
```

### Getting Help

1. Check `pipeline_execution.log` for detailed logs
2. Review `QUICKSTART.md` for setup issues
3. See `README.md` for comprehensive documentation
4. Open GitHub issue with error details

## ✅ Verification Checklist

Before submitting to examiners, verify:

- [ ] `main.py --mode full` runs without errors
- [ ] All output files are generated
- [ ] Performance metrics are within expected ranges
- [ ] Figures are publication-quality
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Documentation is complete

## 📦 Archiving Results

Create reproducibility package:

```bash
# Archive outputs
tar -czf results_$(date +%Y%m%d).tar.gz outputs/

# Archive everything except data
tar -czf dementia-ml_complete_$(date +%Y%m%d).tar.gz \
    --exclude='data/raw/*' \
    --exclude='models/*.pkl' \
    --exclude='models/*.pth' \
    --exclude='.git' \
    .
```

## 🎓 For External Examiners

Dear Examiner,

To reproduce the results presented in this dissertation:

1. **Quick Start** (recommended):
   ```bash
   git clone <repository-url>
   cd dementia-ml
   conda env create -f environment.yml
   conda activate ad-ensemble
   python main.py --mode demo  # Uses synthetic data
   ```

2. **With Real Data**:
   - Download OASIS dataset (see Data Acquisition)
   - Run: `python main.py --mode full`

3. **Expected Time**:
   - Demo mode: ~5 minutes
   - Full pipeline: ~30 minutes

4. **Verification**:
   - Check `outputs/EXECUTIVE_SUMMARY.txt`
   - Review `outputs/tables/model_performance.csv`
   - View `outputs/figures/roc_curves.png`

All results should match the dissertation within statistical variation (±2-3% due to computational differences).

## 📚 Additional Resources

- **Project Manifest**: `PROJECT_MANIFEST.json` (machine-readable metadata)
- **Validation Checklist**: `VALIDATION.md`
- **Quick Setup**: `QUICKSTART.md`
- **Full Documentation**: `README.md`

## 🔐 Data Privacy & Ethics

- All data is publicly available (OASIS)
- No patient identifiers included
- Synthetic demo data used for testing
- Results are aggregated and anonymized

## 📌 Citation

If using this repository for research:

```bibtex
@software{dementia_ml_2024,
  author = {[Your Name]},
  title = {Dementia Prediction using Multi-Modal Machine Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/codewitted/dementia-ml}
}
```

---

**Last Updated**: 2024-01-04  
**Version**: 1.0.0  
**Status**: Examiner-Ready & Reproducible ✅
