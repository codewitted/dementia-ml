# Quick Start Guide

This guide will help you get the dementia-ml project running in under 15 minutes.

## Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- 8GB RAM minimum
- Internet connection for data download

## Step 1: Clone Repository

```bash
git clone https://github.com/codewitted/dementia-ml.git
cd dementia-ml
```

## Step 2: Setup Environment

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

### Option B: pip + venv

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Download Dataset

1. Visit OASIS website or Kaggle:
   - https://www.oasis-brains.org/
   - https://www.kaggle.com/datasets/ninadaithal/imagesoasis

2. Download and extract to:
   ```
   data/raw/clinical.csv
   data/raw/Non Demented/
   data/raw/Mild Dementia/
   ```

**Note**: Data files are large (~1-5GB) and not included in repository.

## Step 4: Run Workflow

### Option 1: Interactive Notebooks

```bash
jupyter lab
```

Then run notebooks in order (00 → 06).

### Option 2: Command-Line Scripts

```bash
# Train all models
python scripts/train_tabular.py
python scripts/train_ensemble.py

# Evaluate and generate reports
python scripts/evaluate_models.py
```

## Step 5: View Results

Results are saved to:
- `outputs/tables/` - Performance metrics (CSV, LaTeX)
- `outputs/figures/` - Visualizations (PNG, PDF)
- `models/` - Trained models

## Troubleshooting

### "Module not found" errors
```bash
# Ensure environment is activated
conda activate ad-ensemble

# Reinstall dependencies
conda env update -f environment.yml
```

### "Data file not found" errors
- Check data is in `data/raw/` directory
- Verify file names match config.yaml
- See `data/README_data.md` for details

### GPU not detected (optional)
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

CNN training will use CPU if GPU is not available (slower but functional).

## Running Tests

```bash
# Verify installation
python -m pytest tests/ -v
```

## Next Steps

- Read full documentation in [README.md](README.md)
- Explore notebooks for detailed explanations
- Customize `scripts/config.yaml` for your needs
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Getting Help

- **Issues**: https://github.com/codewitted/dementia-ml/issues
- **Documentation**: See README.md and notebook comments

---

**Estimated Time**: 10-15 minutes (excluding data download)
