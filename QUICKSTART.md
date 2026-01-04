# Quick Start Guide

Get the dementia prediction pipeline running in under 15 minutes.

## Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- 8GB RAM minimum

## Step 1: Clone Repository

```bash
git clone https://github.com/codewitted/dementia-ml.git
cd dementia-ml
```

## Step 2: Setup Environment

### Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

### pip Alternative

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Prepare Data

### Option A: Download OASIS Dataset

1. Visit https://www.oasis-brains.org/ or https://www.kaggle.com/datasets/ninadaithal/imagesoasis
2. Download and extract to `data/raw/clinical.csv`

### Option B: Generate Synthetic Data

```bash
python scripts/generate_realistic_oasis_data.py
```

## Step 4: Run Pipeline

```bash
python main.py --mode full
```

Or use the automated script:

```bash
python scripts/run_full_pipeline.py
```

## Step 5: View Results

Results are saved to:

- `outputs/tables/model_performance.csv` - Performance metrics
- `outputs/figures/roc_curves.png` - ROC curve visualization
- `outputs/figures/confusion_matrix_*.png` - Confusion matrices
- `outputs/EXECUTIVE_SUMMARY.txt` - Complete summary

## Troubleshooting

### Module not found errors

```bash
conda activate ad-ensemble
conda env update -f environment.yml
```

### Data file not found

Ensure data is in `data/raw/` directory. See `data/README_data.md` for details.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Next Steps

- Read [README.md](README.md) for complete documentation
- Review [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for reproduction guide
- See [VALIDATION.md](VALIDATION.md) for requirements checklist

---

**Estimated Time**: 10-15 minutes (excluding data download)
