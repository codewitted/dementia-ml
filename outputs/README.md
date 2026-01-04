# Outputs Directory

This directory contains all generated outputs from model training and evaluation.

## Directory Structure

```
outputs/
├── README.md                      # This file
├── figures/                       # Visualizations and plots
│   ├── roc_curves.png            # ROC curve comparison
│   ├── confusion_matrix_*.png    # Confusion matrices
│   ├── shap_summary_*.png        # SHAP visualizations
│   └── metrics_comparison.png    # Performance comparison
├── tables/                        # Performance metrics
│   ├── model_performance.csv     # Main results table
│   ├── model_performance.tex     # LaTeX formatted table
│   ├── benchmark_comparison.csv  # Literature comparison
│   └── feature_importance_summary.csv
├── EXECUTIVE_SUMMARY.txt         # Study summary
├── statistical_tests.txt         # Statistical significance tests
└── ensemble_results.csv          # Ensemble performance

```

## Output Files

### Figures

All figures are generated in both PNG (web/presentation) and PDF (publication) formats at 300 DPI.

#### `roc_curves.png` / `roc_curves.pdf`
- ROC curves for all models
- Includes AUC scores
- Distinguishes base models vs. ensembles
- Publication-ready format

#### `confusion_matrix_*.png`
- Confusion matrices for top performing models
- Shows counts and percentages
- Heatmap visualization

#### `shap_summary_*.png`
- SHAP value visualizations
- Feature importance rankings
- Both dot and bar plots

#### `metrics_comparison.png`
- Comprehensive metric comparison across all models
- Includes Accuracy, Precision, Recall, F1, AUC-ROC, Specificity
- Bar chart format

### Tables

#### `model_performance.csv`
Comprehensive performance metrics for all models:
- Model name
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Specificity

**Format**: CSV for easy import into Excel, Python, R

#### `model_performance.tex`
LaTeX formatted table for direct inclusion in dissertation/papers.

#### `benchmark_comparison.csv`
Comparison with published literature:
- Study reference
- Method used
- Dataset
- Performance metrics
- Notes

#### `feature_importance_summary.csv`
Feature importance across different models:
- Feature names
- SHAP importance values
- Traditional feature importance
- Averaged rankings

### Reports

#### `EXECUTIVE_SUMMARY.txt`
High-level summary including:
- Study objectives
- Methods used
- Key findings
- Best performing models
- Clinical implications
- Reproducibility statement

#### `statistical_tests.txt`
Results of statistical significance tests:
- McNemar's test results
- P-values
- Interpretation

#### `ensemble_results.csv`
Specific results for ensemble models:
- Stacking ensemble metrics
- Voting ensemble metrics
- Comparison with base models

## Generating Outputs

### From Notebooks

Run notebooks in order (especially notebook 06):
```bash
jupyter lab
# Run: 06_Results_and_Reporting.ipynb
```

### From Scripts

```bash
python scripts/evaluate_models.py --config scripts/config.yaml
```

## Using Outputs

### For Dissertation/Thesis

1. **Tables**: Use `.tex` files or copy from `.csv`
2. **Figures**: Use PDF versions for LaTeX documents
3. **Summary**: Include in methodology/results sections

### For Presentations

1. Use PNG versions of figures
2. Extract key metrics from CSV files
3. Highlight best performing models

### For Publications

1. All figures are 300 DPI publication quality
2. Tables formatted for academic journals
3. Statistical tests documented

## Customization

To customize output generation, edit:
- `scripts/config.yaml` - Output directories
- Notebook 06 - Visualization styles, metrics
- `scripts/evaluate_models.py` - Evaluation parameters

## File Sizes

Expected file sizes:
- Figures: 200KB - 2MB each (PNG)
- Tables: 1-10KB (CSV)
- Reports: 1-5KB (TXT)
- Total: ~10-50MB

## Note on Version Control

Output files are excluded from git (see `.gitignore`) as they are:
- Generated files (reproducible)
- Potentially large
- Subject to frequent changes

To share outputs:
- Use releases on GitHub
- Upload to research data repositories
- Include in supplementary materials

## Validation

Verify outputs are correct:

```bash
# Check files exist
ls outputs/figures/*.png
ls outputs/tables/*.csv

# Validate CSV format
python -c "import pandas as pd; print(pd.read_csv('outputs/tables/model_performance.csv'))"
```

## Archiving

For long-term storage:

```bash
# Create archive
tar -czf outputs_archive_$(date +%Y%m%d).tar.gz outputs/

# Or zip
zip -r outputs_archive_$(date +%Y%m%d).zip outputs/
```

## Questions?

If outputs are not generated as expected:
1. Check all models are trained (`models/` directory)
2. Verify data is loaded correctly
3. Review script/notebook output for errors
4. See [QUICKSTART.md](../QUICKSTART.md) for troubleshooting

---

Last updated: 2024-01-04
