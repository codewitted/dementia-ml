# Dissertation Outputs

This directory contains publication-ready outputs for dissertation inclusion.

## Contents

### Tables
- `tables/model_performance.csv` - Complete performance metrics for all models

### Figures
- `figures/roc_curves.png` - ROC curve comparison (300 DPI)
- `figures/confusion_matrix_random_forest.png` - Best model confusion matrix
- `figures/confusion_matrix_stacking_ensemble.png` - Stacking ensemble confusion matrix
- `figures/confusion_matrix_voting_ensemble.png` - Voting ensemble confusion matrix

### Summary
- `EXECUTIVE_SUMMARY.txt` - Complete study summary with methodology and results

## Usage in Dissertation

### LaTeX

```latex
% Include figure
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{dissertation/figures/roc_curves.png}
\caption{ROC curves comparing model performance. Random Forest achieved highest AUC-ROC (0.904).}
\label{fig:roc_curves}
\end{figure}

% Include table from CSV
\input{dissertation/tables/model_performance_latex.tex}
```

### Microsoft Word

1. Insert → Picture → From File → Select PNG files
2. Insert → Object → Text from File → Select CSV for tables

## Regenerating Outputs

To regenerate these outputs:

```bash
python main.py --mode full
cp outputs/EXECUTIVE_SUMMARY.txt dissertation/
cp outputs/tables/*.csv dissertation/tables/
cp outputs/figures/*.png dissertation/figures/
```

## Citation

If using these results, cite:

```bibtex
@software{dementia_ml_2026,
  author = {[Your Name]},
  title = {Dementia Prediction using Multi-Modal Machine Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/[your-username]/dementia-ml}
}
```

---

**Generated**: January 2026
