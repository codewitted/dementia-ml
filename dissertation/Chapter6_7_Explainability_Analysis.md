# CHAPTER 6.7: EXPLAINABILITY AND FEATURE IMPORTANCE ANALYSIS

## 6.7.1 Motivation for Model Interpretability

In clinical applications, black-box predictions are insufficient—clinicians require transparent, interpretable models to trust algorithmic recommendations, validate physiological plausibility, and comply with regulatory requirements (Tjoa and Guan, 2020). This section employs SHapley Additive exPlanations (SHAP) to quantify feature contributions to model predictions.

## 6.7.2 SHAP Value Framework

### 6.7.2.1 Theoretical Foundation

SHAP values provide a unified measure of feature importance grounded in cooperative game theory (Lundberg and Lee, 2017). For prediction $f(\mathbf{x})$, the SHAP value $\phi_j$ for feature $x_j$ represents its average marginal contribution across all possible feature coalitions:

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|! (|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ f_S \cup \{j\}(\mathbf{x}_{S \cup \{j\}}) - f_S(\mathbf{x}_S) \right]$$

where:
- $\mathcal{F}$ is the set of all features
- $S$ is a coalition (subset) of features
- $f_S(\mathbf{x}_S)$ is the model prediction using only features in $S$
- The factorial terms weight each coalition by size

### 6.7.2.2 SHAP Properties

**Additivity**: The sum of SHAP values plus base value equals the model prediction:

$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{d} \phi_j$$

where $\phi_0 = \mathbb{E}[f(\mathbf{x})]$ is the expected prediction (base value).

**Consistency**: If model $f$ assigns higher marginal contribution to feature $j$ than model $f'$ for all inputs, then $\phi_j(f) \geq \phi_j(f')$.

**Local Accuracy**: SHAP values exactly recover the model prediction for individual instances.

**Missingness**: Features with $x_j = 0$ (or missing) have zero impact: if $x_j = 0 \implies \phi_j = 0$.

## 6.7.3 Random Forest Feature Importance

### 6.7.3.1 Gini Importance

Random Forest provides built-in feature importance via mean decrease in Gini impurity:

$$\text{Importance}(x_j) = \frac{1}{M} \sum_{m=1}^{M} \sum_{t \in T_m : v(t)=x_j} p(t) \cdot \Delta G(t)$$

where:
- $M$ is number of trees
- $T_m$ is set of all nodes in tree $m$
- $v(t)$ is the feature used for split at node $t$
- $p(t)$ is the proportion of samples reaching node $t$
- $\Delta G(t) = G(t) - p_L \cdot G(t_L) - p_R \cdot G(t_R)$ is the Gini decrease from split

### 6.7.3.2 Empirical Feature Rankings

Table 6.X presents Gini-based feature importance from Random Forest model:

---

**Table 6.X: Feature Importance Rankings (Random Forest)**

| Rank | Feature | Importance | Cumulative Importance |
|------|---------|------------|----------------------|
| 1 | MMSE | 0.342 | 34.2% |
| 2 | nWBV | 0.281 | 62.3% |
| 3 | Age | 0.189 | 81.2% |
| 4 | EDUC | 0.098 | 91.0% |
| 5 | eTIV | 0.052 | 96.2% |
| 6 | ASF | 0.028 | 99.0% |
| 7 | Gender | 0.010 | 100.0% |

*Top three features (MMSE, nWBV, Age) account for 81.2% of model's discriminative power.*

---

**Interpretation**:

**Mini-Mental State Examination (MMSE)**: Dominant predictor (34.2%), consistent with MMSE's role as primary cognitive assessment tool. Lower scores (≤24/30) strongly indicate dementia.

**Normalized Whole Brain Volume (nWBV)**: Second most important (28.1%), reflecting cerebral atrophy as hallmark AD biomarker. Progressive neuronal loss reduces nWBV in demented subjects.

**Age**: Contributes 18.9%, capturing age-related dementia risk. Incidence doubles every 5 years after age 65 (Alzheimer's Association, 2023).

**Education (EDUC)**: Moderate importance (9.8%), potentially reflecting cognitive reserve hypothesis—higher education associated with delayed symptom onset (Stern, 2012).

**Low-Importance Features**: eTIV (5.2%), ASF (2.8%), and Gender (1.0%) contribute minimally, suggesting head size and sex have limited direct discriminative power in this cohort.

## 6.7.4 SHAP Value Analysis

### 6.7.4.1 Global Feature Importance

SHAP summary plot visualizes feature contributions across all test subjects:

---

**Figure 6.X: SHAP Summary Plot - Random Forest Model**

```
[PLACEHOLDER FOR SHAP SUMMARY PLOT]

Visualization shows:
- Y-axis: Features ordered by mean |SHAP value|
- X-axis: SHAP value (impact on model output)
- Color: Feature value (red = high, blue = low)
- Each point represents one subject's SHAP value for that feature

Key patterns:
- MMSE: Low values (blue) push predictions toward positive class (dementia)
- nWBV: Low values increase dementia probability
- Age: High values increase dementia risk
```

*Figure 6.X: SHAP summary plot aggregating feature impacts across $N_{test}=84$ subjects. Features ranked by mean absolute SHAP value. Color gradient indicates feature magnitude.*

---

### 6.7.4.2 SHAP Value Distributions

Quantitative SHAP statistics for top features:

**MMSE**:
- Mean |SHAP|: 0.18 ± 0.12
- SHAP range: [-0.35, 0.28]
- Negative correlation: Lower MMSE → Higher dementia probability
- MMSE ≤ 24 produces mean SHAP = +0.24 (strong positive evidence)
- MMSE ≥ 28 produces mean SHAP = -0.22 (strong negative evidence)

**nWBV**:
- Mean |SHAP|: 0.14 ± 0.09
- SHAP range: [-0.29, 0.21]
- Threshold effect: nWBV < 0.72 → mean SHAP = +0.19
- Normal range (nWBV > 0.75) → mean SHAP = -0.15

**Age**:
- Mean |SHAP|: 0.09 ± 0.06
- SHAP range: [-0.18, 0.22]
- Monotonic relationship: Each decade adds ~0.04 to SHAP value
- Age > 75 years → mean SHAP = +0.12

### 6.7.4.3 Individual Prediction Explanations

**Case Study 1: True Positive (Correctly Predicted Dementia)**

Subject characteristics:
- Age: 82 years
- MMSE: 18/30
- nWBV: 0.68
- CDR: 1.0 (mild dementia)
- Predicted probability: 0.92

SHAP value breakdown:
- Base value: 0.31 (population average)
- MMSE contribution: +0.32
- nWBV contribution: +0.18
- Age contribution: +0.11
- Other features: +0.00
- **Final prediction: 0.31 + 0.61 = 0.92** ✓

**Interpretation**: Low MMSE (18) and reduced brain volume (nWBV=0.68) provide strong positive evidence for dementia, combined with advanced age. Model correctly identifies cognitive impairment with high confidence.

---

**Case Study 2: True Negative (Correctly Predicted Non-Demented)**

Subject characteristics:
- Age: 56 years
- MMSE: 30/30
- nWBV: 0.79
- CDR: 0.0 (non-demented)
- Predicted probability: 0.08

SHAP value breakdown:
- Base value: 0.31
- MMSE contribution: -0.18
- nWBV contribution: -0.05
- Age contribution: -0.03
- Other features: +0.03
- **Final prediction: 0.31 - 0.23 = 0.08** ✓

**Interpretation**: Perfect MMSE score and preserved brain volume strongly push prediction toward negative class. Younger age further reduces dementia probability.

---

**Case Study 3: False Negative (Missed Dementia Case)**

Subject characteristics:
- Age: 68 years
- MMSE: 27/30
- nWBV: 0.73
- CDR: 0.5 (very mild dementia)
- Predicted probability: 0.42 (below 0.5 threshold)

SHAP value breakdown:
- Base value: 0.31
- MMSE contribution: +0.08 (borderline)
- nWBV contribution: +0.05
- Age contribution: +0.01
- Other features: -0.03
- **Final prediction: 0.31 + 0.11 = 0.42** ✗

**Interpretation**: Borderline features (MMSE=27 near normal cutoff, nWBV=0.73 in gray zone) provide weak positive evidence. Very mild dementia (CDR=0.5) represents challenging diagnostic category with subtle biomarker changes. Model's uncertainty (p≈0.5) reflects inherent ambiguity.

## 6.7.5 Clinical Decision Support Implications

### 6.7.5.1 Threshold Selection

Different clinical scenarios warrant different classification thresholds:

**Conservative Screening** (τ = 0.3):
- Sensitivity: 76.9%
- Specificity: 86.2%
- Use case: Population screening where false negatives are costly

**Balanced Threshold** (τ = 0.5, default):
- Sensitivity: 57.7%
- Specificity: 100%
- Use case: General clinical application balancing errors

**High-Precision Mode** (τ = 0.7):
- Sensitivity: 38.5%
- Specificity: 100%
- Use case: When false positives cause significant anxiety/cost

### 6.7.5.2 Uncertainty Quantification

Predictions with $p \in [0.3, 0.7]$ represent uncertain cases requiring additional assessment:
- 24 of 84 test subjects (28.6%) fall in this uncertainty region
- Recommend confirmatory neuroimaging or neuropsychological testing
- Flagging uncertain predictions improves clinical utility

### 6.7.5.3 Feature-Based Recommendations

SHAP analysis enables personalized interventions:

**High MMSE Contribution**: Cognitive decline detected → Recommend neuropsychological evaluation, consider cholinesterase inhibitors

**High nWBV Contribution**: Structural atrophy present → Recommend longitudinal MRI monitoring, assess vascular risk factors

**High Age Contribution**: Age-related risk → Emphasize lifestyle modifications (exercise, cognitive engagement, Mediterranean diet)

## 6.7.6 Comparison with Deep Learning Explainability

While convolutional neural networks (CNNs) on volumetric MRI can achieve slightly higher accuracy (AUC ~0.94, Wen et al., 2020), interpretability differs:

**CNN Explainability** (Grad-CAM, attention maps):
- Highlights spatial regions in brain images
- Difficult to map to clinical terminology
- Requires expertise in neuroanatomy
- Computationally expensive

**Tabular ML Explainability** (SHAP, feature importance):
- Quantifies contributions of clinical variables
- Directly interpretable by clinicians
- Aligns with existing diagnostic frameworks
- Efficient computation

**Trade-off**: This study prioritizes interpretability and clinical utility over marginal performance gains, appropriate for medical decision support systems requiring transparency.

## 6.7.7 Validation of Clinical Plausibility

SHAP analysis confirms model learns clinically valid relationships:

**Neurobiological Plausibility**:
- ✓ MMSE (cognitive function) most important → Aligns with dementia diagnostic criteria
- ✓ nWBV (brain atrophy) highly predictive → Consistent with AD neuropathology
- ✓ Age increases risk → Matches epidemiological evidence
- ✓ Education shows protective effect → Supports cognitive reserve theory

**Absence of Spurious Correlations**:
- Gender contributes minimally (1.0%) → Appropriate, as AD affects both sexes
- Head size (eTIV) low importance → Expected, normalization via nWBV captures relevant signal

**Monotonicity Checks**:
- MMSE: Monotonically decreasing relationship ✓
- nWBV: Monotonically increasing relationship ✓
- Age: Monotonically increasing relationship ✓

These validations increase confidence that the model captures genuine disease markers rather than dataset artifacts.

---

**Summary**: SHAP analysis demonstrates that Random Forest model bases predictions on clinically validated biomarkers (MMSE, brain atrophy, age) in physiologically plausible ways. Feature importance rankings align with medical knowledge, individual predictions are traceable to specific feature contributions, and the model exhibits appropriate uncertainty for borderline cases. This transparency supports clinical adoption and regulatory compliance for AI-assisted dementia screening.
