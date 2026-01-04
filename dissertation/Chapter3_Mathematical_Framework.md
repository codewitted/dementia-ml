# CHAPTER 3.3: MATHEMATICAL FRAMEWORK FOR DEMENTIA PREDICTION

## 3.3.1 Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ denote the OASIS dataset comprising $N=416$ subjects, where:

- $\mathbf{x}_i \in \mathbb{R}^{d}$ represents the feature vector for subject $i$, with $d$ features
- $y_i \in \{0, 1\}$ represents the binary dementia label (0 = non-demented, 1 = demented)

The objective is to learn a mapping $f: \mathbb{R}^{d} \rightarrow [0,1]$ that predicts the probability of dementia:

$$\hat{y}_i = f(\mathbf{x}_i; \theta)$$

where $\theta$ denotes model parameters optimized during training.

## 3.3.2 Feature Space Representation

Each subject is characterized by a feature vector:

$$\mathbf{x}_i = [\text{Age}_i, \text{Gender}_i, \text{EDUC}_i, \text{MMSE}_i, \text{eTIV}_i, \text{nWBV}_i, \text{ASF}_i]^T$$

where:
- $\text{Age}_i \in [18, 96]$ (years)
- $\text{Gender}_i \in \{0, 1\}$ (0=Female, 1=Male)
- $\text{EDUC}_i \in [6, 23]$ (years of education)
- $\text{MMSE}_i \in [0, 30]$ (Mini-Mental State Examination score)
- $\text{eTIV}_i \in \mathbb{R}^+$ (estimated total intracranial volume, mm³)
- $\text{nWBV}_i \in [0, 1]$ (normalized whole brain volume)
- $\text{ASF}_i \in \mathbb{R}^+$ (atlas scaling factor)

## 3.3.3 Data Preprocessing Pipeline

### 3.3.3.1 Missing Value Imputation

For incomplete observations, median imputation is applied to numerical features:

$$\tilde{x}_{ij} = \begin{cases} 
x_{ij} & \text{if } x_{ij} \text{ is observed} \\
\text{median}(\{x_{kj} : x_{kj} \text{ is observed}\}) & \text{otherwise}
\end{cases}$$

where $x_{ij}$ denotes feature $j$ for subject $i$.

### 3.3.3.2 Feature Standardization

To ensure zero mean and unit variance, features are standardized:

$$z_{ij} = \frac{\tilde{x}_{ij} - \mu_j}{\sigma_j}$$

where:
$$\mu_j = \frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{T}_{\text{train}}} \tilde{x}_{ij}$$

$$\sigma_j = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{T}_{\text{train}}} (\tilde{x}_{ij} - \mu_j)^2}$$

computed exclusively on training set $\mathcal{T}_{\text{train}}$ to prevent data leakage.

### 3.3.3.3 Stratified Train-Test Splitting

Dataset $\mathcal{D}$ is partitioned into training and test sets:

$$\mathcal{D} = \mathcal{T}_{\text{train}} \cup \mathcal{T}_{\text{test}}$$

with:
- $|\mathcal{T}_{\text{train}}| = 0.8N = 332$ subjects
- $|\mathcal{T}_{\text{test}}| = 0.2N = 84$ subjects

Stratification preserves class proportions:

$$\frac{|\{i \in \mathcal{T}_{\text{train}} : y_i = c\}|}{|\mathcal{T}_{\text{train}}|} \approx \frac{|\{i \in \mathcal{D} : y_i = c\}|}{|\mathcal{D}|}, \quad \forall c \in \{0,1\}$$

## 3.3.4 Model Architectures

### 3.3.4.1 Logistic Regression

The logistic regression model computes class probability via the logistic sigmoid:

$$P(y_i=1|\mathbf{z}_i; \mathbf{w}, b) = \sigma(\mathbf{w}^T\mathbf{z}_i + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T\mathbf{z}_i + b))}$$

where:
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\mathbf{z}_i$ is the standardized feature vector

**Training Objective** (L2-regularized cross-entropy):

$$\min_{\mathbf{w}, b} \left\{ -\frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{T}_{\text{train}}} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 \right\}$$

with regularization parameter $\lambda = 1.0$.

### 3.3.4.2 Random Forest

Random Forest constructs an ensemble of $M=100$ decision trees:

$$f_{\text{RF}}(\mathbf{z}_i) = \frac{1}{M} \sum_{m=1}^{M} h_m(\mathbf{z}_i)$$

where each tree $h_m$ is trained on a bootstrap sample:

$$\mathcal{B}_m = \{(\mathbf{z}_{j_1}, y_{j_1}), \ldots, (\mathbf{z}_{j_{N_{\text{train}}}}, y_{j_{N_{\text{train}}}})\}$$

with $j_k \sim \text{Uniform}(\{1, \ldots, N_{\text{train}}\})$ sampled with replacement.

**Node Splitting Criterion** (Gini impurity):

$$G(S) = 1 - \sum_{c \in \{0,1\}} p_c^2$$

where $p_c = \frac{|\{i \in S : y_i = c\}|}{|S|}$ is the proportion of class $c$ in node subset $S$.

At each node split, feature subset size $m_{\text{try}} = \lfloor \sqrt{d} \rfloor$ is randomly selected.

### 3.3.4.3 Gradient Boosting Machine

Gradient Boosting sequentially constructs additive models:

$$f_{\text{GB}}(\mathbf{z}_i) = \sum_{m=0}^{M} \nu h_m(\mathbf{z}_i)$$

where:
- $h_0(\mathbf{z}_i) = \text{argmin}_{\gamma} \sum_{i} L(y_i, \gamma)$ (initial constant model)
- $\nu = 0.1$ is the learning rate (shrinkage parameter)
- $M = 100$ is the number of boosting iterations

**Iterative Update Rule**:

For $m = 1, \ldots, M$:

1. Compute pseudo-residuals:
$$r_{im} = -\left[\frac{\partial L(y_i, f(\mathbf{z}_i))}{\partial f(\mathbf{z}_i)}\right]_{f=f_{m-1}}$$

2. Fit regression tree $h_m$ to pseudo-residuals $\{(\mathbf{z}_i, r_{im})\}$

3. Update model:
$$f_m(\mathbf{z}_i) = f_{m-1}(\mathbf{z}_i) + \nu h_m(\mathbf{z}_i)$$

For binary classification with log-loss:

$$L(y, f) = y \log(1 + e^{-f}) + (1-y) \log(1 + e^{f})$$

## 3.3.5 Ensemble Learning via Stacked Generalization

### 3.3.5.1 Base Learner Predictions

Let $f^{(k)}$ denote the $k$-th base learner, $k \in \{1, 2, 3\}$ corresponding to Logistic Regression, Random Forest, and Gradient Boosting respectively.

For each subject $i$, base learners generate class probability predictions:

$$\mathbf{p}_i = [p_i^{(1)}, p_i^{(2)}, p_i^{(3)}]^T$$

where $p_i^{(k)} = P(y_i=1|\mathbf{z}_i; f^{(k)})$.

### 3.3.5.2 Cross-Validated Meta-Features

To prevent overfitting, meta-features are generated via $K=5$-fold cross-validation:

1. Partition training set: $\mathcal{T}_{\text{train}} = \bigcup_{k=1}^{K} F_k$

2. For each fold $k$ and base learner $m$:
   - Train $f_m^{(-k)}$ on $\mathcal{T}_{\text{train}} \setminus F_k$
   - Generate out-of-fold predictions for $i \in F_k$:
   $$\tilde{p}_i^{(m)} = f_m^{(-k)}(\mathbf{z}_i)$$

3. Concatenate out-of-fold predictions to form meta-feature matrix:
$$\tilde{\mathbf{P}} = [\tilde{\mathbf{p}}_1, \ldots, \tilde{\mathbf{p}}_{N_{\text{train}}}]^T \in \mathbb{R}^{N_{\text{train}} \times 3}$$

### 3.3.5.3 Meta-Learner Training

A logistic regression meta-learner is trained on meta-features:

$$f_{\text{meta}}(\tilde{\mathbf{p}}_i) = \sigma(\mathbf{w}_{\text{meta}}^T \tilde{\mathbf{p}}_i + b_{\text{meta}})$$

optimizing:

$$\min_{\mathbf{w}_{\text{meta}}, b_{\text{meta}}} \left\{ -\sum_{i \in \mathcal{T}_{\text{train}}} \left[ y_i \log f_{\text{meta}}(\tilde{\mathbf{p}}_i) + (1-y_i) \log(1 - f_{\text{meta}}(\tilde{\mathbf{p}}_i)) \right] \right\}$$

### 3.3.5.4 Final Stacking Prediction

For test subject $j \in \mathcal{T}_{\text{test}}$:

1. Generate base learner predictions (trained on full $\mathcal{T}_{\text{train}}$):
$$\mathbf{p}_j = [f_1(\mathbf{z}_j), f_2(\mathbf{z}_j), f_3(\mathbf{z}_j)]^T$$

2. Meta-learner computes final prediction:
$$\hat{y}_j = f_{\text{meta}}(\mathbf{p}_j)$$

## 3.3.6 Voting Ensemble

Soft voting averages predicted probabilities:

$$f_{\text{vote}}(\mathbf{z}_i) = \frac{1}{3} \sum_{k=1}^{3} f^{(k)}(\mathbf{z}_i)$$

Hard voting uses majority class:

$$f_{\text{hard}}(\mathbf{z}_i) = \text{mode}\left(\{\mathbb{1}[f^{(k)}(\mathbf{z}_i) > 0.5]\}_{k=1}^{3}\right)$$

This project employs soft voting to leverage probability information.

## 3.3.7 Evaluation Metrics

Model performance is quantified using multiple metrics computed on $\mathcal{T}_{\text{test}}$:

### Confusion Matrix Elements:
- True Positives (TP): $|\{i : y_i=1, \hat{y}_i \geq 0.5\}|$
- True Negatives (TN): $|\{i : y_i=0, \hat{y}_i < 0.5\}|$
- False Positives (FP): $|\{i : y_i=0, \hat{y}_i \geq 0.5\}|$
- False Negatives (FN): $|\{i : y_i=1, \hat{y}_i < 0.5\}|$

### Performance Metrics:

**Accuracy**:
$$\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

**Precision** (Positive Predictive Value):
$$\text{Prec} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Recall** (Sensitivity, True Positive Rate):
$$\text{Rec} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Specificity** (True Negative Rate):
$$\text{Spec} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

**F1-Score** (Harmonic Mean):
$$F_1 = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$

**AUC-ROC** (Area Under Receiver Operating Characteristic Curve):

$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(x)) \, dx$$

where TPR (True Positive Rate) and FPR (False Positive Rate) are computed across classification thresholds $\tau \in [0,1]$:

$$\text{TPR}(\tau) = \frac{|\{i : y_i=1, \hat{y}_i \geq \tau\}|}{|\{i : y_i=1\}|}$$

$$\text{FPR}(\tau) = \frac{|\{i : y_i=0, \hat{y}_i \geq \tau\}|}{|\{i : y_i=0\}|}$$

## 3.3.8 Statistical Significance Testing

To compare model performance, McNemar's test assesses whether classification errors differ significantly:

**Test Statistic**:
$$\chi^2 = \frac{(n_{01} - n_{10})^2}{n_{01} + n_{10}}$$

where:
- $n_{01}$ = number of subjects correctly classified by model A but not model B
- $n_{10}$ = number of subjects correctly classified by model B but not model A

Under null hypothesis (models have equal error rates), $\chi^2 \sim \chi^2(1)$ with 1 degree of freedom. Reject null if $p < 0.05$.

---

This mathematical framework ensures rigorous, reproducible implementation of the dementia prediction pipeline, with explicit formulations enabling verification and extension by future researchers.
