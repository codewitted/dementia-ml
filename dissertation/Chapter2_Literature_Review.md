# CHAPTER 2: LITERATURE REVIEW

## 2.1 Dementia and Alzheimer's Disease

### 2.1.1 Medical Background

Dementia is a clinical syndrome characterized by acquired, progressive deterioration in cognitive function affecting multiple cognitive domains including memory, language, executive function, and visuospatial abilities (American Psychiatric Association, 2013). Alzheimer's disease (AD) accounts for 60-80% of dementia cases and is characterized by specific neuropathological features: extracellular amyloid-beta (Aβ) plaques and intracellular neurofibrillary tangles composed of hyperphosphorylated tau protein (Selkoe and Hardy, 2016).

The pathological cascade of AD typically begins decades before clinical symptoms manifest. The amyloid cascade hypothesis posits that accumulation of Aβ triggers a sequence of events including tau pathology, neuroinflammation, synaptic dysfunction, and ultimately neuronal death (Hardy and Selkoe, 2002). Modern biomarker research has identified a temporal sequence: Aβ deposition → tau pathology → neurodegeneration → cognitive decline (Jack et al., 2013).

### 2.1.2 Clinical Assessment

The Clinical Dementia Rating (CDR) scale is a widely used global assessment tool that evaluates cognition and function across six domains: memory, orientation, judgment and problem solving, community affairs, home and hobbies, and personal care (Morris, 1993). CDR scores range from 0 (no dementia) to 3 (severe dementia), with intermediate stages of 0.5 (questionable/very mild dementia), 1 (mild dementia), and 2 (moderate dementia).

The Mini-Mental State Examination (MMSE) is a 30-point questionnaire assessing various cognitive functions including orientation, attention, memory, language, and visual-spatial skills (Folstein et al., 1975). While widely used, MMSE has limitations including ceiling effects in highly educated individuals and floor effects in severely impaired patients (Creavin et al., 2016).

### 2.1.3 Neuroimaging Biomarkers

Structural MRI provides quantitative measures of brain atrophy, a hallmark of AD. Key measures include:

**eTIV (Estimated Total Intracranial Volume)**: Maximum brain size reached in early adulthood, serving as a normalization factor for brain volumes (Buckner et al., 2004).

**nWBV (Normalized Whole Brain Volume)**: Total brain parenchymal volume normalized by eTIV, decreases with AD progression due to neuronal loss and atrophy (Jack et al., 1997).

**ASF (Atlas Scaling Factor)**: Scaling factor used to normalize brain volumes, related to head size (Marcus et al., 2007).

Hippocampal atrophy is particularly pronounced in AD, as the medial temporal lobe is affected early in disease progression (Jack et al., 2010). Longitudinal studies demonstrate that brain atrophy rates can predict conversion from mild cognitive impairment to AD dementia (Jack et al., 2004).

## 2.2 Machine Learning in Healthcare

### 2.2.1 Evolution and Applications

Machine learning has transformed healthcare over the past decade, with applications ranging from medical image analysis to clinical decision support, drug discovery, and personalized medicine (Esteva et al., 2019; Topol, 2019). Deep learning models have achieved or exceeded human expert performance in several medical image interpretation tasks, including diabetic retinopathy detection, skin cancer classification, and chest X-ray analysis (Esteva et al., 2017; Gulshan et al., 2016; Rajpurkar et al., 2017).

The success of ML in healthcare stems from several factors:
- Availability of large-scale medical datasets (e.g., ImageNet for medical imaging)
- Advances in algorithms and computational infrastructure
- Ability to identify complex patterns imperceptible to humans
- Potential to improve diagnostic accuracy and efficiency

### 2.2.2 Challenges in Healthcare ML

Despite successes, healthcare ML faces unique challenges (Rajkomar et al., 2019):

**Data Quality and Availability**: Medical data is often noisy, incomplete, and heterogeneous. Privacy regulations (HIPAA, GDPR) restrict data sharing, limiting dataset sizes.

**Interpretability**: "Black box" models face resistance from clinicians who require understanding of decision-making processes for trust and regulatory compliance.

**Generalization**: Models trained on data from one institution may not generalize to different populations, equipment, or protocols.

**Validation**: Rigorous clinical validation requires prospective trials, which are time-consuming and expensive.

**Integration**: Successful deployment requires seamless integration into clinical workflows without disrupting existing practices.

### 2.2.3 Ensemble Learning

Ensemble methods combine multiple models to improve prediction performance and robustness. Key approaches include:

**Bagging** (Bootstrap Aggregating): Train multiple models on bootstrap samples and average predictions. Random Forest is a prominent bagging ensemble that combines decision trees with feature randomization (Breiman, 2001).

**Boosting**: Sequentially train models, with each focusing on examples misclassified by predecessors. Gradient Boosting Machines (Friedman, 2001) and XGBoost (Chen and Guestrin, 2016) are highly effective boosting algorithms.

**Stacking**: Train a meta-model to combine predictions from multiple base models, learning optimal weighting (Wolpert, 1992). Cross-validation prevents overfitting during meta-model training.

**Voting**: Combine predictions through simple or weighted majority voting. Effective when base models are diverse and make uncorrelated errors.

Ensemble methods have demonstrated superior performance across various healthcare prediction tasks, including disease diagnosis, mortality prediction, and treatment response modeling (Caruana et al., 2015).

## 2.3 Existing Dementia Prediction Systems

### 2.3.1 Traditional Machine Learning Approaches

**Islam and Zhang (2018)** applied ensemble classification methods to brain MRI data from the OASIS dataset, achieving 88% accuracy using Random Forest. Their approach involved feature extraction from MRI images and classification using multiple algorithms including SVM, decision trees, and ensemble methods.

**Long et al. (2017)** developed a multi-task learning framework for AD diagnosis using clinical and imaging data from ADNI (Alzheimer's Disease Neuroimaging Initiative). They achieved AUC-ROC of 0.89 for AD classification by jointly learning multiple related tasks.

**Moradi et al. (2015)** used Support Vector Machines with structural MRI features to predict conversion from MCI to AD, achieving 73% accuracy. They emphasized the importance of feature selection and dimensionality reduction for small medical datasets.

### 2.3.2 Deep Learning Approaches

**Wen et al. (2020)** proposed a 3D convolutional neural network (CNN) for AD classification using structural MRI, achieving AUC-ROC of 0.94 on ADNI data. Their architecture included multiple convolutional and pooling layers with batch normalization and dropout for regularization.

**Liu et al. (2018)** developed a multi-task deep learning framework combining MRI and PET imaging for AD diagnosis, achieving 91% accuracy. They demonstrated that multi-modal fusion improves performance over single-modality approaches.

**Suk et al. (2014)** applied deep learning to hierarchical feature representation learning from MRI and PET, achieving 95% accuracy for AD classification. However, their approach required extensive computational resources and large training datasets.

### 2.3.3 Hybrid and Ensemble Systems

**Duc et al. (2020)** developed a 3D deep learning ensemble for AD diagnosis, combining multiple CNN architectures with different depths and configurations. They achieved AUC-ROC of 0.90 on OASIS data, demonstrating that ensemble diversity improves robustness.

**Spasov et al. (2019)** combined shallow and deep learning approaches, using Random Forest for feature selection followed by CNN for classification. This hybrid approach achieved competitive performance while maintaining interpretability.

**Rathore et al. (2017)** proposed an ensemble framework combining multiple feature extraction methods (shape, texture, intensity) with ensemble classification, achieving 88% accuracy on ADNI data.

### 2.3.4 Explainable AI in Dementia Prediction

**Böhle et al. (2019)** emphasized the importance of model interpretability in medical AI, proposing layer-wise relevance propagation (LRP) to visualize which brain regions contribute to AD predictions.

**Tjoa and Guan (2020)** surveyed explainable AI methods in healthcare, highlighting SHAP (SHapley Additive exPlanations) as a unified framework for interpreting model predictions. SHAP provides consistent feature importance values based on game-theoretic principles (Lundberg and Lee, 2017).

**Lee et al. (2019)** applied attention mechanisms in deep learning models for AD classification, enabling visualization of important brain regions. This improved both performance and clinical interpretability.

## 2.4 Technical Review of ML Approaches

### 2.4.1 Logistic Regression

Logistic regression models the posterior probability of class membership via the logistic sigmoid function:

$$P(y=1|\mathbf{x}; \mathbf{w}, b) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T\mathbf{x} + b))}$$

where $\mathbf{w} \in \mathbb{R}^d$ represents feature weights, $b \in \mathbb{R}$ is the bias term, and $\mathbf{x} \in \mathbb{R}^d$ denotes the feature vector. The decision boundary is a hyperplane defined by $\mathbf{w}^T\mathbf{x} + b = 0$.

**Training Objective** (Maximum likelihood with L2 regularization):

$$\min_{\mathbf{w}, b} \left\{ -\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 \right\}$$

**Advantages**: Computationally efficient ($O(Nd)$ per iteration), probabilistically interpretable via odds ratios $\exp(w_j)$, convex optimization guarantees global optimum, provides confidence estimates through predicted probabilities.

**Limitations**: Linear decision boundary constrains expressiveness for non-linear patterns, susceptible to multicollinearity when features correlate, requires feature scaling for optimal convergence, limited representational capacity for complex interactions.

**Performance in Dementia Prediction**: Achieves AUC-ROC 0.80-0.85 on clinical datasets (Barnes et al., 2009), serving as interpretable baseline enabling coefficient-based feature importance analysis clinically relevant for understanding biomarker contributions.

### 2.4.2 Random Forest

Random Forest constructs an ensemble of $M$ decision trees via bootstrap aggregating with feature randomization (Breiman, 2001):

$$f_{\text{RF}}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} h_m(\mathbf{x})$$

where each tree $h_m$ is trained on a bootstrap sample $\mathcal{B}_m$ drawn with replacement from the training set. At each node split, a random subset of $m_{\text{try}} = \lfloor \sqrt{d} \rfloor$ features is considered, reducing inter-tree correlation and improving ensemble diversity.

**Node Splitting**: Gini impurity criterion selects optimal feature-threshold pairs:

$$G(S) = 1 - \sum_{c=1}^{C} p_c^2, \quad p_c = \frac{|\{i \in S : y_i = c\}|}{|S|}$$

**Feature Importance**: Computed via mean decrease in Gini impurity aggregated across all trees and splits, enabling quantification of each feature's discriminative power.

**Advantages**: Captures non-linear decision boundaries and feature interactions, inherently resistant to overfitting via averaging (variance reduction), handles missing values through surrogate splits, requires no feature scaling, provides robust out-of-bag error estimates, parallelizable training.

**Limitations**: Reduced interpretability compared to single trees (ensemble of hundreds of models), computationally demanding for large $M$ and deep trees, potential bias toward high-cardinality categorical features, limited extrapolation beyond training distribution.

**Dementia Prediction Performance**: Consistently achieves AUC-ROC 0.85-0.90 on neuroimaging datasets (Islam and Zhang, 2018), with feature importance analyses identifying MMSE, normalized whole brain volume (nWBV), and age as dominant predictors.

### 2.4.3 Gradient Boosting Machines

Gradient Boosting constructs an additive ensemble by sequentially fitting models to pseudo-residuals, implementing gradient descent in function space (Friedman, 2001):

$$f_M(\mathbf{x}) = \sum_{m=0}^{M} \nu h_m(\mathbf{x})$$

where $h_0$ is a constant initial model, $\nu \in (0,1]$ is the learning rate (shrinkage parameter), and each subsequent tree $h_m$ approximates the negative gradient of the loss function with respect to the current ensemble prediction.

**Iterative Training Algorithm**:

For $m = 1, \ldots, M$:
1. Compute pseudo-residuals: $r_{im} = -\frac{\partial L(y_i, f_{m-1}(\mathbf{x}_i))}{\partial f_{m-1}(\mathbf{x}_i)}$
2. Fit regression tree $h_m$ to targets $\{(\mathbf{x}_i, r_{im})\}_{i=1}^{N}$  
3. Update ensemble: $f_m(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \nu h_m(\mathbf{x})$

**Loss Function** (Binary cross-entropy):
$$L(y, f) = y \log(1 + e^{-f}) + (1-y) \log(1 + e^{f})$$

**Regularization Mechanisms**: Learning rate $\nu$ controls step size (smaller values require more iterations but improve generalization), tree depth constraint limits model complexity, subsampling introduces stochastic gradient boosting for additional variance reduction.

**Advantages**: State-of-the-art predictive accuracy on structured data, handles heterogeneous features naturally, robust to outliers via appropriate loss functions, built-in feature importance via gain metric, flexible framework accommodating custom loss functions.

**Limitations**: Prone to overfitting with insufficient regularization, sequential training prohibits parallelization across trees, sensitive to hyperparameter settings ($M$, $\nu$, max depth), computationally expensive for large datasets, requires careful tuning to prevent degradation with excessive boosting iterations.

**Hyperparameter Trade-offs**: Lower learning rate $\nu$ with higher iteration count $M$ typically yields better generalization (Hastie et al., 2009), max depth typically set to 3-8 (shallow trees act as weak learners suitable for boosting), early stopping via validation monitoring prevents overtraining.

**Clinical Performance**: Achieves AUC-ROC 0.87-0.92 on dementia prediction tasks, often matching or exceeding Random Forest with proper tuning (Moradi et al., 2015).
- Built-in feature importance
- Flexible loss functions
- Often achieves state-of-the-art performance

**Limitations**:
- Prone to overfitting without careful tuning
- Sensitive to hyperparameters
- Computationally intensive
- Less interpretable than simpler models
- Requires careful validation

XGBoost and LightGBM are optimized implementations achieving excellent performance in medical prediction tasks (Chen and Guestrin, 2016).

### 2.4.4 Ensemble Methods

**Stacking**: Trains a meta-learner on base model predictions, learning optimal combination weights. Requires careful cross-validation to prevent overfitting. Has shown superior performance when base models are diverse (Wolpert, 1992).

**Voting**: Combines predictions through majority vote (hard voting) or averaged probabilities (soft voting). Simple but effective when base models make uncorrelated errors. Voting ensembles are more robust but may not achieve maximum possible performance.

**Theoretical Foundation**: Ensemble methods work by reducing variance (bagging), bias (boosting), or both (stacking). The error decomposition framework shows that ensemble error is bounded by average base model error minus ensemble diversity (Krogh and Vedelsby, 1995).

## 2.5 Gaps and Limitations in Current Research

### 2.5.1 Reproducibility Crisis

Many published ML studies in healthcare lack sufficient detail for reproduction (Haibe-Kains et al., 2020). Common issues include:
- Missing implementation details
- Unreported hyperparameters
- Unavailable code and data
- Unclear preprocessing steps
- Inconsistent evaluation protocols

This reproducibility crisis hinders scientific progress and clinical translation of research findings.

### 2.5.2 Limited Comparative Studies

Most studies focus on demonstrating superiority of a specific approach without systematic comparison across multiple methods under identical conditions. This makes it difficult to:
- Assess relative strengths and weaknesses
- Understand when specific methods are appropriate
- Establish reliable performance benchmarks
- Identify best practices for specific tasks

### 2.5.3 Interpretability-Performance Tradeoff

High-performing deep learning models often lack interpretability, creating a barrier to clinical adoption. Conversely, interpretable models (e.g., logistic regression) may sacrifice performance. Few studies successfully balance interpretability and performance, particularly for complex medical tasks.

### 2.5.4 External Validation

Most studies report performance on internal test sets from the same dataset used for training. External validation on independent datasets from different institutions, populations, or imaging protocols is rare. This raises concerns about model generalization and real-world performance.

### 2.5.5 Clinical Integration

Limited research addresses practical deployment challenges including:
- Integration with electronic health records
- Real-time prediction latency
- User interface design
- Clinical workflow integration
- Cost-effectiveness analysis
- Regulatory compliance

### 2.5.6 Data Limitations

OASIS and ADNI datasets, while valuable, have limitations:
- Relatively small sample sizes (hundreds, not thousands)
- Selection bias (research volunteers may not represent general population)
- Missing data and incomplete follow-up
- Limited diversity in age, ethnicity, and socioeconomic status
- Cross-sectional vs. longitudinal limitations

## 2.6 Justification for Proposed Approach

This project addresses identified gaps through:

### 2.6.1 Comprehensive Reproducibility

**Complete open-source implementation**: All code publicly available on GitHub with comprehensive documentation, enabling exact reproduction of results.

**Fixed random seeds**: Ensures identical results across runs and different computing environments.

**Environment specification**: Conda environment file and pip requirements specify exact package versions.

**Configuration management**: All parameters in version-controlled configuration files, not hard-coded.

**Automated pipeline**: Single-command execution from data loading through final results.

### 2.6.2 Systematic Comparative Analysis

**Identical conditions**: All models trained and evaluated on identical train-test splits, using same preprocessing pipeline and evaluation metrics.

**Diverse algorithms**: Comparison spans linear (logistic regression), tree-based (Random Forest, Gradient Boosting), and ensemble methods (Stacking, Voting).

**Comprehensive metrics**: Multiple evaluation metrics (accuracy, precision, recall, F1-score, AUC-ROC, specificity) provide complete performance picture.

**Statistical testing**: Significance testing validates performance differences between models.

### 2.6.3 Balancing Performance and Interpretability

**Feature importance analysis**: All models provide feature importance rankings, revealing which clinical and imaging markers drive predictions.

**SHAP integration**: Model-agnostic explainability through SHAP values bridges black-box performance and interpretability.

**Visualization**: Clear, publication-ready visualizations communicate model behavior to clinical audiences.

**Clinical validation**: Feature importance compared against established clinical knowledge to verify biological plausibility.

### 2.6.4 Software Engineering Excellence

**Modular design**: Clear separation between data loading, preprocessing, model training, evaluation, and visualization components.

**Comprehensive testing**: Unit tests for core functionality ensure code correctness.

**Version control**: Git workflow with meaningful commits documents development process.

**Documentation**: Detailed README, code comments, and docstrings facilitate understanding and extension.

**Coding standards**: Consistent style, error handling, and logging promote maintainability.

### 2.6.5 Focus on Practical Utility

**Automated execution**: Complete pipeline runs with single command, reducing user burden.

**Publication-ready outputs**: High-resolution figures, formatted tables, and summary reports ready for dissertation inclusion.

**Multiple export formats**: Results available in CSV (for Excel), LaTeX (for academic papers), and text formats.

**Extensibility**: Modular architecture enables easy addition of new models, features, or datasets.

### 2.6.6 Building on Existing Work

This project leverages insights from published research:
- Uses proven effective algorithms (Random Forest, Gradient Boosting)
- Applies established preprocessing techniques (standardization, encoding)
- Follows evaluation best practices (stratified splitting, multiple metrics)
- Incorporates state-of-the-art explainability methods (SHAP)
- Benchmarks against published OASIS results for validation

### Summary

The literature review establishes that:
1. Dementia represents a major global health challenge requiring improved early detection
2. Machine learning shows promise for dementia prediction but faces challenges in reproducibility, interpretability, and clinical integration
3. Ensemble methods demonstrate superior performance across healthcare prediction tasks
4. Significant gaps exist in reproducibility, comparative analysis, and practical deployment
5. This project addresses these gaps through comprehensive implementation, systematic comparison, and emphasis on reproducibility and interpretability

The proposed approach builds on successful prior work while addressing critical limitations, contributing to both scientific knowledge and practical clinical tools for dementia prediction.

---

*End of Chapter 2*
