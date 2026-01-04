# DISSERTATION VIDEO PRESENTATION SCRIPT
## Dementia Prediction using Multi-Modal Machine Learning Approaches

**Duration:** 8-10 minutes  
**Tone:** Professional, eloquent, confident  
**Delivery:** Clear, measured pace with appropriate pauses

---

## INTRODUCTION (60 seconds)

Good afternoon. I am [Your Name], and I am delighted to present my Master's dissertation titled "Dementia Prediction using Multi-Modal Machine Learning Approaches."

*[Pause]*

Dementia represents one of the most pressing global health challenges of our time, affecting over fifty-five million people worldwide. The burden extends far beyond medical implications—with global costs exceeding one point three trillion dollars annually—impacting families, communities, and healthcare systems across the globe.

*[Pause]*

Early detection is paramount. Yet traditional diagnostic approaches often identify dementia only after significant neurological damage has occurred. This research addresses a critical question: Can machine learning provide an objective, accessible, and effective tool for early dementia screening?

*[Pause]*

Today, I will demonstrate how a comprehensive machine learning system, leveraging publicly available clinical and neuroimaging data, can achieve state-of-the-art performance in dementia prediction—with profound implications for clinical practice.

---

## PROJECT OVERVIEW (90 seconds)

This dissertation presents the development and rigorous evaluation of a complete machine learning pipeline for early dementia detection, utilizing the OASIS dataset—the Open Access Series of Imaging Studies—a landmark collection from Washington University containing clinical dementia ratings and structural MRI measurements from four hundred sixteen subjects.

*[Pause]*

The research encompasses five distinct machine learning approaches: Logistic Regression, serving as our interpretable baseline; Random Forest, an ensemble of decision trees; Gradient Boosting Machine, employing sequential error correction; and two advanced ensemble methods—Stacking and Voting classifiers—that synthesize predictions from multiple models to enhance robustness and performance.

*[Pause]*

Critically, this project emphasizes not merely algorithmic performance, but comprehensive software engineering excellence. The entire system is openly available on GitHub, featuring modular architecture, automated execution, comprehensive testing, and complete reproducibility through fixed random seeds and version-controlled configurations.

*[Pause]*

This commitment to open science ensures that our findings can be validated, extended, and ultimately translated into clinical practice by the broader research community.

---

## METHODOLOGY AND APPROACH (90 seconds)

Our methodological approach follows rigorous scientific principles. We employed an iterative development framework, enabling continuous refinement based on experimental results and emerging insights—a critical advantage when navigating the inherent uncertainties of machine learning research.

*[Pause]*

The data pipeline implements best practices throughout. We handle missing values through statistically robust median imputation for numerical features and mode imputation for categorical variables. Features undergo standardization to zero mean and unit variance—essential for optimal algorithm performance. Critically, we employ stratified train-test splitting at an eighty-twenty ratio, preserving the natural class distribution and ensuring unbiased evaluation.

*[Pause]*

For ensemble methods, we implement five-fold cross-validation, generating out-of-fold predictions that prevent overfitting during meta-learner training. This sophisticated approach enables our stacking ensemble to learn optimal combination weights from diverse base models.

*[Pause]*

Each model was configured with carefully selected hyperparameters informed by literature review and validated through systematic experimentation. All stochastic operations utilize a fixed random seed of forty-two, guaranteeing exact reproducibility across executions and computing environments.

---

## RESULTS AND PERFORMANCE (120 seconds)

Now, let me share the results—the culmination of this comprehensive research effort.

*[Pause]*

The Random Forest model emerged as our top performer, achieving an exceptional AUC-ROC of zero point nine zero four. Even more remarkably, it attained perfect specificity—one hundred percent—meaning zero false positive predictions. This is particularly significant for screening applications, where false alarms create patient anxiety and unnecessary healthcare burden.

*[Pause]*

Our ensemble methods demonstrated robust, balanced performance. The Stacking Ensemble achieved an AUC-ROC of zero point eight nine nine, while the Voting Ensemble matched this performance exactly. These results validate the theoretical foundation of ensemble learning: diverse models, making uncorrelated errors, can be effectively combined to produce superior, more reliable predictions.

*[Pause]*

When we benchmark against published literature on the OASIS dataset, our results are highly competitive. Islam and Zhang reported zero point eight eight for Random Forest; Duc and colleagues achieved zero point nine zero using deep learning on raw MRI images. Our Random Forest, using only tabular clinical features, matches or exceeds these benchmarks—demonstrating that classical machine learning, properly engineered, remains remarkably effective.

*[Pause]*

Feature importance analysis reveals clinically meaningful patterns. The Mini-Mental State Examination score—our primary cognitive assessment—emerges as the most important predictor, accounting for twenty-eight percent of model importance. Normalized whole brain volume, reflecting brain atrophy, ranks second at twenty-two percent. Age, the primary demographic risk factor, contributes nineteen percent. These rankings align perfectly with established clinical knowledge, providing crucial validation of our approach.

*[Pause]*

Through SHAP analysis—SHapley Additive exPlanations—we provide model-agnostic interpretability. Low MMSE scores strongly predict dementia. Brain atrophy, indicated by reduced brain volume, increases dementia probability. Higher education demonstrates a protective effect, consistent with the cognitive reserve hypothesis. This explainability is not merely academic—it builds clinical trust and enables meaningful dialogue between machine learning systems and healthcare practitioners.

---

## TECHNICAL IMPLEMENTATION (75 seconds)

The technical implementation merits attention for its adherence to software engineering best practices and commitment to reproducibility.

*[Pause]*

The codebase is fully modular, with clear separation between data loading, preprocessing, model training, evaluation, and visualization components. This architecture enables independent development, testing, and extension of each module without affecting others.

*[Pause]*

We developed a comprehensive test suite with fifteen passing unit and integration tests, validating core functionality from data loading through metric calculation. Configuration management through YAML files externalizes all parameters, enabling experimentation without code modification.

*[Pause]*

The complete pipeline executes with a single command: "python main dot py dash dash mode full." Within approximately twenty minutes on standard hardware, it progresses from raw data through trained models to publication-ready outputs—including high-resolution figures, formatted performance tables, and comprehensive executive summaries.

*[Pause]*

All code, data processing scripts, trained models, and complete documentation reside in a public GitHub repository, embodying the principles of open science and reproducible research.

---

## IMPACT AND SIGNIFICANCE (60 seconds)

The significance of this work extends across multiple dimensions.

*[Pause]*

Clinically, we demonstrate that machine learning can effectively support early dementia detection using readily available data—standard clinical assessments and routine MRI measurements. The exceptional specificity of our best model makes it particularly suitable for population-level screening, where minimizing false alarms is paramount.

*[Pause]*

Scientifically, we contribute a rigorous comparative analysis of five distinct approaches under identical conditions—a systematic evaluation often absent in the literature. Our results provide clear guidance: ensemble methods offer robust performance, while classical Random Forest remains highly competitive when properly implemented.

*[Pause]*

Methodologically, we establish a framework for reproducible machine learning research in healthcare. Fixed random seeds, version-controlled configurations, comprehensive documentation, and public code availability address the reproducibility crisis that challenges contemporary computational research.

*[Pause]*

For the open-source community, we provide a production-ready tool that can be validated, extended, and adapted to new datasets and clinical contexts—accelerating future research and potential clinical translation.

---

## LIMITATIONS AND FUTURE WORK (60 seconds)

Intellectual honesty compels acknowledgment of limitations.

*[Pause]*

The OASIS dataset, while valuable, contains only four hundred sixteen subjects—limiting statistical power. It represents a cross-sectional snapshot, preventing longitudinal progression modeling. The dataset's demographic composition may not fully represent the global dementia population, raising important questions about generalization across diverse communities.

*[Pause]*

Our binary classification approach—demented versus non-demented—does not capture disease severity or progression. Moderate recall of fifty-seven point seven percent means some demented cases are missed, though this is acceptable for initial screening when paired with perfect specificity.

*[Pause]*

Future work beckons in multiple directions. Integration of deep learning for raw MRI image analysis could further enhance performance. Longitudinal modeling using OASIS-2 or ADNI datasets would enable progression prediction. External validation across independent datasets from different institutions and populations is essential for assessing true generalization.

*[Pause]*

Perhaps most importantly, real-world deployment requires seamless integration with electronic health records, development of clinician-friendly interfaces, and rigorous prospective clinical validation.

---

## CONCLUSION (45 seconds)

In conclusion, this dissertation demonstrates that machine learning can meaningfully contribute to addressing the global dementia crisis.

*[Pause]*

We have developed a comprehensive, reproducible system achieving state-of-the-art performance—zero point nine zero four AUC-ROC with perfect specificity—matching published benchmarks while maintaining interpretability and clinical relevance.

*[Pause]*

Through rigorous methodology, comprehensive evaluation, and commitment to open science, this work provides both immediate scientific contribution and a foundation for future research and clinical tools that may improve outcomes for millions of patients and families affected by dementia worldwide.

*[Pause]*

The code, documentation, and complete results are publicly available on GitHub at github dot com slash codewitted slash dementia dash ml.

*[Pause]*

Thank you for your attention. I welcome your questions and feedback.

---

## DELIVERY NOTES

**Pacing Guidelines:**
- Speak at approximately 140-160 words per minute
- Pause at marked *[Pause]* points for 2-3 seconds
- Emphasize key numbers (0.904, 100%, etc.)
- Maintain confident, professional tone throughout
- Allow natural variation in pace for emphasis

**Slide Timing Recommendations:**
1. Title slide during Introduction (60s)
2. Project overview diagram during Overview (90s)
3. Methodology flowchart during Methodology (90s)
4. ROC curves and confusion matrices during Results (120s)
5. Code structure and architecture during Implementation (75s)
6. Impact summary slide during Impact (60s)
7. Future work roadmap during Limitations (60s)
8. Final summary slide during Conclusion (45s)

**Total Script Duration:** ~600 seconds (10 minutes)  
**Recommended Recording Duration:** 8-10 minutes with natural delivery

**Technical Setup:**
- Clear audio recording environment
- Screen recording software (OBS Studio recommended)
- Prepared slides synchronized with script sections
- Practice delivery 2-3 times for smooth execution

---

**Script prepared for:**  
MSc Computer Science Dissertation  
Keele University  
January 2026

**Tone achieved:** Professional, eloquent, confident, academically rigorous  
**Structure:** Introduction → Overview → Methodology → Results → Implementation → Impact → Limitations → Conclusion
