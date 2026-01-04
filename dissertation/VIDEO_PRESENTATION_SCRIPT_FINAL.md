# DISSERTATION VIDEO PRESENTATION SCRIPT
## Dementia Prediction using Machine Learning: A Comprehensive Multimodal Approach

**Duration:** 9-10 minutes  
**Tone:** Professional, confident, natural conversational flow  
**Delivery:** Clear, measured pace with appropriate pauses and emphasis

---

## OPENING (50 seconds)

Hello, my name is [Your Name], and I'm pleased to present my Master's dissertation on "Dementia Prediction using Machine Learning: A Comprehensive Multimodal Approach."

*[Brief pause, smile]*

Dementia is one of the most significant global health challenges we face today. More than fifty-five million people worldwide are living with dementia, and this number is projected to triple by twenty-fifty. The economic impact is staggering—over three hundred forty-five billion dollars annually in healthcare costs in the United States alone, according to the twenty twenty-three Alzheimer's Association report.

*[Pause]*

But beyond these numbers are real people—our parents, grandparents, friends—whose lives are profoundly affected by cognitive decline. Early detection is crucial. When we can identify dementia in its earliest stages, we open the door to timely interventions, better treatment planning, and ultimately, improved quality of life for patients and their families.

*[Pause]*

This research tackles a fundamental question: Can we develop a machine learning system that's not only accurate, but also practical, interpretable, and deployable in real-world clinical settings?

---

## PROJECT MOTIVATION AND SCOPE (80 seconds)

My research uses the OASIS dataset—the Open Access Series of Imaging Studies—which contains clinical and neuroimaging data from four hundred sixteen individuals. What makes OASIS particularly valuable is its open accessibility and comprehensive data: clinical dementia ratings, cognitive test scores, demographic information, and brain volume measurements derived from structural MRI scans.

*[Pause]*

I developed and evaluated five different machine learning approaches. Starting with Logistic Regression as an interpretable baseline, moving to Random Forest—an ensemble of decision trees—and Gradient Boosting, which sequentially corrects errors from previous models. Then I implemented two advanced ensemble techniques: Stacking, which uses a meta-learner to optimally combine predictions, and Voting, which averages probability outputs across models.

*[Pause]*

A key aspect of this work is its commitment to reproducibility and transparency. The entire codebase is publicly available on GitHub. Everything—from data preprocessing to model training to evaluation—can be executed with a single command. I've used fixed random seeds, version-controlled dependencies, and comprehensive documentation, which means any researcher can reproduce my exact results.

*[Pause]*

This isn't just good practice; it's essential for translating research into clinical reality. If we can't reproduce results, we can't trust them for patient care.

---

## METHODOLOGY (90 seconds)

Let me walk you through the methodological approach. I employed an iterative development framework, which proved invaluable given the inherent uncertainties in machine learning research. This allowed me to continuously refine the system based on experimental results and emerging insights.

*[Pause]*

The data pipeline follows rigorous best practices throughout. Missing values—and there are some in any real-world medical dataset—are handled through median imputation for numerical features and mode imputation for categorical variables. These methods are statistically robust and minimize bias.

*[Pause]*

All features are standardized to zero mean and unit variance. This might sound technical, but it's crucial—it ensures that variables measured on different scales contribute appropriately to the models. For example, age might range from twenty to ninety, while brain volume fractions range from zero point six to zero point eight. Without standardization, the models would be biased.

*[Pause]*

I used stratified train-test splitting with an eighty-twenty ratio, which preserves the natural distribution of dementia cases across both sets. This ensures unbiased performance evaluation.

*[Pause]*

For the ensemble methods, I implemented five-fold cross-validation to generate what we call "out-of-fold predictions." This sophisticated technique prevents the meta-learner from overfitting—essentially, we're ensuring the ensemble learns from predictions it hasn't seen during base model training.

*[Pause]*

Every hyperparameter choice was informed by literature review and validated through systematic experimentation. The random seed is set to forty-two throughout—yes, a Douglas Adams reference—guaranteeing exact reproducibility across different computing environments.

---

## RESULTS AND PERFORMANCE (110 seconds)

Now for the results, which I'm genuinely excited to share.

*[Show screen: Performance metrics table]*

The Random Forest model achieved an AUC-ROC of zero point nine zero four. For those less familiar with this metric, it measures the model's ability to discriminate between dementia and non-dementia cases across all possible classification thresholds. An AUC of point nine zero four is considered excellent—we're talking about a model that correctly ranks a random dementia patient higher than a random healthy individual ninety point four percent of the time.

*[Pause]*

But here's what's particularly striking: this model achieved perfect specificity—one hundred percent. That means zero false positives. Every person the model flagged as potentially having dementia actually had dementia. For screening applications, this is incredibly important. False positives create unnecessary anxiety for patients and burden healthcare systems with follow-up testing.

*[Show screen: Confusion matrix]*

Looking at the confusion matrix, we see fifty-eight true negatives—non-demented individuals correctly identified. Zero false positives. Eleven false negatives—these are dementia cases the model missed—and fifteen true positives. The sensitivity is fifty-eight percent, which might seem moderate, but coupled with perfect specificity, this represents an appropriate operating point for population screening.

*[Pause]*

Our ensemble methods—both Stacking and Voting—achieved AUC-ROC of zero point eight nine nine. Interestingly, they performed identically, which tells us that simple probability averaging works just as well as meta-learning when your base models are already well-optimized.

*[Show screen: Comparison with literature]*

When I compare these results with published work on the same dataset, the performance is highly competitive. Islam and Zhang in twenty eighteen reported zero point eight eight AUC. Duc and colleagues in twenty twenty achieved zero point nine zero using computationally expensive deep learning on raw MRI images. My Random Forest, using just seven engineered clinical features, matches or exceeds these benchmarks while training in three seconds on a standard laptop CPU versus their forty hours on high-end GPUs.

*[Pause]*

This demonstrates something important: classical machine learning, when properly engineered and rigorously evaluated, can be remarkably effective—and far more practical for real-world deployment.

---

## EXPLAINABILITY AND CLINICAL INSIGHTS (90 seconds)

One of my major focuses was interpretability, because in healthcare, we can't just deploy a "black box" that makes predictions without explanation.

*[Show screen: Feature importance chart]*

Using feature importance analysis, we see that the Mini-Mental State Examination score—our primary cognitive assessment—is the strongest predictor, accounting for thirty-four percent of the model's discriminative power. This makes perfect sense clinically. MMSE directly measures cognitive function, so low scores strongly indicate dementia.

*[Pause]*

The second most important feature is normalized whole brain volume at twenty-eight percent, which reflects brain atrophy—a hallmark of Alzheimer's disease. Age contributes nineteen percent, which aligns with epidemiological evidence that dementia risk doubles every five years after sixty-five.

*[Pause]*

*[Show screen: SHAP analysis]*

I also implemented SHAP analysis—Shapley Additive Explanations—which provides deeper interpretability. For individual patients, we can trace exactly how each feature contributed to their prediction. Low MMSE scores push the prediction toward dementia. Reduced brain volume increases dementia probability. Higher education shows a protective effect, consistent with what we call the cognitive reserve hypothesis.

*[Pause]*

This level of transparency isn't just academically interesting—it builds clinical trust. When a physician can see why the model made a specific prediction, they're more likely to trust and use the system. It also enables meaningful dialogue between the machine learning system and healthcare practitioners, which is essential for adoption.

---

## CHALLENGES AND LIMITATIONS (75 seconds)

I want to be transparent about the challenges I encountered, because I think this is valuable for future researchers.

*[Pause]*

Data access was a major hurdle. I initially wanted to use the ADNI dataset—the Alzheimer's Disease Neuroimaging Initiative—which is more comprehensive than OASIS. But the approval process takes four to eight weeks minimum, requires institutional sponsorship, and has restrictions for student researchers. Given dissertation timeline constraints, this wasn't feasible.

*[Pause]*

Computational resources were another challenge. I spent approximately one hundred eighty pounds on GPU rentals through RunPod, attempting to train deep learning models on three-dimensional MRI volumes. Even with a high-end RTX four-thousand-ninety GPU, I encountered out-of-memory errors and spot instance preemptions that lost hours of training progress.

*[Pause]*

After carefully analyzing the cost-benefit trade-off, I made a pragmatic decision to pivot to the tabular machine learning approach. This turned out to be the right choice—achieving comparable performance at a fraction of the computational cost and complexity.

*[Pause]*

The dissertation includes a comprehensive discussion of fourteen specific challenges—from data downloading issues with four-oh-three errors to class imbalance problems—along with concrete solutions for each. My goal is to help future researchers navigate these obstacles more efficiently than I did.

---

## PEER-REVIEWED COMPARISONS (55 seconds)

I conducted thorough comparisons with twelve peer-reviewed studies to properly contextualize my results.

*[Pause]*

Some interesting findings emerged. Basaia and colleagues demonstrated that models trained on ADNI data experienced a six to eight percent performance drop when tested on external datasets like OASIS. This domain shift is a critical issue for clinical deployment.

*[Pause]*

I also identified methodological issues in some published work. Tufail and colleagues reported ninety-nine percent accuracy using a Kaggle dataset, but my analysis suggests potential data leakage—the clinical dementia rating may have been inadvertently included as a feature. This highlights the importance of rigorous experimental design.

*[Pause]*

Liu and colleagues achieved ninety-six percent AUC using both MRI and PET imaging, but at a cost of five thousand dollars per patient compared to my hundred-dollar screening approach. Cost-effectiveness matters for population-scale deployment.

---

## FUTURE DIRECTIONS (45 seconds)

Looking forward, several research directions are particularly promising.

*[Pause]*

Multi-site validation is critical. I need to test this model on ADNI, NACC, and international cohorts to assess true generalization capability. Based on Basaia's work, I expect a six to eight percent performance degradation, but domain adaptation techniques could help.

*[Pause]*

Longitudinal modeling is another key direction. Ju and colleagues showed that incorporating temporal data—tracking how cognitive scores and brain volumes change over time—improves AUC by approximately five to seven percent and enables prediction of future conversion from mild cognitive impairment to dementia.

*[Pause]*

Ultimately, prospective clinical trials are needed to validate real-world deployment. That means partnering with hospitals, integrating with electronic health record systems, and demonstrating value in actual clinical practice.

---

## CLOSING (40 seconds)

To conclude, this dissertation demonstrates that machine learning can effectively support clinical decision-making for dementia screening at population scale.

*[Pause]*

The Random Forest model achieves state-of-the-art performance—zero point nine zero four AUC, perfect specificity—while maintaining interpretability, reproducibility, and practical deployability. The complete open-source implementation ensures that these findings can be independently validated and built upon by the research community.

*[Pause]*

Beyond the technical achievements, I hope this work contributes to the broader goal of developing trustworthy, explainable, and equitable healthcare machine learning systems that can genuinely improve patient outcomes.

*[Pause]*

Thank you for your attention. I'm happy to answer any questions.

*[End screen: Contact information and GitHub repository link]*

---

## TECHNICAL DEMONSTRATION SEGMENT (Optional - 2 minutes)

*[If time permits and demonstration is required]*

Let me briefly demonstrate the system in action.

*[Screen share: Terminal]*

The entire pipeline executes with a single command: "python main dot p-y dash dash mode full"

*[Show execution]*

You'll see it loading the data... four hundred sixteen subjects... performing train-test split... eighty percent training, twenty percent testing...

Now it's training the Logistic Regression model... complete in less than a second.

Random Forest training... you can see it using all CPU cores for parallel processing... and that's done in about three seconds.

Gradient Boosting... Stacking Ensemble with five-fold cross-validation... Voting Ensemble...

*[Show output]*

The system automatically generates all visualizations—confusion matrices, ROC curves—and saves them in publication-ready format at three hundred DPI resolution. Performance metrics are exported to CSV for further analysis.

*[Show generated figures]*

Here's the ROC curve showing all five models. The Random Forest curve is closest to the top-left corner, confirming its superior performance.

And here are the confusion matrices, clearly labeled with true positives, true negatives, false positives, and false negatives.

*[Pause]*

The entire process, from raw data to publication-ready outputs, takes less than thirty seconds. This is reproducible research in action.

---

## PREPARATION NOTES FOR PRESENTER

**Pacing:**
- Speak at 120-130 words per minute (slightly slower than conversational)
- Use pauses for emphasis and to let complex concepts settle
- Vary intonation to maintain engagement
- Smile naturally, especially during opening and closing

**Screen Sharing:**
- Have all figures pre-loaded in separate tabs for quick switching
- Ensure text and numbers are clearly visible (zoom if needed)
- Practice transitions between slides smoothly

**Emphasis Points:**
- When stating key results (AUC 0.904, 100% specificity)
- When discussing reproducibility and open science
- When explaining clinical relevance and interpretability
- When being transparent about limitations

**Body Language (if camera on):**
- Maintain eye contact with camera
- Use hand gestures naturally when explaining concepts
- Sit up straight but relaxed
- Nod occasionally when transitioning between points

**Technical Terms:**
- Pronounce clearly: "AUC" as "A-U-C" or "area under curve"
- "SHAP" as "shap" (like "shape" but ending with "p")
- "OASIS" as "oh-ay-sis"
- "Shapley" as "shap-lee"

**Common Questions to Prepare For:**
1. Why not use deep learning given its superior reported performance?
2. How would you address the false negatives?
3. What are the ethical considerations for deployment?
4. How does class imbalance affect your results?
5. What would it take to deploy this in a hospital setting?

**Backup Responses:**
- Have specific performance numbers memorized (0.904, 86.9%, 100%)
- Know your top 3 features and their percentages (MMSE 34.2%, nWBV 28.1%, Age 18.9%)
- Recall at least 2-3 peer-reviewed study comparisons with numbers
- Remember key limitations (single-site validation, class imbalance, computational constraints)

---

**FINAL CHECKLIST:**

- [ ] Practice full presentation 2-3 times
- [ ] Time yourself (aim for 9-10 minutes)
- [ ] Test screen sharing and figure visibility
- [ ] Have backup slides ready if demo fails
- [ ] Ensure good audio quality (test microphone)
- [ ] Clear, well-lit background if camera on
- [ ] Close unnecessary applications to avoid notifications
- [ ] Have water nearby (but away from keyboard!)
- [ ] Relax, breathe, and remember—you know this material better than anyone

**You've got this!**
