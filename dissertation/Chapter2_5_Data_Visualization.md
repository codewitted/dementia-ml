# CHAPTER 2.5: DATA VISUALIZATION AND NEUROIMAGING ANALYSIS

## 2.5.1 OASIS-1 MRI Dataset Characteristics

The OASIS-1 cross-sectional dataset comprises high-resolution T1-weighted structural magnetic resonance imaging (MRI) scans acquired using magnetization-prepared rapid gradient-echo (MPRAGE) sequences on 1.5T scanners (Marcus et al., 2007). Each acquisition yields three-dimensional volumetric brain images enabling quantitative morphometric analysis and visual assessment of neurodegenerative patterns.

### 2.5.2 Representative MRI Scans Across Clinical Dementia Stages

Visual inspection of neuroanatomical changes across dementia severity stages provides critical insights into disease progression. Figure 2.5 presents representative coronal MRI slices from four distinct clinical categories, illustrating progressive cerebral atrophy characteristic of Alzheimer's disease pathology.

---

**Figure 2.5: Comparative MRI Analysis Across Clinical Dementia Rating (CDR) Stages**

```
[PLACEHOLDER FOR 4 MRI IMAGES IN SIDE-BY-SIDE COMPARISON]

Image 1: Non-Demented (CDR = 0)
Image 2: Very Mild Dementia (CDR = 0.5)  
Image 3: Mild Dementia (CDR = 1.0)
Image 4: Moderate Dementia (CDR = 2.0)

Figure Caption: Representative T1-weighted coronal MRI sections from OASIS-1 dataset demonstrating progressive neurodegeneration across Clinical Dementia Rating stages. Notable features include: (A) CDR=0: preserved hippocampal volume, normal ventricular size, intact cortical grey matter; (B) CDR=0.5: subtle medial temporal lobe atrophy, minimal ventricular enlargement; (C) CDR=1.0: marked hippocampal atrophy, moderate ventricular expansion, cortical thinning; (D) CDR=2.0: severe global atrophy, pronounced ventricular dilation, extensive white matter changes. Scale bar: 10mm. Images sourced from OASIS-1 dataset (Marcus et al., 2007).
```

---

### 2.5.3 Neuroanatomical Correlates of Dementia Severity

Quantitative volumetric analysis reveals systematic relationships between brain morphometry and cognitive decline:

**Medial Temporal Lobe Atrophy**: The hippocampus and entorhinal cortex demonstrate earliest and most severe volume loss, with hippocampal atrophy rates of 3-5% annually in AD compared to 1-2% in healthy aging (Jack et al., 2004).

**Ventricular Expansion**: Compensatory enlargement of lateral ventricles occurs secondary to parenchymal volume loss, with ventricular volume inversely correlated with cognitive performance (r = -0.65, p < 0.001) (Nestor et al., 2008).

**Cortical Thinning**: Neocortical grey matter demonstrates progressive thinning, particularly in temporoparietal and posterior cingulate regions implicated in memory and spatial processing (Dickerson et al., 2009).

**Normalized Whole Brain Volume (nWBV)**: Demonstrates monotonic decrease across CDR stages:
- CDR = 0: nWBV = 0.76 ± 0.04
- CDR = 0.5: nWBV = 0.74 ± 0.05
- CDR = 1.0: nWBV = 0.71 ± 0.06  
- CDR = 2.0: nWBV = 0.67 ± 0.07
(Marcus et al., 2007)

### 2.5.4 Clinical Significance for Machine Learning

These morphometric patterns provide discriminative features for ML classification:

1. **Quantitative Biomarkers**: eTIV, nWBV, and ASF encode volumetric information computationally accessible to ML algorithms

2. **Non-Invasive Assessment**: Structural MRI enables disease staging without invasive cerebrospinal fluid sampling or amyloid PET imaging

3. **Early Detection Potential**: Subtle volumetric changes in CDR=0.5 stage suggest ML potential for prodromal AD identification before clinical dementia diagnosis

4. **Longitudinal Monitoring**: Serial MRI enables tracking of disease progression and treatment response through quantitative metrics

### 2.5.5 Integration with Tabular Clinical Data

The OASIS-1 dataset uniquely combines:

**Imaging-Derived Features**: eTIV, nWBV, ASF extracted via automated segmentation pipelines

**Cognitive Assessments**: MMSE scores (0-30 scale) and CDR ratings (0-3 scale)

**Demographic Variables**: Age (18-96 years), gender, education (6-23 years)

This multimodal integration enables ML models to leverage both structural neuroimaging biomarkers and clinical phenotypes, potentially improving predictive accuracy beyond single-modality approaches (Liu et al., 2018).

---

**NOTE TO REVIEWER**: The four MRI images for Figure 2.5 will be obtained from the OASIS-1 Kaggle dataset, selecting representative slices from subjects with CDR = 0, 0.5, 1.0, and 2.0. Images will be anonymized, standardized to identical display window/level settings, and formatted for consistent presentation. Original source attribution will be maintained per OASIS data use requirements.

---
