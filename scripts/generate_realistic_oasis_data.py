#!/usr/bin/env python3
"""
Generate Realistic OASIS-1 Style Dataset for Dissertation Research.

This script creates a synthetic dataset that mimics the statistical properties
of the OASIS-1 cross-sectional MRI dataset based on published literature:

Reference:
Marcus, D.S., Wang, T.H., Parker, J., Csernansky, J.G., Morris, J.C., & Buckner, R.L. (2007).
Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, 
middle aged, nondemented, and demented older adults. Journal of Cognitive Neuroscience, 
19(9), 1498-1507.

The OASIS-1 dataset characteristics:
- 416 subjects aged 18-96
- 100 subjects over 60 with mild to moderate Alzheimer's Disease
- Clinical Dementia Rating (CDR): 0 (non-demented), 0.5 (very mild), 1 (mild), 2 (moderate)
- Key features: Age, EDUC (education years), SES, MMSE, CDR, eTIV, nWBV, ASF

This synthetic data is suitable for ML model development and evaluation,
demonstrating the complete pipeline functionality.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def generate_oasis1_clinical_data(n_samples: int = 416, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OASIS-1 style clinical data based on published statistics.
    
    Parameters from literature:
    - Age: 18-96 years (with focus on older adults for dementia studies)
    - EDUC: 1-23 years (education)
    - SES: 1-5 (socioeconomic status)
    - MMSE: 0-30 (Mini-Mental State Examination)
    - CDR: 0, 0.5, 1, 2 (Clinical Dementia Rating)
    - eTIV: ~1400-1600 cm³ (estimated Total Intracranial Volume)
    - nWBV: 0.65-0.85 (normalized Whole Brain Volume)
    - ASF: 0.88-1.6 (Atlas Scaling Factor)
    
    Returns:
        pd.DataFrame: Synthetic OASIS-1 style clinical data
    """
    np.random.seed(random_state)
    
    # Generate subject IDs
    subject_ids = [f'OAS1_{i:04d}' for i in range(1, n_samples + 1)]
    
    # Gender distribution (OASIS-1 has more females)
    # Approximately 62% female in original dataset
    n_females = int(n_samples * 0.62)
    n_males = n_samples - n_females
    genders = ['F'] * n_females + ['M'] * n_males
    np.random.shuffle(genders)
    
    # Hand dominance (mostly right-handed)
    hands = np.random.choice(['R', 'L'], n_samples, p=[0.9, 0.1])
    
    # Age distribution (bimodal: young adults + older adults for dementia studies)
    # About 50% are younger adults (18-40), rest are older (60-96)
    n_young = int(n_samples * 0.25)  # Young adults
    n_old = n_samples - n_young  # Older adults
    
    ages_young = np.random.randint(18, 45, n_young)
    ages_old = np.random.normal(75, 8, n_old).clip(60, 96).astype(int)
    ages = np.concatenate([ages_young, ages_old])
    np.random.shuffle(ages)
    
    # Education (1-23 years, normal distribution around 12-14)
    education = np.random.normal(13, 3, n_samples).clip(1, 23).astype(int)
    
    # Socioeconomic Status (1-5 scale)
    ses = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    
    # CDR (Clinical Dementia Rating) - key outcome variable
    # Distribution: 0 (non-demented): ~75%, 0.5 (very mild): ~15%, 1 (mild): ~8%, 2 (moderate): ~2%
    # CDR is correlated with age - older subjects more likely to have dementia
    cdr_values = []
    for age in ages:
        if age < 50:
            # Young adults are non-demented
            cdr = 0
        else:
            # Older adults have varying CDR
            age_factor = (age - 50) / 50  # Increases with age
            base_probs = [0.75, 0.15, 0.08, 0.02]
            # Shift probabilities towards dementia for older subjects
            adjusted_probs = [
                base_probs[0] * (1 - age_factor * 0.5),
                base_probs[1] + age_factor * 0.15,
                base_probs[2] + age_factor * 0.1,
                base_probs[3] + age_factor * 0.05
            ]
            # Normalize
            total = sum(adjusted_probs)
            adjusted_probs = [p / total for p in adjusted_probs]
            cdr = np.random.choice([0, 0.5, 1, 2], p=adjusted_probs)
        cdr_values.append(cdr)
    cdr = np.array(cdr_values)
    
    # MMSE (Mini-Mental State Exam) - inversely correlated with CDR
    # Normal cognition: 27-30, Mild dementia: 20-26, Moderate: 10-20
    mmse_values = []
    for c, age in zip(cdr, ages):
        if c == 0:
            base_mmse = np.random.normal(29, 1)
        elif c == 0.5:
            base_mmse = np.random.normal(26, 2)
        elif c == 1:
            base_mmse = np.random.normal(22, 3)
        else:
            base_mmse = np.random.normal(17, 4)
        # Age effect (slight decline with age)
        age_effect = max(0, (age - 60) * 0.03)
        mmse = base_mmse - age_effect
        mmse_values.append(int(np.clip(mmse, 0, 30)))
    mmse = np.array(mmse_values)
    
    # eTIV (estimated Total Intracranial Volume) in cm³
    # Males typically have larger eTIV than females
    etiv_values = []
    for g in genders:
        if g == 'M':
            etiv = np.random.normal(1550, 120)
        else:
            etiv = np.random.normal(1350, 100)
        etiv_values.append(int(np.clip(etiv, 1000, 2000)))
    etiv = np.array(etiv_values)
    
    # nWBV (normalized Whole Brain Volume)
    # Decreases with age and is lower in dementia
    nwbv_values = []
    for c, age in zip(cdr, ages):
        base_nwbv = 0.78  # Normal baseline
        # Age effect
        age_effect = max(0, (age - 30) * 0.002)
        # Dementia effect
        dementia_effect = c * 0.03
        nwbv = base_nwbv - age_effect - dementia_effect + np.random.normal(0, 0.02)
        nwbv_values.append(round(np.clip(nwbv, 0.60, 0.85), 3))
    nwbv = np.array(nwbv_values)
    
    # ASF (Atlas Scaling Factor)
    # Inversely related to eTIV
    asf = (1482.0 / etiv).round(3)  # Reference brain volume
    
    # Number of visits (1-2 for cross-sectional, some have follow-up)
    n_visits = np.random.choice([1, 2], n_samples, p=[0.8, 0.2])
    
    # MR Delay (days between clinical and MRI assessment)
    mr_delay = np.random.randint(0, 30, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': subject_ids,
        'M/F': genders,
        'Hand': hands,
        'Age': ages,
        'EDUC': education,
        'SES': ses,
        'MMSE': mmse,
        'CDR': cdr,
        'eTIV': etiv,
        'nWBV': nwbv,
        'ASF': asf,
        'Delay': mr_delay
    })
    
    # Add some missing values (realistic for clinical data)
    # About 5% missing for some columns
    for col in ['SES', 'MMSE']:
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan
    
    return df


def generate_summary_statistics(df: pd.DataFrame) -> str:
    """Generate summary statistics report."""
    summary = []
    summary.append("=" * 60)
    summary.append("OASIS-1 STYLE SYNTHETIC DATASET SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nTotal Samples: {len(df)}")
    
    # Gender distribution
    gender_counts = df['M/F'].value_counts()
    summary.append(f"\nGender Distribution:")
    summary.append(f"  Female: {gender_counts.get('F', 0)} ({100*gender_counts.get('F', 0)/len(df):.1f}%)")
    summary.append(f"  Male: {gender_counts.get('M', 0)} ({100*gender_counts.get('M', 0)/len(df):.1f}%)")
    
    # Age distribution
    summary.append(f"\nAge Statistics:")
    summary.append(f"  Range: {df['Age'].min()} - {df['Age'].max()} years")
    summary.append(f"  Mean: {df['Age'].mean():.1f} years")
    summary.append(f"  Std: {df['Age'].std():.1f} years")
    
    # CDR distribution
    cdr_counts = df['CDR'].value_counts().sort_index()
    summary.append(f"\nClinical Dementia Rating (CDR) Distribution:")
    for cdr, count in cdr_counts.items():
        label = {0: 'Non-Demented', 0.5: 'Very Mild', 1: 'Mild', 2: 'Moderate'}.get(cdr, str(cdr))
        summary.append(f"  CDR={cdr} ({label}): {count} ({100*count/len(df):.1f}%)")
    
    # MMSE statistics
    summary.append(f"\nMMSE Statistics (excluding missing):")
    mmse_valid = df['MMSE'].dropna()
    summary.append(f"  Range: {mmse_valid.min():.0f} - {mmse_valid.max():.0f}")
    summary.append(f"  Mean: {mmse_valid.mean():.1f}")
    summary.append(f"  Std: {mmse_valid.std():.1f}")
    
    # Brain volume metrics
    summary.append(f"\nBrain Volume Metrics:")
    summary.append(f"  eTIV Range: {df['eTIV'].min()} - {df['eTIV'].max()} mm³")
    summary.append(f"  nWBV Range: {df['nWBV'].min():.3f} - {df['nWBV'].max():.3f}")
    
    # Binary classification target
    n_demented = (df['CDR'] > 0).sum()
    n_non_demented = (df['CDR'] == 0).sum()
    summary.append(f"\nBinary Classification Target:")
    summary.append(f"  Non-Demented (CDR=0): {n_non_demented} ({100*n_non_demented/len(df):.1f}%)")
    summary.append(f"  Demented (CDR>0): {n_demented} ({100*n_demented/len(df):.1f}%)")
    
    summary.append("\n" + "=" * 60)
    return "\n".join(summary)


def main():
    """Generate and save realistic OASIS-1 style dataset."""
    print("=" * 60)
    print("GENERATING REALISTIC OASIS-1 STYLE DATASET")
    print("=" * 60)
    
    # Set project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'raw'
    
    # Create directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    print("\n📊 Generating synthetic clinical data...")
    df = generate_oasis1_clinical_data(n_samples=416, random_state=42)
    
    # Save to CSV
    output_path = data_dir / 'clinical.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Saved clinical data to: {output_path}")
    
    # Print summary
    summary = generate_summary_statistics(df)
    print(summary)
    
    # Save summary
    summary_path = data_dir / 'data_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("This data mimics OASIS-1 statistical properties for")
    print("demonstration of the ML pipeline.")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()
