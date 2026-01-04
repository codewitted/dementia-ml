"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import get_preprocessing_pipeline, get_fit_transform


def test_get_preprocessing_pipeline():
    """Test that preprocessing pipeline is created correctly."""
    numeric_features = ['age', 'score']
    categorical_features = ['gender']
    
    pipeline = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    assert pipeline is not None
    assert 'num' in pipeline.named_transformers_
    assert 'cat' in pipeline.named_transformers_


def test_get_fit_transform():
    """Test preprocessing fit and transform."""
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, None, 40],
        'score': [1.5, 2.0, 1.8, None],
        'gender': ['M', 'F', 'M', 'F']
    })
    
    numeric_features = ['age', 'score']
    categorical_features = ['gender']
    
    # Fit and transform
    X_processed, pipeline = get_fit_transform(df, numeric_features, categorical_features)
    
    # Check output
    assert X_processed is not None
    assert len(X_processed) == len(df)
    assert X_processed.shape[1] > len(numeric_features)  # Should have one-hot encoded features
    assert not X_processed.isnull().any().any()  # No missing values after imputation


def test_preprocessing_handles_missing_values():
    """Test that preprocessing correctly handles missing values."""
    df = pd.DataFrame({
        'age': [25, None, 35, 40],
        'score': [None, 2.0, 1.8, 2.1],
        'gender': ['M', 'F', None, 'F']
    })
    
    numeric_features = ['age', 'score']
    categorical_features = ['gender']
    
    X_processed, _ = get_fit_transform(df, numeric_features, categorical_features)
    
    # Check no NaN values remain
    assert not X_processed.isnull().any().any()


if __name__ == '__main__':
    pytest.main([__file__])
