"""
Unit tests for model training functions.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.tabular_models import train_logistic_regression, train_random_forest, train_gbm


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y


def test_train_logistic_regression(sample_data):
    """Test logistic regression training."""
    X, y = sample_data
    
    model = train_logistic_regression(X, y, random_state=42)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert all(pred in [0, 1] for pred in y_pred)


def test_train_random_forest(sample_data):
    """Test random forest training."""
    X, y = sample_data
    
    model = train_random_forest(X, y, n_estimators=10, random_state=42)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert hasattr(model, 'feature_importances_')
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)


def test_train_gbm(sample_data):
    """Test gradient boosting training."""
    X, y = sample_data
    
    model = train_gbm(X, y, n_estimators=10, random_state=42)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert hasattr(model, 'feature_importances_')
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)


def test_model_accuracy(sample_data):
    """Test that models achieve reasonable accuracy on training data."""
    X, y = sample_data
    
    models = [
        train_logistic_regression(X, y, random_state=42),
        train_random_forest(X, y, n_estimators=50, random_state=42),
        train_gbm(X, y, n_estimators=50, random_state=42)
    ]
    
    for model in models:
        y_pred = model.predict(X)
        accuracy = (y_pred == y).mean()
        assert accuracy > 0.5  # Should be better than random


if __name__ == '__main__':
    pytest.main([__file__])
