"""
Tabular model training for dementia classification.
Provides functions to train Logistic Regression, Random Forest, and Gradient Boosting Machine (GBM) models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train a Logistic Regression model.
    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        **kwargs: Extra parameters for LogisticRegression.
    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest model.
    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        **kwargs: Extra parameters for RandomForestClassifier.
    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_gbm(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting Machine (GBM) model.
    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        **kwargs: Extra parameters for GradientBoostingClassifier.
    Returns:
        Trained GradientBoostingClassifier model.
    """
    model = GradientBoostingClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model
