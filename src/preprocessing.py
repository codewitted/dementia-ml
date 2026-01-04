"""
Preprocessing pipeline for dementia-ml dataset using scikit-learn.

This module provides a function to construct and use a preprocessing pipeline
that applies median imputation and standard scaling to numerical columns,
and most frequent imputation with one-hot encoding to categorical columns.

Example usage:
    from src.preprocessing import get_fit_transform
    numeric_features = ['age', 'test_score']
    categorical_features = ['gender', 'education_level']
    X_train_transformed, pipeline = get_fit_transform(X_train, numeric_features, categorical_features)

"""
from typing import List, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_preprocessing_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Construct a sklearn ColumnTransformer pipeline for preprocessing data.
    - Numeric columns: median imputation + StandardScaler
    - Categorical columns: most frequent imputation + OneHotEncoder
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    Returns:
        sklearn.compose.ColumnTransformer pipeline
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def get_fit_transform(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Fit the preprocessing pipeline on the input DataFrame and transform the data.
    Args:
        df: Input pandas DataFrame
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    Returns:
        Tuple of (preprocessed DataFrame, fitted preprocessing pipeline)
    """
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(df)
    # Get output feature names
    num_features_out = numeric_features
    if hasattr(preprocessor.named_transformers_["cat"].named_steps["onehot"], "get_feature_names_out"):
        cat_features_out = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
    else:
        # sklearn <1.0
        cat_features_out = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names(categorical_features)
    import numpy as np
    all_features = num_features_out + list(cat_features_out)
    X_processed_df = pd.DataFrame(X_processed, columns=all_features, index=df.index)
    return X_processed_df, preprocessor


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    data = {
        "age": [70, 82, None, 65],
        "test_score": [1.3, 2.1, 0.7, None],
        "gender": ["F", "M", "F", None],
        "education_level": ["HS", "College", None, "HS"],
    }
    df = pd.DataFrame(data)
    numeric_features = ["age", "test_score"]
    categorical_features = ["gender", "education_level"]

    X_processed_df, pipeline = get_fit_transform(df, numeric_features, categorical_features)
    print(X_processed_df)
