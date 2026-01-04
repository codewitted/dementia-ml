import shap
import numpy as np


def compute_shap_values(model, X, explainer_type="TreeExplainer", nsamples=100):
    """
    Compute SHAP values for given model and data.
    Args:
        model: Trained model (e.g., XGBoost, LightGBM, scikit-learn tree-based models, etc.)
        X: Input data as a numpy array or pandas DataFrame
        explainer_type: Type of explainer ("TreeExplainer" or "KernelExplainer")
        nsamples: Number of samples for KernelExplainer (for large datasets, use a small subset)
    Returns:
        shap_values: SHAP values for the input data
        explainer: SHAP Explainer object
    """
    if explainer_type == "TreeExplainer":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif explainer_type == "KernelExplainer":
        # For KernelExplainer, sample background data if needed
        if isinstance(X, np.ndarray):
            bg = shap.utils.sample(X, min(100, len(X)))
        else:
            bg = shap.utils.sample(X.values, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(X, nsamples=nsamples)
    else:
        raise ValueError("Unsupported explainer_type: {}".format(explainer_type))
    return shap_values, explainer


def plot_shap_summary(shap_values, X, plot_type="dot"):
    """
    Display SHAP summary plot.
    Args:
        shap_values: Computed SHAP values
        X: Feature dataset corresponding to SHAP values
        plot_type: "dot", "bar", or any supported summary plot type
    """
    shap.summary_plot(shap_values, X, plot_type=plot_type)


def plot_shap_force(explainer, shap_values, X, index=0):
    """
    Plot SHAP force plot for a single sample.
    Args:
        explainer: SHAP Explainer object
        shap_values: Computed SHAP values
        X: Feature dataset
        index: Index of sample to plot
    Returns:
        force_plot: The SHAP force plot object (for display in notebooks)
    """
    return shap.force_plot(explainer.expected_value, shap_values[index], X.iloc[index] if hasattr(X, "iloc") else X[index])
