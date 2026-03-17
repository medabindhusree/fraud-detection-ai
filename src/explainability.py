"""
explainability.py
Model explainability using SHAP values.
This is what impresses FAANG interviewers —
showing WHY the model made a decision, not just WHAT it predicted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os


def explain_model(model, X_train, X_test,
                  model_name: str = "model",
                  save_dir: str = "models/plots/"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating SHAP explanations for {model_name}...")

    # Use TreeExplainer for tree-based models (fast)
    # Use LinearExplainer for logistic regression
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # --- Plot 1: Summary bar plot (feature importance) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_test,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP bar plot: {path}")

    # --- Plot 2: Beeswarm plot (impact distribution) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_test,
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Beeswarm Plot — {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP beeswarm plot: {path}")

    # --- Top 10 most important features ---
    feature_importance = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": np.abs(shap_vals).mean(axis=0)
    }).sort_values("importance", ascending=False).head(10)

    print(f"\n  Top 10 features for {model_name}:")
    print(feature_importance.to_string(index=False))

    return shap_vals, feature_importance


def explain_single_prediction(model, X_test, index: int = 0,
                               model_name: str = "model",
                               save_dir: str = "models/plots/"):
    """
    Explain a single transaction prediction.
    This is shown in the Streamlit app for real-time explainability.
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        explainer   = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Waterfall plot for single prediction
    shap_explanation = shap.Explanation(
        values        = shap_vals[index],
        base_values   = explainer.expected_value if not isinstance(
                            explainer.expected_value, list)
                        else explainer.expected_value[1],
        data          = X_test.iloc[index].values,
        feature_names = X_test.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, show=False)
    plt.title(f"Single Transaction Explanation — {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_single_explanation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved single explanation: {path}")

    return shap_vals[index]