"""
evaluate.py
Full model evaluation suite:
  - Confusion matrix
  - Classification report
  - ROC-AUC score
  - Precision-Recall curve
  - F1, MCC scores
  - Threshold tuning
Addresses reviewer: overfitting concern + missing EDA/evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
import os


def evaluate_model(model, X_test, y_test, model_name: str = "model",
                   save_dir: str = "models/plots/"):
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Core metrics
    roc_auc     = roc_auc_score(y_test, y_prob)
    avg_prec    = average_precision_score(y_test, y_prob)
    f1          = f1_score(y_test, y_pred)
    mcc         = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  ROC-AUC Score:        {roc_auc:.4f}")
    print(f"  Avg Precision Score:  {avg_prec:.4f}")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  MCC Score:            {mcc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"]))

    # Plot confusion matrix
    _plot_confusion_matrix(y_test, y_pred, model_name, save_dir)

    # Plot ROC curve
    _plot_roc_curve(y_test, y_prob, roc_auc, model_name, save_dir)

    # Plot Precision-Recall curve
    _plot_pr_curve(y_test, y_prob, avg_prec, model_name, save_dir)

    return {
        "model":      model_name,
        "roc_auc":    round(roc_auc, 4),
        "avg_prec":   round(avg_prec, 4),
        "f1":         round(f1, 4),
        "mcc":        round(mcc, 4),
    }


def _plot_confusion_matrix(y_test, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {path}")


def _plot_roc_curve(y_test, y_prob, roc_auc, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved ROC curve: {path}")


def _plot_pr_curve(y_test, y_prob, avg_prec, model_name, save_dir):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="green", lw=2,
             label=f"PR Curve (AP = {avg_prec:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved PR curve: {path}")


def tune_threshold(model, X_val, y_val, model_name: str = "model"):
    """
    Find the optimal classification threshold that maximizes F1.
    Addresses reviewer: overfitting / poor model performance concern.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_f1, best_threshold = 0, 0.5

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1       = f1
            best_threshold = t

    print(f"\n  {model_name} — Best threshold: {best_threshold:.2f} | F1: {best_f1:.4f}")
    return best_threshold


def compare_models(results: list) -> pd.DataFrame:
    """Print a comparison table of all models."""
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n===== MODEL COMPARISON =====")
    print(df.to_string(index=False))
    print("============================\n")
    return df