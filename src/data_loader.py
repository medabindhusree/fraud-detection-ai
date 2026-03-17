"""
data_loader.py
Loads the Kaggle Credit Card Fraud Detection dataset.
Dataset: 284,807 transactions | 492 frauds | 30 features
"""

import pandas as pd
import numpy as np
import os


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv inside the data/ folder."
        )
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    summary = {
        "total_records":      len(df),
        "total_features":     df.shape[1] - 1,
        "fraud_records":      len(fraud),
        "legit_records":      len(legit),
        "fraud_pct":          round(len(fraud) / len(df) * 100, 4),
        "missing_values":     int(df.isnull().sum().sum()),
        "duplicate_rows":     int(df.duplicated().sum()),
        "amount_min":         round(df["Amount"].min(), 2),
        "amount_max":         round(df["Amount"].max(), 2),
        "amount_mean":        round(df["Amount"].mean(), 2),
        "fraud_amount_mean":  round(fraud["Amount"].mean(), 2),
        "legit_amount_mean":  round(legit["Amount"].mean(), 2),
        "time_span_hours":    round(df["Time"].max() / 3600, 1),
    }

    print("\n===== DATASET SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")
    print("===========================\n")
    return summary


def get_feature_names(df: pd.DataFrame) -> dict:
    return {
        "pca_features":   [f"V{i}" for i in range(1, 29)],
        "time_feature":   ["Time"],
        "amount_feature": ["Amount"],
        "target":         ["Class"],
        "all_features":   [c for c in df.columns if c != "Class"],
    }