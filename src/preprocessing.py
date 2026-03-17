"""
preprocessing.py
Complete data preparation pipeline:
  1. Feature engineering
  2. Scaling
  3. SMOTE oversampling
  4. Train / val / test split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df["is_night"]    = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    # Amount features
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_bin"] = pd.cut(
        df["Amount"],
        bins=[-1, 10, 100, 500, 1e6],
        labels=["low", "mid", "high", "very_high"]
    ).cat.codes

    # Statistical features across all V columns
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"]  = df[v_cols].std(axis=1)

    # Fraud-correlated extreme value flags
    df["high_v14"] = (df["V14"].abs() > 5).astype(int)
    df["high_v10"] = (df["V10"].abs() > 5).astype(int)
    df["high_v12"] = (df["V12"].abs() > 5).astype(int)

    print(f"Feature engineering done. Shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target: str = "Class"):
    X = df.drop(columns=[target])
    y = df[target]

    # First split: 85% train+val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Second split: 70% train, 15% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    print(f"Train:      {X_train.shape[0]:,} rows | Fraud: {y_train.sum():,}")
    print(f"Validation: {X_val.shape[0]:,} rows   | Fraud: {y_val.sum():,}")
    print(f"Test:       {X_test.shape[0]:,} rows   | Fraud: {y_test.sum():,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, save_dir: str = "models/"):
    os.makedirs(save_dir, exist_ok=True)

    robust_cols   = ["Amount", "Time", "log_amount"]
    standard_cols = ["hour_of_day", "v_mean", "v_std"]

    rb_scaler = RobustScaler()
    st_scaler = StandardScaler()

    rb_present_train = [c for c in robust_cols   if c in X_train.columns]
    st_present_train = [c for c in standard_cols if c in X_train.columns]

    X_train[rb_present_train] = rb_scaler.fit_transform(X_train[rb_present_train])
    X_train[st_present_train] = st_scaler.fit_transform(X_train[st_present_train])

    X_val[rb_present_train]   = rb_scaler.transform(X_val[rb_present_train])
    X_val[st_present_train]   = st_scaler.transform(X_val[st_present_train])

    X_test[rb_present_train]  = rb_scaler.transform(X_test[rb_present_train])
    X_test[st_present_train]  = st_scaler.transform(X_test[st_present_train])

    joblib.dump(rb_scaler, os.path.join(save_dir, "robust_scaler.pkl"))
    joblib.dump(st_scaler, os.path.join(save_dir, "standard_scaler.pkl"))
    print("Scalers saved.")

    return X_train, X_val, X_test, rb_scaler, st_scaler


def apply_smote(X_train, y_train, random_state: int = 42):
    print(f"Before SMOTE — Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE  — Fraud: {y_resampled.sum():,} | Legit: {(y_resampled==0).sum():,}")
    return X_resampled, y_resampled