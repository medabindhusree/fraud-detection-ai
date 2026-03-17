"""
models.py
Trains 5 models without MLflow conflicts.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression as MetaLearner
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=577,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0
        ),
    }
    return models


def get_stacking_model():
    estimators = [
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=577,
            eval_metric="aucpr", random_state=42, verbosity=0
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=MetaLearner(C=0.01, max_iter=1000, solver="saga"),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1
    )
    return stack


def train_all_models(X_train, y_train, save_dir: str = "models/"):
    os.makedirs(save_dir, exist_ok=True)
    trained = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --- Train standard models ---
    for name, model in get_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=skf, scoring="f1", n_jobs=1
            )
            print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        except Exception as e:
            print(f"  CV skipped: {e}")

        path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved to {path}")
        trained[name] = model

    # --- Train stacking ensemble ---
    print("\nTraining stacking ensemble...")
    stack = get_stacking_model()
    stack.fit(X_train, y_train)
    path = os.path.join(save_dir, "stacking_ensemble.pkl")
    joblib.dump(stack, path)
    print(f"  Saved to {path}")
    trained["stacking_ensemble"] = stack

    # --- Train Isolation Forest ---
    print("\nTraining isolation forest...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.00172,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    path = os.path.join(save_dir, "isolation_forest.pkl")
    joblib.dump(iso, path)
    trained["isolation_forest"] = iso
    print(f"  Saved to {path}")

    print("\nAll models trained and saved!")
    return trained


def load_model(name: str, save_dir: str = "models/"):
    path = os.path.join(save_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)