"""
train.py
Master training script — run this file to:
  1. Load data
  2. Engineer features
  3. Split & scale
  4. Apply SMOTE
  5. Train all 5 models
  6. Evaluate all models
  7. Generate SHAP explanations
  8. Save everything to models/
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader     import load_data, dataset_summary
from preprocessing   import engineer_features, split_data, scale_features, apply_smote
from models          import train_all_models
from evaluate        import evaluate_model, tune_threshold, compare_models
from explainability  import explain_model


def main():
    print("=" * 60)
    print("   FRAUD DETECTION — FULL TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/7] Loading dataset...")
    df = load_data("data/creditcard.csv")
    dataset_summary(df)

    # ── Step 2: Feature engineering ────────────────────────────
    print("\n[2/7] Engineering features...")
    df = engineer_features(df)

    # ── Step 3: Split data ─────────────────────────────────────
    print("\n[3/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── Step 4: Scale features ─────────────────────────────────
    print("\n[4/7] Scaling features...")
    X_train, X_val, X_test, _, _ = scale_features(
        X_train.copy(), X_val.copy(), X_test.copy()
    )

    # ── Step 5: Apply SMOTE ────────────────────────────────────
    print("\n[5/7] Applying SMOTE to balance classes...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── Step 6: Train all models ───────────────────────────────
    print("\n[6/7] Training all models...")
    trained_models = train_all_models(X_train_sm, y_train_sm)

    # ── Step 7: Evaluate all models ────────────────────────────
    print("\n[7/7] Evaluating all models...")
    results = []
    skip    = ["isolation_forest"]

    for name, model in trained_models.items():
        if name in skip:
            continue

        # Tune threshold on validation set
        best_threshold = tune_threshold(model, X_val, y_val, name)

        # Evaluate on test set
        import numpy as np
        y_prob   = model.predict_proba(X_test)[:, 1]
        y_pred   = (y_prob >= best_threshold).astype(int)

        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

        # SHAP explanations for XGBoost only (fastest tree explainer)
        if name == "xgboost":
            explain_model(model, X_train_sm, X_test, name)

    # ── Final comparison table ─────────────────────────────────
    compare_models(results)

    print("\nTraining complete!")
    print("All models saved to:  models/")
    print("All plots saved to:   models/plots/")
    print("\nNext step — run the Streamlit app:")
    print("  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()