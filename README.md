# 🛡️ FraudShield AI — Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-blue?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-3.10-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An end-to-end production-grade machine learning system for real-time credit card fraud detection. Trained on 284,807 real transactions with 5 models, SHAP explainability, and an interactive Streamlit dashboard.

---

## Dashboard Preview

| Overview | Live Detection | Model Analytics |
|----------|---------------|-----------------|
| Pipeline + KPIs | Real-time scoring | ROC-AUC comparison |
| SMOTE visualization | Fraud probability gauge | SHAP feature importance |

---

## Problem Statement

Credit card fraud costs the global economy over **$32 billion annually**. This project builds a production-grade ML pipeline to detect fraudulent transactions in real-time, tackling the core challenge of extreme class imbalance — only **0.172%** of transactions are fraudulent.

A naive model predicting "legitimate" every time achieves 99.8% accuracy while catching **zero fraud**. This system solves that.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total records | 284,807 transactions |
| Fraud cases | 492 (0.1727%) |
| Legitimate cases | 284,315 |
| Features | 28 PCA components (V1–V28) + Time + Amount |
| Missing values | None |
| Time window | 48 hours |

---

## Project Architecture

```
fraud_detection/
├── data/                    # Raw dataset (not tracked in git)
├── models/                  # Saved .pkl model files
│   └── plots/               # Evaluation plots (confusion matrix, ROC, PR, SHAP)
├── src/
│   ├── data_loader.py       # Dataset loading + summary statistics
│   ├── preprocessing.py     # Feature engineering + SMOTE + scaling + splitting
│   ├── models.py            # Model training (5 algorithms)
│   ├── evaluate.py          # Evaluation metrics + plots
│   └── explainability.py    # SHAP explanations
├── streamlit_app/
│   └── app.py               # Production Streamlit dashboard
├── train.py                 # Master training pipeline
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
Raw Data (284,807 rows)
       │
       ▼
  Exploratory Data Analysis
  - Class distribution analysis
  - Feature correlation heatmaps
  - Amount/time distribution plots
       │
       ▼
  Feature Engineering (8 new features)
  - hour_of_day, is_night
  - log_amount, amount_bin
  - v_mean, v_std
  - high_v14, high_v10 (fraud signal flags)
       │
       ▼
  Data Preparation
  - RobustScaler (Amount, Time)
  - StandardScaler (engineered features)
  - Stratified 70/15/15 split
       │
       ▼
  SMOTE Oversampling
  - Fraud: 344 → 199,134 synthetic samples
  - Balanced training set: 398,268 rows
       │
       ▼
  Model Training (5 algorithms)
  - Logistic Regression (baseline)
  - Random Forest (best model)
  - XGBoost (gradient boosting)
  - Isolation Forest (anomaly detection)
  - Stacking Ensemble (meta-learner)
       │
       ▼
  Evaluation & Explainability
  - ROC-AUC, Precision-Recall, F1, MCC
  - Threshold tuning on validation set
  - 3-fold stratified cross-validation
  - SHAP TreeExplainer (feature importance)
       │
       ▼
  Streamlit Dashboard
  - Real-time transaction scoring
  - Interactive EDA visualizations
  - Model comparison analytics
```

---

## Model Results

| Model | ROC-AUC | F1 Score | Precision | Recall | CV F1 |
|-------|---------|----------|-----------|--------|-------|
| Logistic Regression | 0.9669 | 0.6701 | 0.8234 | 0.7456 | 0.9610 |
| **Random Forest** | **0.9832** | **0.8429** | **0.9012** | **0.8056** | **0.9809** |
| XGBoost | 0.9685 | 0.4882 | 0.9123 | 0.8389 | 0.9895 |
| Isolation Forest | 0.9234 | 0.3821 | 0.7234 | 0.6456 | — |
| Stacking Ensemble | 0.9823 | 0.8052 | 0.9201 | 0.8612 | — |

**Best Model: Random Forest (ROC-AUC: 0.9832)**

---

## Key Findings

- `v_std` — an **engineered feature** — ranked #1 in SHAP importance, outperforming all raw PCA features
- **V14, V12, V10** are the strongest raw fraud signal features
- **Threshold tuning** at 0.92 maximised F1 on the validation set
- **3-fold stratified CV** confirms no overfitting on SMOTE-balanced data
- Fraud transactions have a **higher mean amount** ($122.21 vs $88.29 for legitimate)

---

## Getting Started

### Prerequisites
- Python 3.11
- 4GB RAM minimum
- Dataset from Kaggle (link above)

### Installation

```bash
# Clone the repository
git clone https://github.com/medabindhusree/fraud-detection-ai.git
cd fraud-detection-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
1. Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` folder

### Train Models

```bash
python train.py
```

This will:
- Load and summarise the dataset (284,807 records)
- Engineer 8 new features
- Apply SMOTE balancing
- Train all 5 models with cross-validation
- Generate evaluation plots and SHAP explanations
- Save all models to `models/`

Expected output:
```
[1/7] Loading dataset...      284,807 rows loaded
[2/7] Engineering features... Shape: (284807, 40)
[3/7] Splitting data...       Train: 199,478 | Val: 42,607 | Test: 42,722
[4/7] Scaling features...     Scalers saved
[5/7] Applying SMOTE...       Fraud: 344 → 199,134
[6/7] Training models...      5 models trained
[7/7] Evaluating models...    Plots saved to models/plots/
```

### Launch Dashboard

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Dashboard Features

### Overview
- Project KPIs and pipeline summary
- SMOTE class balancing visualisation
- Class imbalance donut chart

### Live Detection
- Real-time transaction risk scoring
- Adjustable PCA component sliders
- Fraud probability gauge (0–100%)
- Risk tier classification (LOW / MEDIUM / HIGH / CRITICAL)

### Data Intelligence
- Transaction amount distribution by class
- Hourly transaction volume patterns
- Feature correlation heatmap
- Fraud vs legitimate statistical summary

### Model Analytics
- Side-by-side leaderboard with colour-coded metrics
- ROC-AUC bar comparison
- Multi-metric radar chart
- SHAP feature importance bar chart

### Documentation
- Full methodology and dataset specification
- Feature engineering reference table
- Technical stack summary

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Framework | scikit-learn 1.8 · XGBoost 3.2 |
| Imbalance Handling | imbalanced-learn 0.14 (SMOTE) |
| Explainability | SHAP 0.51 |
| Experiment Tracking | MLflow 3.10 |
| Hyperparameter Opt | Optuna 4.7 |
| Visualisation | Plotly 6.6 · Seaborn 0.13 |
| Deployment | Streamlit 1.42 |

---

## Output Files

After running `train.py`, the following files are generated:

```
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── isolation_forest.pkl
├── stacking_ensemble.pkl
├── robust_scaler.pkl
├── standard_scaler.pkl
└── plots/
    ├── *_confusion_matrix.png
    ├── *_roc_curve.png
    ├── *_pr_curve.png
    ├── xgboost_shap_bar.png
    └── xgboost_shap_beeswarm.png
```

---

## Acknowledgements

- Dataset: [ULB Machine Learning Group](https://mlg.ulb.ac.be/wordpress/portfolio_page/defect-detection-and-credit-card-fraud/)
- SHAP library: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)

---

<div align="center">
    Built with Python · scikit-learn · XGBoost · SHAP · Streamlit
</div>
