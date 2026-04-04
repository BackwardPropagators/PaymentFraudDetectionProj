"""
Logistic Regression pipeline for Dataset 3.

Logistic Regression:
- Linear baseline classifier using the sigmoid function and Binary Cross-Entropy (NLL) loss
- Objective: Z₁ = min_{w,c} -(1/n) Σ [yᵢ log(σ(wᵀxᵢ+c)) + (1-yᵢ) log(1-σ(wᵀxᵢ+c))] + λ||w||₂²
- Using ℓ₂ regularisation (Ridge) via scikit-learn's LogisticRegression(penalty='l2')
- Evaluated across all 4 expanding-window time-based splits
- SMOTE applied to training data only per split to avoid data leakage
- Metrics: Precision, Recall, F1, MCC, AUC-ROC

Why Logistic Regression?
It is the baseline linear classifier for binary classification. Its interpretability (feature coefficients map directly
to log-odds) makes it valuable for stakeholders who need to understand WHY a transaction is flagged. Its weakness
(linear decision boundary) is a key discussion point when comparing against Random Forest and the FFNN.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils import RANDOM_STATE, RESULTS_DIR, scale_and_resample
from src.evaluation import (
    compute_metrics,
    print_classification_report,
    print_summary_table,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_bars,
    plot_feature_coefficients,
)


