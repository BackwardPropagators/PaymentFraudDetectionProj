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

def _build_lr():
    """Create a fresh LogisticRegression instance with default hyperparams."""
    return LogisticRegression(
        C=1.0,
        l1_ratio=0,          # equivalent to L2 penalty (Ridge)
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def train_and_evaluate(splits, X, y):
    """
    Train & evaluate LR across all time-based splits.

    Returns: results DataFrame, last trained model, feature_names
    """
    print("\n" + "=" * 60)
    print("11. LOGISTIC REGRESSION – TRAINING & EVALUATION")
    print("=" * 60)

    X_features = X.drop(columns=["Time"])
    feature_names = X_features.columns.tolist()

    lr_results = []
    last_model = None

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n  ── Split {split_num} ──")

        X_train_res, y_train_res, X_test_scaled, y_test, _ = scale_and_resample(
            X_features, y, train_idx, test_idx,
        )

        print(f"    Train: {len(X_train_res):,} samples (after SMOTE)  |  Test: {len(X_test_scaled):,} samples")

        model = _build_lr()
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["split"] = split_num
        lr_results.append(metrics)

        print(f"    Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}  |  "
              f"F1: {metrics['f1']:.4f}  |  MCC: {metrics['mcc']:.4f}  |  AUC-ROC: {metrics['auc_roc']:.4f}")
        print_classification_report(y_test, y_pred)

        last_model = model

    lr_df = pd.DataFrame(lr_results)
    lr_df.to_csv(os.path.join(RESULTS_DIR, "lr_results.csv"), index=False)
    print_summary_table(lr_df, "Logistic Regression")

    return lr_df, last_model, feature_names


def generate_visualisations(splits, X, y, lr_df, last_model, feature_names):
    """
    Generate all LR visualisation artefacts.

    Visualisations:
    - Confusion matrices for each split (shows FP/FN tradeoff per time window)
    - ROC curves overlaid per split (shows AUC consistency across time)
    - Metrics bar chart across splits (quick comparison)
    - Feature coefficients (interpretability – which PCA components drive predictions)
    """
    print("\n" + "=" * 60)
    print("12. LOGISTIC REGRESSION – VISUALISATIONS")
    print("=" * 60)

    X_features = X.drop(columns=["Time"])

    # Helper lambdas that train a fresh model per split for the plotting functions
    def _train_predict(train_idx, test_idx):
        X_tr, y_tr, X_te, _, _ = scale_and_resample(X_features, y, train_idx, test_idx)
        m = _build_lr()
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    def _train_predict_proba(train_idx, test_idx):
        X_tr, y_tr, X_te, _, _ = scale_and_resample(X_features, y, train_idx, test_idx)
        m = _build_lr()
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]

    plot_confusion_matrices(splits, X_features, y, _train_predict,
                            "Logistic Regression", "12_lr_confusion_matrices.png")
    plot_roc_curves(splits, X_features, y, _train_predict_proba,
                    "Logistic Regression", "13_lr_roc_curves.png")
    plot_metrics_bars(lr_df, splits, "Logistic Regression", "14_lr_metrics_by_split.png")
    plot_feature_coefficients(last_model.coef_[0], feature_names,
                              "Logistic Regression", f"Split {len(splits)}", "15_lr_feature_coefficients.png")

    print("\nLogistic Regression complete.")



