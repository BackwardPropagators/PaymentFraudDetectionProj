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
    feature_names = X_features.columns.tolist() # for later use in coefficient plotting

    lr_results = [] # to store metrics per split for summary table and visualisations
    last_model = None # we will keep the last trained model (from the final split) for feature coefficient visualisation later

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1): # split_num starts at 1 for easier readability in printouts
        print(f"\n  ── Split {split_num} ──")

        X_train_res, y_train_res, X_test_scaled, y_test, _ = scale_and_resample( # scaling + SMOTE only on training data to prevent data leakage
            X_features, y, train_idx, test_idx, # we return the scaled test set for evaluation, but only resample the training set
        )

        print(f"    Train: {len(X_train_res):,} samples (after SMOTE)  |  Test: {len(X_test_scaled):,} samples")

        model = _build_lr() # create a fresh model instance for this split
        model.fit(X_train_res, y_train_res) # fit on the resampled training data

        y_pred = model.predict(X_test_scaled) # predict on the scaled test set (no SMOTE applied to test data)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] # get probabilities for the positive class for AUC-ROC calculation

        metrics = compute_metrics(y_test, y_pred, y_prob) # compute all relevant metrics (precision, recall, f1, mcc, auc_roc)
        metrics["split"] = split_num # add split number to metrics for later use in summary table and visualisations
        lr_results.append(metrics) # store metrics for this split

        print(f"    Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}  |  "
              f"F1: {metrics['f1']:.4f}  |  MCC: {metrics['mcc']:.4f}  |  AUC-ROC: {metrics['auc_roc']:.4f}")
        print_classification_report(y_test, y_pred)

        last_model = model # keep the last trained model for feature coefficient visualisation later

    lr_df = pd.DataFrame(lr_results) # convert list of dicts to DataFrame for easier analysis and visualisation
    lr_df.to_csv(os.path.join(RESULTS_DIR, "lr_results.csv"), index=False) # save results to CSV
    print_summary_table(lr_df, "Logistic Regression") # print a nice summary table of metrics across splits

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

    X_features = X.drop(columns=["Time"]) # we need to drop 'Time' again here for the plotting functions, which expect the same feature set as during training. We could refactor this to avoid repeating, but it's a minor redundancy for clarity in this context.

    # Helper lambdas that train a fresh model per split for the plotting functions
    def _train_predict(train_idx, test_idx): # this will be called by the plotting functions to get predictions for confusion matrices and ROC curves
        X_tr, y_tr, X_te, _, _ = scale_and_resample(X_features, y, train_idx, test_idx) # we need to scale and resample again here because the plotting functions call this lambda separately for each split, and we want to ensure they get the same data processing as during training (including SMOTE on the training set). This is a bit redundant but ensures consistency in the visualisations. We could refactor to store the processed data per split to avoid this repetition, but it keeps the code straightforward for now.
        m = _build_lr() # create a fresh model instance for this split
        m.fit(X_tr, y_tr) # fit on the resampled training data for this split
        return m.predict(X_te) # return predictions for the test set of this split (no SMOTE applied to test data)

    def _train_predict_proba(train_idx, test_idx): # similar to the above but returns probabilities for AUC-ROC plotting
        X_tr, y_tr, X_te, _, _ = scale_and_resample(X_features, y, train_idx, test_idx) # same data processing as above to ensure consistency in visualisations
        m = _build_lr() # create a fresh model instance for this split
        m.fit(X_tr, y_tr) # fit on the resampled training data for this split
        return m.predict_proba(X_te)[:, 1] # return probabilities for the positive class for AUC-ROC calculation

    plot_confusion_matrices(splits, X_features, y, _train_predict, # this will generate confusion matrices for each split using the helper lambda that trains a fresh model per split
                            "Logistic Regression", "12_lr_confusion_matrices.png") # we pass the feature set without 'Time' to the plotting function, which expects the same features as during training. The plotting function will call the _train_predict lambda for each split, which will handle the scaling and resampling internally to ensure consistency with the training process.
    plot_roc_curves(splits, X_features, y, _train_predict_proba,# this will generate ROC curves for each split using the helper lambda that trains a fresh model per split and returns probabilities for AUC-ROC calculation
                    "Logistic Regression", "13_lr_roc_curves.png")
    plot_metrics_bars(lr_df, splits, "Logistic Regression", "14_lr_metrics_by_split.png") # this will generate a bar chart comparing the key metrics across splits using the results DataFrame we created during training
    plot_feature_coefficients(last_model.coef_[0], feature_names,
                              "Logistic Regression", f"Split {len(splits)}", "15_lr_feature_coefficients.png") # this will plot the feature coefficients from the last trained model (from the final split) to show which PCA components are driving the predictions. We pass the feature names for better interpretability in the plot.

    print("\nLogistic Regression complete.")



