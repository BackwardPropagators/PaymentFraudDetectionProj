"""
Evaluation metrics and visualisation helpers for all models.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils import RESULTS_DIR, save_fig

'''
Precision:  Of all transactions the model flagged as fraud, what fraction actually were fraud? Low values (like your ~0.058) mean lots of false alarms.
Recall:    Of all actual fraud transactions, what fraction did the model catch? Your ~0.875 means the model catches most fraud.
F1-Score:   Harmonic mean of precision and recall — a single number balancing both. Heavily penalised when either is low.
MCC:    	Matthews Correlation Coefficient — a balanced measure that accounts for all four quadrants of the confusion matrix (TP, TN, FP, FN). Ranges from −1 (total disagreement) to +1 (perfect). More robust than F1 on imbalanced data.
AUC-ROC:    Area under the ROC curve — measures how well the model's probability scores separate the two classes, regardless of the threshold chosen. 0.5 = random, 1.0 = perfect separation.

'''
def compute_metrics(y_true, y_pred, y_prob): # Computes the five core evaluation metrics for a single split and returns them as a dictionary.
    """Compute the standard metric suite and return as a dict."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_prob),
    }

'''
Prints sklearn's classification_report. 
Shows per-class precision, recall, F1, and support (sample count), plus macro/weighted averages.
'''
def print_classification_report(y_true, y_pred):
    """Print a classification report with Legitimate / Fraudulent labels."""
    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
    ))

'''
Prints the mean += standard deviation of each metric across all 4 time-based splits. This tells you:

Mean: overall expected performance
Std: how stable the model is across different time windows (low std = consistent)
'''
def print_summary_table(results_df, model_name):
    """Print mean ± std for each metric across splits."""
    print(f"\n  ── {model_name} Summary Across All Splits ──")
    print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
    print(f"  {'-' * 32}")
    for metric in ["precision", "recall", "f1", "mcc", "auc_roc"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric:<12} {mean_val:>10.4f} {std_val:>10.4f}")



"""
Plot confusion matrices for each split.

train_and_predict_fn(train_idx, test_idx) -> y_pred
Produces one heatmap per split, side by side. Each heatmap is a 2×2 grid

Top-left (TN): Legitimate transactions correctly classified — should be large.
Top-right (FP): Legitimate transactions wrongly flagged as fraud — these are false alarms. With SMOTE + LR, this number is high (hence low precision).
Bottom-left (FN): Fraud that slipped through undetected — want this as close to 0 as possible.
Bottom-right (TP): Fraud correctly caught — want this as high as possible.
Comparing across splits shows whether the FP/FN tradeoff shifts as the training window expands.
"""
def plot_confusion_matrices(splits, X_features, y, train_and_predict_fn, model_name, filename):

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        y_pred = train_and_predict_fn(train_idx, test_idx)
        y_test = y.loc[test_idx]

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[split_num - 1],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[split_num - 1].set_title(f"Split {split_num}", fontsize=12, fontweight="bold")
        axes[split_num - 1].set_ylabel("Actual")
        axes[split_num - 1].set_xlabel("Predicted")

    fig.suptitle(f"{model_name} – Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, filename)



"""
Plot ROC curves for each split.

train_and_predict_proba_fn(train_idx, test_idx) -> y_prob
X-axis (False Positive Rate): fraction of legit transactions wrongly flagged.
Y-axis (True Positive Rate / Recall): fraction of fraud correctly caught.
Each curve shows the tradeoff as you sweep the classification threshold from 0 to 1.
The dashed diagonal is a random classifier (AUC = 0.5). The further a curve bows toward the top-left corner, the better.
AUC values in the legend quantify this — closer to 1.0 is better.
If the curves from different splits are tightly clustered, the model generalises consistently across time windows.
"""
def plot_roc_curves(splits, X_features, y, train_and_predict_proba_fn, model_name, filename):

    fig, ax = plt.subplots(figsize=(8, 6))

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        y_prob = train_and_predict_proba_fn(train_idx, test_idx)
        y_test = y.loc[test_idx]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"Split {split_num} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} – ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, filename)


"""
Plot a grouped bar chart of metrics across splits.
Lets you visually compare all 5 metrics side-by-side across splits.
Look for bars that stay roughly the same height across splits (stability).
If precision bars are short but recall/AUC bars are tall, the model is catching fraud but generating many false positives — expected for a linear model on severely imbalanced data.

"""
def plot_metrics_bars(results_df, splits, model_name, filename):

    fig, ax = plt.subplots(figsize=(10, 5))

    metrics_to_plot = ["precision", "recall", "f1", "mcc", "auc_roc"]
    x = np.arange(len(splits))
    width = 0.15
    colors_metrics = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for i, metric in enumerate(metrics_to_plot):
        vals = results_df[metric].values
        ax.bar(x + i * width, vals, width, label=metric.upper().replace("_", "-"), color=colors_metrics[i])

    ax.set_xlabel("Split", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{model_name} – Metrics by Split", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Split {i+1}" for i in range(len(splits))])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, filename)



"""
Plot horizontal bar chart of top feature coefficients.
Green bars (positive coefficient): increasing this feature's value pushes the prediction toward fraud.
Red bars (negative coefficient): increasing this feature's value pushes the prediction away from fraud.
Longer bars = stronger influence on the model's decision.
Since V1–V28 are PCA components, you can't directly name the underlying card features, but this shows which components the model relies on most — useful for interpretability discussions and comparing against feature importance from Random Forest.
"""
def plot_feature_coefficients(coefs, feature_names, model_name, split_label, filename, top_n=15):

    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    ax.barh(
        range(top_n),
        coefs[sorted_idx[:top_n]],
        color=["#e74c3c" if c < 0 else "#2ecc71" for c in coefs[sorted_idx[:top_n]]],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title(f"{model_name} – Top {top_n} Feature Coefficients ({split_label})", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, filename)
