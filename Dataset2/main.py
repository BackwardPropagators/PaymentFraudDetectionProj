"""
Dataset 2 - Credit Card Fraud Detection (2023 Kaggle Dataset)
=============================================================

Full pipeline: Data Analysis → Preprocessing → Feed-Forward Neural Network → Evaluation

This file implements the complete experiment for Dataset 2 of the Payment Fraud Detection project.
Dataset 2 is loaded from Kaggle (nelgiriyewithana/credit-card-fraud-detection-dataset-2023) and
contains 550,000+ anonymised credit card transactions from European cardholders in 2023.

Key Features:
  - id:      Unique identifier for each transaction (dropped before training — not a real feature)
  - V1–V28: Anonymised PCA-transformed features representing various transaction attributes
  - Amount:  The transaction amount
  - Class:   Binary label — 0 (legitimate) or 1 (fraudulent)

Unlike Dataset 3, this dataset has NO Time column, so we use k-Stratified Fold Cross Validation
(k=10) instead of time-based splitting. Stratification ensures each fold preserves the original
fraud-to-legitimate ratio.

IMPORTANT INFO ON CLASS BALANCE:
  Unlike most real-world fraud datasets, this Kaggle dataset has been PRE-BALANCED by the
  publisher — it contains a near-perfect 50/50 split between fraudulent and legitimate
  transactions (284,315 legit vs 284,314 fraud). This means SMOTE is NOT needed here.
  In Datasets 1 and 3, fraud makes up <1% of transactions, requiring SMOTE to prevent
  the model from defaulting to "predict everything as legitimate." Here, the model sees
  equal numbers of both classes naturally.

  This is a valuable discussion point: comparing FFNN performance on a balanced dataset
  (Dataset 2) vs. SMOTE-augmented imbalanced datasets (Datasets 1 & 3) reveals how
  class distribution affects model behaviour.

Algorithms implemented:
  - Feed-Forward Neural Network (FFNN)

Mathematical Objective (FFNN):
  Z₃ = min_{W₁,b₁,...,Wₗ,bₗ} -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] + λ Σₗ ||Wₗ||₂²
  where:
    h₁ = ReLU(W₁ᵀx + b₁)        — first hidden layer
    h₂ = ReLU(W₂ᵀh₁ + b₂)       — second hidden layer
    ŷ  = σ(Wₒᵀh₂ + bₒ)          — output layer (sigmoid → [0,1])
  Optimised via backpropagation with Adam optimiser.

Evaluation Metrics:
  - Precision, Recall, F1, MCC, AUC-ROC

Preprocessing:
  - StandardScaler: fit on training fold only, transform both train and test
  - No SMOTE needed — dataset is already balanced (50/50 split)

References:
  - Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
  - Project Plan: fri_apr_03_2026_payment_fraud_detection_project_plan.md
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Docker (no display server)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
import kagglehub
from kagglehub import KaggleDatasetAdapter

'''
SHARED CONSTANTS

These constants are used throughout the pipeline to ensure consistency
and reproducibility. Everything is controlled from here so nothing
is scattered across the codebase.
── FFNN Hyperparameters ──
These control the architecture and training behaviour of the neural network.
Chosen to balance model capacity (enough neurons to learn fraud patterns) with
regularisation (dropout + weight decay to prevent overfitting).
'''
RANDOM_STATE = 42          # Seed for all random operations — ensures reproducibility
N_FOLDS = 10               # Number of stratified cross-validation folds
RESULTS_DIR = "results"    # Directory to save all output plots and CSV results


HIDDEN1 = 64               # Neurons in first hidden layer
HIDDEN2 = 32               # Neurons in second hidden layer (compression — forces the network to learn a more compact representation)
DROPOUT = 0.3              # Dropout probability — randomly zeroes 30% of neurons each forward pass during training
EPOCHS = 50                # Number of full passes through the training data
BATCH_SIZE = 256            # Number of samples per gradient update (mini-batch SGD)
LEARNING_RATE = 1e-3        # Adam learning rate — controls step size during gradient descent
WEIGHT_DECAY = 1e-4         # L2 regularisation strength (λ) — penalises large weights to reduce overfitting

os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save a matplotlib figure to the results directory and close it to free memory."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

'''
 SECTION 1: DATA LOADING
 Load the dataset from Kaggle using kagglehub. This downloads the latest
 version of the dataset and returns it as a pandas DataFrame.
 The dataset contains 550,000+ transactions with 31 columns:
   id, V1-V28, Amount, Class
The 'id' column is just a row index from Kaggle and carries no predictive
information — including it would let the model memorise row numbers instead
of learning actual fraud patterns (same reasoning as dropping nameOrig/nameDest
in Dataset 1). We drop it immediately after loading.

Class is already integer (0/1) in this dataset, unlike Dataset 3 where it
arrives as quoted strings ("0"/"1").
'''


def load_data():
    """
    Load Dataset 2 from Kaggle via kagglehub.
    Returns the cleaned DataFrame with 'id' dropped.
    """
    print("=" * 60)
    print("1. LOADING DATA FROM KAGGLE")
    print("=" * 60)

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nelgiriyewithana/credit-card-fraud-detection-dataset-2023",
        "creditcard_2023.csv",
    )

    print(f"  Raw shape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"\n  Column types:\n{df.dtypes.value_counts()}")
    print(f"\n  First 5 rows:\n{df.head()}")
    print(f"\n  Basic statistics:\n{df.describe()}")
    '''
    Drop the 'id' column — it's just a sequential integer index from Kaggle,
    not a real transaction feature. Keeping it would let the model cheat by
    memorising row positions instead of learning fraud patterns.
    '''

    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print("\n  → Dropped 'id' column (sequential index, not a feature)")

    # Check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n  Missing values: {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,}")
    if duplicates > 0:
        print("  → Dropping duplicates")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  → New shape: {df.shape[0]:,} rows")

    return df

'''
SECTION 2: CLASS DISTRIBUTION ANALYSIS
This section analyses the balance between fraudulent and legitimate
transactions. Unlike Datasets 1 and 3 where class imbalance is the
central challenge (<1% fraud), this dataset has been pre-balanced by
the Kaggle publisher — it contains a near-perfect 50/50 split.

This is unusual for real-world fraud data and has major implications:
   1. SMOTE is NOT needed — the classes are already balanced
   2. The model won't default to "predict all legitimate" since it sees
      equal numbers of both classes during training
   3. Accuracy becomes a more meaningful metric here (unlike imbalanced
      datasets), though we still use MCC and AUC-ROC for consistency
      with Datasets 1 and 3

The bar chart shows raw counts with percentages annotated above each bar.
The pie chart confirms the 50/50 split visually.
'''


def analyse_class_distribution(df):
    """
    Analyse and visualise the class distribution (fraud vs legitimate).
    Returns colors and labels for reuse in later plots.
    """
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = df["Class"].value_counts().sort_index()
    class_pct = df["Class"].value_counts(normalize=True).sort_index() * 100

    print(f"  Legitimate (0): {class_counts[0]:>8,}  ({class_pct[0]:.4f}%)")
    print(f"  Fraudulent (1): {class_counts[1]:>8,}  ({class_pct[1]:.4f}%)")
    print(f"  Ratio: 1 : {class_counts[0] / class_counts[1]:.2f}  (near-perfect balance)")

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of class counts
    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Class Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, (cnt, pct) in enumerate(zip(class_counts.values, class_pct.values)):
        axes[0].text(i, cnt + cnt * 0.01, f"{cnt:,}\n({pct:.3f}%)", ha="center", fontsize=10)

    # Pie chart — no explode needed since both slices are ~equal
    axes[1].pie(
        class_counts.values, labels=labels, colors=colors,
        autopct="%1.3f%%", startangle=90,
    )
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    fig.suptitle("Class Distribution in Dataset 2 (Credit Card Fraud 2023)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "01_class_distribution.png")

    return colors, labels

'''
SECTION 3: TRANSACTION AMOUNT ANALYSIS
The Amount feature is the only non-PCA feature in this dataset.
It represents the actual transaction amount in euros. Analysing its
distribution helps us understand:
  - Whether fraudulent transactions tend to be larger or smaller
  - Whether the distribution is heavily skewed (spoiler: it is)
  - Whether log transformation would help normalise it

We produce three visualisations:
   1. Full range histogram — shows the overall distribution shape
   2. Zoomed in (≤ $500) — most transactions are small, so this is where
      the detail is. Without the zoom you can barely see anything because
      a few massive outliers stretch the x-axis
   3. Log-transformed histogram — log1p(Amount) to handle the skew.
      log1p adds 1 before taking log to handle Amount=0 (log(0) = -inf)
      I learned this the hard way working on Dataset 3...
'''


def analyse_amounts(df):
    """Analyse and visualise transaction amount distributions for fraud vs legitimate."""
    print("\n" + "=" * 60)
    print("3. TRANSACTION AMOUNT ANALYSIS")
    print("=" * 60)

    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    print(f"  {'Metric':<20} {'Legitimate':>14} {'Fraudulent':>14}")
    print(f"  {'-' * 48}")
    print(f"  {'Mean':<20} {legit.mean():>14.2f} {fraud.mean():>14.2f}")
    print(f"  {'Median':<20} {legit.median():>14.2f} {fraud.median():>14.2f}")
    print(f"  {'Std Dev':<20} {legit.std():>14.2f} {fraud.std():>14.2f}")
    print(f"  {'Min':<20} {legit.min():>14.2f} {fraud.min():>14.2f}")
    print(f"  {'Max':<20} {legit.max():>14.2f} {fraud.max():>14.2f}")
    print(f"  {'25th Percentile':<20} {legit.quantile(0.25):>14.2f} {fraud.quantile(0.25):>14.2f}")
    print(f"  {'75th Percentile':<20} {legit.quantile(0.75):>14.2f} {fraud.quantile(0.75):>14.2f}")

    # Full range + zoomed histograms
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].hist(legit, bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axes[0].hist(fraud, bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axes[0].set_title("Amount Distribution (Full Range)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(legit[legit <= 500], bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axes[1].hist(fraud[fraud <= 500], bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axes[1].set_title("Amount Distribution (≤ $500)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("Transaction Amount: Legitimate vs Fraudulent", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_amount_distribution.png")

    # Log-transformed histogram — log1p avoids log(0) = -inf
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(legit), bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    ax.hist(np.log1p(fraud), bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    ax.set_title("Log-Transformed Amount Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_amount_log_distribution.png")


'''
SECTION 4: PCA FEATURE ANALYSIS (V1–V28)

V1 through V28 are the result of PCA (Principal Component Analysis)
applied to the original transaction features before the dataset was
released. PCA is a dimensionality reduction technique that transforms
the original correlated features into a set of uncorrelated components
ordered by the amount of variance they explain. Because PCA was applied,
we can't know the original feature names (e.g., "merchant category",
"transaction time of day"), but we CAN identify which components
differ most between fraud and legitimate transactions.

We compute the absolute difference in means between fraud and legit
for each V feature. Features with large mean differences are the ones
that contain the most discriminative information — they're the features
the model will rely on most heavily.

Top 6 features are plotted as histograms (6 because any more gets
overwhelming), and all 28 are shown as box plots for completeness.
'''

def analyse_pca_features(df):
    """
    Analyse PCA features V1-V28 to identify which components differ most
    between fraudulent and legitimate transactions.
    Returns the list of V column names.
    """
    print("\n" + "=" * 60)
    print("4. PCA FEATURE ANALYSIS (V1–V28)")
    print("=" * 60)

    v_cols = [f"V{i}" for i in range(1, 29)]

    # Calculate mean of each PCA feature for legit vs fraud
    mean_legit = df[df["Class"] == 0][v_cols].mean()
    mean_fraud = df[df["Class"] == 1][v_cols].mean()
    mean_diff = (mean_fraud - mean_legit).abs().sort_values(ascending=False)

    print("  Top PCA features by |mean(fraud) – mean(legit)|:")
    for feat, diff in mean_diff.head(10).items():
        print(f"    {feat:<5}  delta = {diff:.4f}  (legit: {mean_legit[feat]:>8.4f}, fraud: {mean_fraud[feat]:>8.4f})")

    top_features = mean_diff.head(6).index.tolist()

    # Histogram of top 6 most discriminative features
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        axes[i].hist(df[df["Class"] == 0][feat], bins=80, alpha=0.5, color="#2ecc71", label="Legitimate", density=True)
        axes[i].hist(df[df["Class"] == 1][feat], bins=80, alpha=0.5, color="#e74c3c", label="Fraudulent", density=True)
        axes[i].set_title(f"{feat} Distribution", fontsize=13, fontweight="bold")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=9)

    fig.suptitle("Top 6 Discriminating PCA Features: Fraud vs Legitimate", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "04_top_pca_features.png")

    # Box plots for all 28 V features — gives a complete picture
    fig, axes = plt.subplots(4, 7, figsize=(28, 16))
    axes = axes.flatten()

    for i, col in enumerate(v_cols):
        data_legit = df[df["Class"] == 0][col]
        data_fraud = df[df["Class"] == 1][col]
        bp = axes[i].boxplot(
            [data_legit.values, data_fraud.values],
            tick_labels=["0", "1"],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].tick_params(labelsize=8)

    fig.suptitle("V1–V28 Feature Distributions by Class", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "05_all_v_features_boxplots.png")

    return v_cols


'''
SECTION 5: CORRELATION ANALYSIS

Pearson correlation measures the linear relationship between each
feature and the target. A value close to +1 means the feature increases
with fraud, close to -1 means it decreases with fraud, and ~0 means
no linear relationship.

This is useful because:
  1. It tells us which features are most informative for a linear model
     (Logistic Regression would rely heavily on high-|r| features)
  2. The FFNN can also learn non-linear relationships that correlation
     can't capture — so if the FFNN outperforms LR, it suggests there
     are non-linear fraud patterns in the data
  3. The heatmap shows feature-to-feature correlations — since V1-V28
     are PCA components, they should be nearly uncorrelated with each
     other (PCA produces orthogonal components by definition), which
     we can verify visually
'''

def analyse_correlations(df, v_cols):
    """
    Compute Pearson correlations between features and the Class target,
    and produce a bar chart and heatmap visualisation.
    """
    print("\n" + "=" * 60)
    print("5. CORRELATION ANALYSIS")
    print("=" * 60)

    feature_cols = v_cols + ["Amount"]
    corr_with_class = df[feature_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

    print("  Features most negatively correlated with Class (fraud):")
    for feat, corr in corr_with_class.head(5).items():
        print(f"    {feat:<8}  r = {corr:+.4f}")

    print("\n  Features most positively correlated with Class (fraud):")
    for feat, corr in corr_with_class.tail(5).items():
        print(f"    {feat:<8}  r = {corr:+.4f}")

    # Horizontal bar chart of correlations
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_with_class.values]
    ax.barh(corr_with_class.index, corr_with_class.values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pearson Correlation with Class", fontsize=12)
    ax.set_title("Feature Correlation with Fraud Label", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.tight_layout()
    save_fig(fig, "06_feature_correlation_with_class.png")

    # Full correlation heatmap — mask the upper triangle since it's symmetric
    corr_matrix = df[feature_cols + ["Class"]].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="RdBu_r", center=0,
        annot=False, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_correlation_heatmap.png")


'''
SECTION 6: PREPROCESSING — FEATURE/TARGET SEPARATION

Separate the feature matrix X (V1–V28 + Amount = 29 features) from
the target vector y (Class). The 'id' column was already dropped in
load_data(). No additional feature engineering is needed because the
V features are already PCA-transformed and scaled relative to each other.

The Amount column will be StandardScaled along with the V features
during the train/test split pipeline. Scaling ensures all features
contribute equally to gradient descent in the neural network — without
scaling, Amount (which has a different magnitude than the PCA components)
would dominate the learning process.
'''

def prepare_features(df):
    """
    Separate features (X) and target (y) from the DataFrame.
    Returns X (29 features: V1-V28 + Amount) and y (Class).
    """
    print("\n" + "=" * 60)
    print("6. PREPROCESSING — FEATURE/TARGET SEPARATION")
    print("=" * 60)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"  Feature matrix X: {X.shape}")
    print(f"  Target vector y:  {y.shape}  (fraud={y.sum():,}, legit={len(y) - y.sum():,})")
    print(f"  Features: {list(X.columns)}")

    return X, y


'''
SECTION 7: STRATIFIED K-FOLD CROSS VALIDATION SETUP

Since Dataset 2 has no Time column, we can't do time-based splitting
like Dataset 3 does. Instead we use k-Stratified Fold Cross Validation
with k=10, which is the standard approach for non-temporal classification
tasks (as recommended in the course ML Experiments material).

Stratified means each fold preserves the original fraud/legitimate ratio.
With the 50/50 balance in this dataset, stratification ensures each fold
gets roughly equal numbers of fraud and legitimate transactions.

The 10-fold setup gives us 10 independent train/test splits where each
sample appears in the test set exactly once. This produces 10 metric
measurements per algorithm, allowing us to compute mean ± std and
perform statistical significance tests.
'''

def create_stratified_splits(X, y):
    """
    Create 10-fold stratified cross-validation splits.
    Returns a list of (train_indices, test_indices) tuples.
    """
    print("\n" + "=" * 60)
    print("7. STRATIFIED K-FOLD CROSS VALIDATION SETUP")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))

    print(f"  Number of folds: {N_FOLDS}")
    print(f"  Shuffle: True (with seed={RANDOM_STATE} for reproducibility)")
    print(f"\n  {'Fold':<6} {'Train Size':>12} {'Train Fraud':>13} {'Test Size':>11} {'Test Fraud':>12}")
    print(f"  {'-' * 54}")

    for i, (train_idx, test_idx) in enumerate(splits):
        train_fraud = y.iloc[train_idx].sum()
        test_fraud = y.iloc[test_idx].sum()
        print(f"  {i + 1:<6} {len(train_idx):>12,} {train_fraud:>13,} {len(test_idx):>11,} {test_fraud:>12,}")

    return splits


'''
SECTION 8: CLASS BALANCE VERIFICATION & SCALING DEMONSTRATION (Fold 1)

In Datasets 1 and 3, this section would apply SMOTE (Synthetic Minority
Over-sampling Technique) to balance the classes. SMOTE creates synthetic
fraud samples by interpolating between existing fraud samples and their
k nearest neighbours, giving the model equal exposure to both classes.

However, Dataset 2 is ALREADY BALANCED (50/50 split), so SMOTE is
not needed. Applying SMOTE to already-balanced data would just generate
unnecessary synthetic samples that add noise without improving the
class distribution.

Instead, this section:
  1. Verifies the class balance in the training fold
  2. Demonstrates the StandardScaler transformation
  3. Visualises the training and test class distributions to confirm balance

StandardScaler is still critical here — it centres each feature to
mean=0, std=1 which ensures the neural network's gradient descent
converges efficiently (features on different scales would cause
the optimiser to oscillate rather than converge smoothly).
'''

def demonstrate_scaling(splits, X, y, colors, labels):
    """
    Demonstrate feature scaling on Fold 1 and verify class balance.
    SMOTE is NOT applied because the dataset is already balanced.
    """
    print("\n" + "=" * 60)
    print("8. FEATURE SCALING & CLASS BALANCE VERIFICATION (Fold 1)")
    print("=" * 60)

    train_idx, test_idx = splits[0]

    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # StandardScaler: fit on train, transform both
    # This centres each feature to mean=0, std=1 which is critical for
    # neural network convergence (gradient descent works better with normalised inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    train_fraud = y_train.sum()
    train_legit = len(y_train) - train_fraud
    test_fraud = y_test.sum()
    test_legit = len(y_test) - test_fraud

    print(f"  Train shape: {X_train_scaled.shape}  (fraud={train_fraud:,}, legit={train_legit:,})")
    print(f"  Test  shape: {X_test_scaled.shape}  (fraud={test_fraud:,}, legit={test_legit:,})")
    print(f"  Train fraud ratio: {train_fraud / len(y_train) * 100:.2f}%")
    print(f"  Test  fraud ratio: {test_fraud / len(y_test) * 100:.2f}%")
    print(f"\n  → Dataset is already balanced — SMOTE not applied")
    print(f"    (In Datasets 1 & 3, SMOTE would be needed here to balance <1% fraud)")

    # Training vs test class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    axes[0].bar(labels, train_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Training Set (Fold 1)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, cnt in enumerate(train_counts.values):
        axes[0].text(i, cnt + cnt * 0.01, f"{cnt:,}", ha="center", fontsize=10)

    axes[1].bar(labels, test_counts.values, color=colors, edgecolor="black")
    axes[1].set_title("Test Set (Fold 1)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, cnt in enumerate(test_counts.values):
        axes[1].text(i, cnt + cnt * 0.01, f"{cnt:,}", ha="center", fontsize=10)

    fig.suptitle("Class Balance Verification — No SMOTE Needed (Fold 1)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "08_class_balance_verification.png")


'''
SECTION 9: PREPROCESSING SUMMARY
'''

def print_preprocessing_summary(df, n_splits):
    """Print a summary of the preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("9. PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"""
  Dataset:               Kaggle Credit Card Fraud 2023 (Dataset 2)
  Source:                 kagglehub (nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
  Total transactions:    {len(df):,}
  Fraudulent:            {df['Class'].sum():,} ({df['Class'].mean() * 100:.4f}%)
  Legitimate:            {(df['Class'] == 0).sum():,} ({(1 - df['Class'].mean()) * 100:.4f}%)
  Features used:         V1–V28 + Amount (29 features)
  Features excluded:     id (sequential index, not informative)
  Scaling:               StandardScaler (fit on train fold, transform both)
  Class balancing:       Not needed (dataset is pre-balanced at 50/50)
  Validation strategy:   {n_splits}-fold Stratified Cross Validation
  Evaluation metrics:    Precision, Recall, F1, MCC, AUC-ROC

  Plots saved to: {os.path.abspath(RESULTS_DIR)}
""")
    print("  Preprocessing complete. Ready for model training.")


'''
SECTION 10: SCALE UTILITY

This is a shared utility function used by the FFNN (and any future
algorithms like LR or RF) to consistently scale features for a given
train/test split.

Why scale? Neural networks are sensitive to feature magnitudes because
gradient descent updates are proportional to feature values. If Amount
is in the thousands while V-features are near zero, the gradients for
Amount-connected weights would dominate training. StandardScaler
normalises all features to mean=0, std=1, ensuring equal contribution.

Note: Unlike Datasets 1 and 3, we do NOT apply SMOTE here because
the dataset is already balanced (50/50). SMOTE on balanced data would
just add synthetic noise without improving the class distribution.
'''

def scale_split(X, y, train_idx, test_idx):
    """
    Scale features (fit on train only).
    No SMOTE applied — dataset is already balanced.

    Returns: X_train_scaled, y_train, X_test_scaled, y_test, scaler
    """
    feature_names = X.columns.tolist()

    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_names,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index,
    )

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


'''
SECTION 11: EVALUATION METRICS

We use 5 metrics, each capturing a different aspect of model performance:

Precision:  Of all transactions the model flagged as fraud, what fraction
            actually were fraud? Low precision = lots of false alarms that
            would annoy legitimate customers.

Recall:     Of all actual fraud transactions, what fraction did the model
            catch? Low recall = missed fraud = financial losses.

F1-Score:   Harmonic mean of precision and recall — a single number that
            balances both. Heavily penalised when either is low.

MCC:        Matthews Correlation Coefficient — accounts for all four
            quadrants of the confusion matrix (TP, TN, FP, FN). Ranges
            from -1 (total disagreement) to +1 (perfect). More robust
            than F1 on imbalanced data because it considers true negatives.

AUC-ROC:    Area under the ROC curve — measures how well the model's
            probability scores separate the two classes across ALL
            possible thresholds, not just 0.5. AUC=0.5 means random,
            AUC=1.0 means perfect separation.

We still use MCC and AUC-ROC (rather than just Accuracy) for consistency
with Datasets 1 and 3, where Accuracy IS misleading due to severe
class imbalance. On this balanced dataset, Accuracy would actually be
a fair metric, but MCC and AUC-ROC provide richer information about
model performance regardless of class distribution.
'''

def compute_metrics(y_true, y_pred, y_prob):
    """Compute the five core evaluation metrics and return as a dict."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_prob),
    }


def print_classification_report_wrapper(y_true, y_pred):
    """Print sklearn's classification report with Legitimate / Fraudulent labels."""
    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
    ))


def print_summary_table(results_df, model_name):
    """
    Print mean ± std of each metric across all folds.
    Mean = overall expected performance.
    Std = how stable/consistent the model is across different folds (low std = good).
    """
    print(f"\n  ── {model_name} Summary Across All Folds ──")
    print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
    print(f"  {'-' * 32}")
    for metric in ["precision", "recall", "f1", "mcc", "auc_roc"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric:<12} {mean_val:>10.4f} {std_val:>10.4f}")


'''
SECTION 12: VISUALISATION HELPERS

These functions generate the standard set of visualisations used
to evaluate each model. They are designed to be model-agnostic —
they take predictions as input rather than a trained model, so
they can be reused for LR, RF, FFNN, or any other algorithm.
'''

def plot_confusion_matrices(splits, X, y, predictions_by_fold, model_name, filename):
    """
    Plot confusion matrices for each fold.
    Each heatmap is a 2×2 grid:
      Top-left (TN):     Legitimate correctly classified — should be large
      Top-right (FP):    Legitimate wrongly flagged as fraud — false alarms
      Bottom-left (FN):  Fraud that slipped through — want this near 0
      Bottom-right (TP): Fraud correctly caught — want this high
    """
    n_to_show = min(5, len(splits))  # Show at most 5 to keep the figure readable
    fig, axes = plt.subplots(1, n_to_show, figsize=(5 * n_to_show, 4))
    if n_to_show == 1:
        axes = [axes]

    for fold_num in range(n_to_show):
        _, test_idx = splits[fold_num]
        y_test = y.iloc[test_idx]
        y_pred = predictions_by_fold[fold_num + 1]["y_pred"]

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[fold_num],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[fold_num].set_title(f"Fold {fold_num + 1}", fontsize=12, fontweight="bold")
        axes[fold_num].set_ylabel("Actual")
        axes[fold_num].set_xlabel("Predicted")

    fig.suptitle(f"{model_name} – Confusion Matrices (Folds 1–{n_to_show})", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_roc_curves(splits, X, y, predictions_by_fold, model_name, filename):
    """
    Plot ROC curves for all folds overlaid on one plot.
    X-axis (FPR): fraction of legit transactions wrongly flagged
    Y-axis (TPR/Recall): fraction of fraud correctly caught
    The dashed diagonal = random classifier (AUC=0.5)
    Curves bowing toward top-left = better performance
    Tightly clustered curves = consistent performance across folds
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for fold_num in range(len(splits)):
        _, test_idx = splits[fold_num]
        y_test = y.iloc[test_idx]
        y_prob = predictions_by_fold[fold_num + 1]["y_prob"]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"Fold {fold_num + 1} (AUC={auc_val:.4f})", alpha=0.7)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} – ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_metrics_bars(results_df, splits, model_name, filename):
    """
    Grouped bar chart of all 5 metrics across folds.
    Bars at roughly the same height across folds = stable model.
    Short precision but tall recall = catching fraud but many false positives.
    """
    n_to_show = min(5, len(splits))
    fig, ax = plt.subplots(figsize=(12, 5))

    metrics_to_plot = ["precision", "recall", "f1", "mcc", "auc_roc"]
    x = np.arange(n_to_show)
    width = 0.15
    colors_metrics = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for i, metric in enumerate(metrics_to_plot):
        vals = results_df[metric].values[:n_to_show]
        ax.bar(x + i * width, vals, width, label=metric.upper().replace("_", "-"), color=colors_metrics[i])

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{model_name} – Metrics by Fold", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_to_show)])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, filename)


'''
SECTION 13: FEED-FORWARD NEURAL NETWORK — MODEL DEFINITION

The FFNN is a non-linear multi-layer classifier. Unlike Logistic
Regression (single linear layer + sigmoid), the FFNN stacks multiple
layers with non-linear activations (ReLU), allowing it to learn
complex decision boundaries that a linear model would miss.

Architecture:
  Input(29) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(1)  → Sigmoid

BatchNorm normalises each mini-batch's activations to zero mean and
unit variance, which stabilises training and acts as mild regularisation.

Dropout randomly zeroes 30% of neurons during training, forcing the
network to learn redundant representations — this prevents overfitting,
especially when training on a large dataset (568K+ samples).

The output is a single neuron with Sigmoid activation, producing a
probability in [0,1]. Transactions with probability >= 0.5 are
classified as fraud.
'''

class FraudDetectorNN(nn.Module):
    """
    Feed-forward neural network for binary fraud classification.

    Architecture:
      Input(29) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
                → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
                → Linear(1)  → Sigmoid
    """

    def __init__(self, input_dim, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT):
        super().__init__()
        self.network = nn.Sequential(
            # Hidden layer 1: maps 29 features → 64 hidden units
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),                      # ReLU(z) = max(0, z) — introduces non-linearity
            nn.Dropout(dropout),

            # Hidden layer 2: maps 64 → 32 hidden units (compression)
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer: maps 32 → 1 (fraud probability)
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),                   # σ(z) = 1/(1+e^(-z)) — squashes to [0, 1]
        )

    def forward(self, x):
        """Forward pass: push input through all layers and return fraud probability."""
        return self.network(x).squeeze(-1)  # squeeze removes trailing dim: (batch,1) → (batch,)


'''
SECTION 14: FFNN — TRAINING FUNCTION

Training uses mini-batch gradient descent with the Adam optimiser.
Adam maintains per-parameter adaptive learning rates using first
and second moment estimates of the gradients — it converges faster
than vanilla SGD (as discussed in the course FFNN material).

The training loop for each batch:
  1. Forward pass: compute ŷ = model(X_batch)
  2. Compute loss: BCE(ŷ, y_batch) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  3. Backward pass: compute ∂L/∂W for all weights via backpropagation
     (chain rule applied through all layers — this is the key insight
     from the FFNN lecture material)
  4. Update step: W ← W - α · ∂L/∂W  (Adam uses adaptive α)

Weight decay (L2 regularisation) is applied directly in the Adam
optimiser, which is mathematically equivalent to adding λ||W||₂²
to the loss function. This penalises large weights and reduces
overfitting — larger λ = more regularisation = simpler model.
'''

def _set_seeds():
    """
    Set all random seeds for reproducibility.
    PyTorch uses separate RNG states for CPU and CUDA operations.
    We set both plus numpy's seed to ensure identical results across runs.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def train_nn(X_train, y_train, input_dim, epochs=EPOCHS, batch_size=BATCH_SIZE,
             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """
    Train a FFNN on the given training data.

    Returns: trained model, list of epoch losses (for training loss curve)
    """
    _set_seeds()

    model = FraudDetectorNN(input_dim)
    criterion = nn.BCELoss()  # Binary Cross-Entropy: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Convert pandas/numpy data to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, "values") else np.array(X_train))
    y_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, "values") else np.array(y_train))

    dataset = TensorDataset(X_tensor, y_tensor)
    # Generator with fixed seed ensures same shuffle order across runs
    g = torch.Generator().manual_seed(RANDOM_STATE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    epoch_losses = []
    model.train()  # enable dropout and batch norm training mode

    for epoch in range(epochs):
        running_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()             # reset gradients from previous batch
            y_hat = model(X_batch)            # forward pass
            loss = criterion(y_hat, y_batch)  # compute BCE loss
            loss.backward()                   # backward pass — backpropagation
            optimizer.step()                  # update weights using Adam

            running_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)

        avg_loss = running_loss / n_samples
        epoch_losses.append(avg_loss)

        # Print progress every 10 epochs (and the first) to track convergence
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch + 1:>3}/{epochs}  |  Loss: {avg_loss:.6f}")

    return model, epoch_losses


'''
SECTION 15: FFNN — PREDICTION FUNCTION

model.eval() is critical during inference — it:
  - Disables dropout (all neurons active, no random zeroing)
  - Switches BatchNorm to use population statistics (accumulated
    during training) instead of batch statistics

torch.no_grad() disables gradient tracking, reducing memory usage
and speeding up inference since we don't need gradients for prediction.

Threshold is 0.5: probability >= 0.5 → fraud, < 0.5 → legitimate.
'''

def predict_nn(model, X_test):
    """
    Get predictions and probabilities from a trained model.
    Returns: y_pred (binary), y_prob (continuous probabilities in [0,1])
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, "values") else np.array(X_test))
        y_prob = model(X_tensor).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


'''
SECTION 16: FFNN — TRAINING & EVALUATION ACROSS ALL FOLDS

This is the main experiment loop. For each of the 10 stratified folds:
  1. Scale features (StandardScaler fit on train, transform both)
  2. Train a fresh FFNN from scratch (weights reinitialised each fold)
  3. Predict on the scaled test set
  4. Compute all 5 evaluation metrics
  5. Store predictions for later visualisation (avoids expensive re-training)

No SMOTE is applied — the dataset is already balanced (50/50).

A fresh model is trained per fold because:
  - Each fold has different train/test data
  - Carrying over weights would bias the model toward earlier folds
  - Fresh training ensures each fold is an independent evaluation
'''

def train_and_evaluate_nn(splits, X, y):
    """
    Train & evaluate FFNN across all stratified folds.

    Returns:
      - nn_df: DataFrame of metrics per fold
      - fold_predictions: dict mapping fold_num → {y_pred, y_prob}
      - all_epoch_losses: dict mapping fold_num → list of training losses per epoch
    """
    print("\n" + "=" * 60)
    print("10. FEED-FORWARD NEURAL NETWORK — TRAINING & EVALUATION")
    print("=" * 60)

    input_dim = X.shape[1]  # 29 features: V1–V28 + Amount

    print(f"\n  Architecture: Input({input_dim}) → Dense({HIDDEN1}, ReLU, BN, Drop) "
          f"→ Dense({HIDDEN2}, ReLU, BN, Drop) → Dense(1, Sigmoid)")
    print(f"  Optimiser:    Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"  Loss:         Binary Cross-Entropy")
    print(f"  Epochs: {EPOCHS}  |  Batch Size: {BATCH_SIZE}  |  Dropout: {DROPOUT}")

    nn_results = []
    fold_predictions = {}
    all_epoch_losses = {}

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n  ── Fold {fold_num}/{N_FOLDS} ──")

        # Scale features using the shared utility (no SMOTE — dataset is balanced)
        X_train_scaled, y_train, X_test_scaled, y_test, _ = scale_split(
            X, y, train_idx, test_idx,
        )

        print(f"    Train: {len(X_train_scaled):,} samples  |  Test: {len(X_test_scaled):,} samples")

        # Train a fresh model for this fold
        model, epoch_losses = train_nn(X_train_scaled, y_train, input_dim)
        y_pred, y_prob = predict_nn(model, X_test_scaled)

        # Compute the 5 evaluation metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_num
        nn_results.append(metrics)

        # Store predictions to avoid re-training during visualisation
        fold_predictions[fold_num] = {"y_pred": y_pred, "y_prob": y_prob}
        all_epoch_losses[fold_num] = epoch_losses

        print(f"    Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}  |  "
              f"F1: {metrics['f1']:.4f}  |  MCC: {metrics['mcc']:.4f}  |  AUC-ROC: {metrics['auc_roc']:.4f}")
        print_classification_report_wrapper(y_test, y_pred)

    # Save results to CSV for later cross-algorithm comparison
    nn_df = pd.DataFrame(nn_results)
    nn_df.to_csv(os.path.join(RESULTS_DIR, "nn_results.csv"), index=False)
    print_summary_table(nn_df, "Feed-Forward Neural Network")

    return nn_df, fold_predictions, all_epoch_losses


'''
SECTION 17: FFNN — VISUALISATIONS

Generate all FFNN visualisation artefacts:
  1. Training loss curves per fold — shows convergence (does the loss flatten?)
     If curves flatten early, the model converges quickly. If they keep dropping,
     more epochs might help. Comparing across folds: all should converge similarly
     since each fold is a random partition (unlike Dataset 3 where later splits
     have more training data due to the expanding window).
  2. Confusion matrices — shows the FP/FN tradeoff at each fold
  3. ROC curves overlaid — shows AUC consistency across folds
  4. Metrics bar chart — quick visual comparison of all 5 metrics

Unlike Logistic Regression, the FFNN does not produce directly interpretable
feature coefficients (no single weight vector mapping features to log-odds).
This "black box" nature is a key discussion point in the comparative analysis.
'''

def generate_nn_visualisations(splits, X, y, nn_df, fold_predictions, all_epoch_losses):
    """Generate all FFNN visualisation artefacts."""
    print("\n" + "=" * 60)
    print("11. FEED-FORWARD NEURAL NETWORK — VISUALISATIONS")
    print("=" * 60)

    # ── Training Loss Curves ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_splits = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 folds

    for fold_num in sorted(all_epoch_losses.keys()):
        losses = all_epoch_losses[fold_num]
        color = colors_splits[(fold_num - 1) % len(colors_splits)]
        ax.plot(range(1, len(losses) + 1), losses,
                label=f"Fold {fold_num}", color=color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_title("Feed-Forward Neural Network – Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "09_nn_training_loss.png")

    # ── Confusion Matrices (first 5 folds) ──
    plot_confusion_matrices(splits, X, y, fold_predictions,
                            "Feed-Forward Neural Network", "10_nn_confusion_matrices.png")

    # ── ROC Curves (all folds) ──
    plot_roc_curves(splits, X, y, fold_predictions,
                    "Feed-Forward Neural Network", "11_nn_roc_curves.png")

    # ── Metrics Bar Chart (first 5 folds) ──
    plot_metrics_bars(nn_df, splits, "Feed-Forward Neural Network", "12_nn_metrics_by_fold.png")

    print("\n  Feed-Forward Neural Network visualisations complete.")


'''
SECTION 18: MAIN — ORCHESTRATE THE FULL PIPELINE
'''

if __name__ == "__main__":
    # -- 1. Data Loading --
    df = load_data()

    # -- 2–5. Exploratory Data Analysis --
    colors, labels = analyse_class_distribution(df)
    analyse_amounts(df)
    v_cols = analyse_pca_features(df)
    analyse_correlations(df, v_cols)

    # -- 6. Preprocessing --
    X, y = prepare_features(df)

    # -- 7. Cross-Validation Setup --
    splits = create_stratified_splits(X, y)

    # -- 8. Scaling & Class Balance Verification --
    demonstrate_scaling(splits, X, y, colors, labels)

    # -- 9. Summary --
    print_preprocessing_summary(df, N_FOLDS)

    # -- 10–11. Feed-Forward Neural Network --
    nn_df, fold_predictions, all_epoch_losses = train_and_evaluate_nn(splits, X, y)
    generate_nn_visualisations(splits, X, y, nn_df, fold_predictions, all_epoch_losses)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  All results saved to: {os.path.abspath(RESULTS_DIR)}")
