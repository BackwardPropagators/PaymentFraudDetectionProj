"""
Data loading, cleaning, exploratory analysis, and preprocessing for Dataset 3.

notes:
- Gotta print out stuff to show progress as it goes along -  add to top
- I had to look up a million different things for this, I will try to find and add the links to the individual areas as needed

Outlining what I have to do and how I intend to do it as I go along so i dont get sidetracked
1. Load the dataset and figure out some general stuff about it (shape, types, missing values, duplicates)
   Decide On my diagrams and what I want to show with them (class distribution, time distribution, amount distribution, PCA features, correlation)
2. Class distribution analysis (bar chart + pie chart)

To Do's:
Maybe make the iamage analysis less scrunched up
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.utils import DATA_PATH, RESULTS_DIR, RANDOM_STATE, save_fig, create_time_splits


def load_and_clean():
    """
    Load Dataset 3, convert types, drop duplicates.
    Returns the cleaned DataFrame.
    """
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # Class column arrives as quoted strings ("0" / "1") – convert to int
    df["Class"] = df["Class"].astype(int)

    print(f"Shape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")

    # Check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values: {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates:,}")
    if duplicates > 0:
        print("  → Dropping duplicates")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  → New shape: {df.shape[0]:,} rows")

    return df


def analyse_class_distribution(df):
    """
    Class Distribution:
    - Show the extreme imbalance (fraudulent transactions are a tiny fraction of the dataset)
    - Bar chart of class counts with percentages annotated
    - Pie chart to visually reinforce the imbalance
    """
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = df["Class"].value_counts().sort_index()
    class_pct = df["Class"].value_counts(normalize=True).sort_index() * 100

    print(f"  Legitimate (0): {class_counts[0]:>8,}  ({class_pct[0]:.4f}%)")
    print(f"  Fraudulent (1): {class_counts[1]:>8,}  ({class_pct[1]:.4f}%)")
    print(f"  Imbalance ratio: 1 : {class_counts[0] / class_counts[1]:.0f}")

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Class Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, (cnt, pct) in enumerate(zip(class_counts.values, class_pct.values)):
        axes[0].text(i, cnt + cnt * 0.01, f"{cnt:,}\n({pct:.3f}%)", ha="center", fontsize=10)

    axes[1].pie(class_counts.values, labels=labels, colors=colors, autopct="%1.3f%%", startangle=90, explode=(0, 0.1))
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    fig.suptitle("Severe Class Imbalance in Dataset 3", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "01_class_distribution.png")

    return colors, labels


def analyse_time_features(df):
    """
    A lil note : I never did take on the time column before becasue I just saw the 1 through however much,
    but when I started making this I realized that there are repeats?, it took me a good minute but I realized
    that its just transactions every second after the first transaction until the 172k seconds ish mark, which
    is about 48 hours. Based on this, since Mr Manohar suggested a sectioning out based on time, I decided to
    section it out hour to hour and do training vs testing based on that, maybe 2:1?

    Time Feature Analysis:
    - Convert Time from seconds to hours for better interpretability
    - Plot transaction volume over time (histogram of transactions by hour)
    - Plot fraud rate by hour to see if there are temporal patterns in fraudulent activity
    """
    print("\n" + "=" * 60)
    print("3. TIME FEATURE ANALYSIS")
    print("=" * 60)

    df["Hour"] = df["Time"] / 3600
    total_hours = df["Hour"].max()
    print(f"  Time range: 0 – {df['Time'].max():,.0f} seconds  ({total_hours:.1f} hours ≈ {total_hours / 24:.1f} days)")

    # ── Transaction volume over time ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].hist(df["Hour"], bins=48, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].set_title("All Transactions Over Time", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")

    fraud_hours = df[df["Class"] == 1]["Hour"]
    axes[1].hist(fraud_hours, bins=48, color="#e74c3c", edgecolor="black", alpha=0.7)
    axes[1].set_title("Fraudulent Transactions Over Time", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].set_xlabel("Time (Hours)")

    fig.suptitle("Transaction Volume Distribution Over ~48 Hours", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_time_distribution.png")


    df["HourBin"] = df["Hour"].astype(int)
    hourly_stats = df.groupby("HourBin").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
    ).reset_index()
    hourly_stats["fraud_rate"] = hourly_stats["fraud"] / hourly_stats["total"] * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.bar(hourly_stats["HourBin"], hourly_stats["total"], color="#3498db", alpha=0.4, label="Total Transactions")
    ax2.plot(hourly_stats["HourBin"], hourly_stats["fraud_rate"], color="#e74c3c", marker="o", linewidth=2, label="Fraud Rate (%)")

    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Transaction Count", fontsize=12, color="#3498db")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=12, color="#e74c3c")
    ax1.set_title("Transaction Volume vs Fraud Rate by Hour", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    save_fig(fig, "03_fraud_rate_by_hour.png")

    return total_hours


def analyse_amounts(df):
    """
    Amount Distribution Analysis.
    """
    print("\n" + "=" * 60)
    print("4. TRANSACTION AMOUNT ANALYSIS")
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
    save_fig(fig, "04_amount_distribution.png")



    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(legit), bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    ax.hist(np.log1p(fraud), bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    ax.set_title("Log-Transformed Amount Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "05_amount_log_distribution.png")


def analyse_pca_features(df):
    """
    Feature Analysis:
    - Since the V1-V28 features are PCA components, we can analyze which ones differ most
      between fraudulent and legitimate transactions by comparing their distributions.
    - After running this, I found that V14, V17, V12, V10, V11, and V3 had the largest mean
      differences between fraud and legit transactions.
    """
    print("\n" + "=" * 60)
    print("5. PCA FEATURE ANALYSIS (V1–V28)")
    print("=" * 60)

    v_cols = [f"V{i}" for i in range(1, 29)]

    mean_legit = df[df["Class"] == 0][v_cols].mean()
    mean_fraud = df[df["Class"] == 1][v_cols].mean()
    mean_diff = (mean_fraud - mean_legit).abs().sort_values(ascending=False)

    print("  Top PCA features by |mean(fraud) – mean(legit)|:")
    for feat, diff in mean_diff.head(10).items():
        print(f"    {feat:<5}  Δ = {diff:.4f}  (legit: {mean_legit[feat]:>8.4f}, fraud: {mean_fraud[feat]:>8.4f})")

    top_features = mean_diff.head(6).index.tolist()

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
    save_fig(fig, "06_top_pca_features.png")



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
    save_fig(fig, "07_all_v_features_boxplots.png")

    return v_cols


def analyse_correlations(df, v_cols):
    """
    Pearson correlation coefficient analysis.
    """
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS")
    print("=" * 60)

    feature_cols = v_cols + ["Amount"]
    corr_with_class = df[feature_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

    print("  Features most negatively correlated with Class (fraud):")
    for feat, corr in corr_with_class.head(5).items():
        print(f"    {feat:<8}  r = {corr:+.4f}")

    print("\n  Features most positively correlated with Class (fraud):")
    for feat, corr in corr_with_class.tail(5).items():
        print(f"    {feat:<8}  r = {corr:+.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_with_class.values]
    ax.barh(corr_with_class.index, corr_with_class.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pearson Correlation with Class", fontsize=12)
    ax.set_title("Feature Correlation with Fraud Label", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.tight_layout()
    save_fig(fig, "08_feature_correlation_with_class.png")

    corr_matrix = df[feature_cols + ["Class"]].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "09_correlation_heatmap.png")


def prepare_features(df):
    """
    Preprocessing Pipeline:
    - Drop any helper columns created during analysis (e.g., Hour, HourBin)
    - Separate features (X) and target variable (y)
    - Keep Time for time-based splitting but exclude it from the feature set used for training.
    """
    print("\n" + "=" * 60)
    print("7. PREPROCESSING PIPELINE")
    print("=" * 60)

    df.drop(columns=["Hour", "HourBin"], inplace=True, errors="ignore")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"  Feature matrix X: {X.shape}")
    print(f"  Target vector y:  {y.shape}  (fraud={y.sum():,}, legit={len(y) - y.sum():,})")

    return X, y


def print_split_summary(splits, X, y):
    """Print summary table of time-based splits."""
    hours = X["Time"] / 3600
    total_hours = hours.max()

    print("\n" + "=" * 60)
    print("8. TIME-BASED TRAIN / TEST SPLITTING")
    print("=" * 60)

    print(f"  Total time span: {total_hours:.1f} hours")
    print(f"  Training starts at: 16h | Test window: 8h")
    print(f"  Number of time-based splits: {len(splits)}\n")

    print(f"  {'Split':<8} {'Train Size':>12} {'Train Fraud':>13} {'Test Size':>11} {'Test Fraud':>12} {'Train Hours':>13} {'Test Hours':>12}")
    print(f"  {'-' * 81}")

    for i, (train_idx, test_idx) in enumerate(splits):
        train_fraud = y.loc[train_idx].sum()
        test_fraud = y.loc[test_idx].sum()
        train_hours_range = f"0–{hours.loc[train_idx].max():.0f}h"
        test_hours_range = f"{hours.loc[test_idx].min():.0f}–{hours.loc[test_idx].max():.0f}h"
        print(
            f"  {i + 1:<8} {len(train_idx):>12,} {train_fraud:>13,} "
            f"{len(test_idx):>11,} {test_fraud:>12,} {train_hours_range:>13} {test_hours_range:>12}"
        )


    fig, ax = plt.subplots(figsize=(14, 4))

    for i, (train_idx, test_idx) in enumerate(splits):
        train_end = hours.loc[train_idx].max()
        test_min = hours.loc[test_idx].min()
        test_max = hours.loc[test_idx].max()

        ax.barh(i, train_end, height=0.5, color="#3498db", alpha=0.7, label="Train" if i == 0 else "")
        ax.barh(i, test_max - test_min, left=test_min, height=0.5, color="#e74c3c", alpha=0.7, label="Test" if i == 0 else "")

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Split {i + 1}" for i in range(len(splits))])
    ax.set_xlabel("Time (Hours)", fontsize=12)
    ax.set_title("Expanding Window Time-Based Train/Test Splits", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "10_time_based_splits.png")


def demonstrate_smote(splits, X, y, colors, labels):
    """
    Feature Scaling & SMOTE Demonstration (Split 1).

    What SMOTE does: Synthetic Minority Over-sampling Technique. For each fraud sample,
    it picks a random nearest neighbour (also fraud), draws a random point on the line
    segment between them, and adds that synthetic point.

    Why train only? If you SMOTE the test set, you'd be evaluating the model on synthetic
    data that doesn't exist in reality.
    """
    print("\n" + "=" * 60)
    print("9. FEATURE SCALING & SMOTE DEMONSTRATION (Split 1)")
    print("=" * 60)

    X_features = X.drop(columns=["Time"])
    feature_names = X_features.columns.tolist()

    train_idx, test_idx = splits[0]

    X_train_raw = X_features.loc[train_idx]
    X_test_raw = X_features.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

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

    print(f"  Train shape before SMOTE: {X_train_scaled.shape}  (fraud={y_train.sum()}, legit={len(y_train) - y_train.sum()})")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"  Train shape after  SMOTE: {X_train_resampled.shape}  (fraud={y_train_resampled.sum()}, legit={len(y_train_resampled) - y_train_resampled.sum()})")
    print(f"  Test  shape (untouched):  {X_test_scaled.shape}  (fraud={y_test.sum()}, legit={len(y_test) - y_test.sum()})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    before_counts = y_train.value_counts().sort_index()
    after_counts = pd.Series(y_train_resampled).value_counts().sort_index()

    axes[0].bar(labels, before_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Training Set BEFORE SMOTE", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, cnt in enumerate(before_counts.values):
        axes[0].text(i, cnt + cnt * 0.01, f"{cnt:,}", ha="center", fontsize=10)

    axes[1].bar(labels, after_counts.values, color=colors, edgecolor="black")
    axes[1].set_title("Training Set AFTER SMOTE", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, cnt in enumerate(after_counts.values):
        axes[1].text(i, cnt + cnt * 0.01, f"{cnt:,}", ha="center", fontsize=10)

    fig.suptitle("SMOTE Resampling Effect on Class Balance (Split 1)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "11_smote_effect.png")


def print_preprocessing_summary(df, total_hours, n_splits):
    """Print final preprocessing summary."""
    import os
    print("\n" + "=" * 60)
    print("10. PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"""
  Dataset:               Kaggle Credit Card Fraud (Dataset 3)
  Total transactions:    {len(df):,}
  Fraudulent:            {df['Class'].sum():,} ({df['Class'].mean() * 100:.4f}%)
  Legitimate:            {(df['Class'] == 0).sum():,} ({(1 - df['Class'].mean()) * 100:.4f}%)
  Time span:             ~{total_hours:.0f} hours (~{total_hours / 24:.0f} days)
  Features used:         V1–V28 + Amount (29 features)
  Features excluded:     Time (used for splitting only)
  Scaling:               StandardScaler (fit on train, transform both)
  Class balancing:       SMOTE (applied to training folds only)
  Validation strategy:   Expanding-window time-based split ({n_splits} splits)
  Evaluation metrics:    Precision, Recall, F1, MCC, AUC-ROC

  Plots saved to: {os.path.abspath(RESULTS_DIR)}
""")
    print("Preprocessing complete. Ready for model training.")


def run_preprocessing():
    """
    Execute the full preprocessing pipeline.
    Returns: df, X, y, splits, total_hours, colors, labels
    """
    df = load_and_clean()
    colors, labels = analyse_class_distribution(df)
    total_hours = analyse_time_features(df)
    analyse_amounts(df)
    v_cols = analyse_pca_features(df)
    analyse_correlations(df, v_cols)
    X, y = prepare_features(df)
    splits = create_time_splits(X)
    print_split_summary(splits, X, y)
    demonstrate_smote(splits, X, y, colors, labels)
    print_preprocessing_summary(df, total_hours, len(splits))

    return df, X, y, splits, total_hours
