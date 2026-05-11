import os

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from src.utils import DATA_PATH, RANDOM_STATE, RESULTS_DIR, create_time_splits, save_fig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean():
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    data = pd.read_csv(DATA_PATH)
    data["Class"] = data["Class"].astype(int)

    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Column types:\n{data.dtypes.value_counts()}")
    print(f"First 5 rows:\n{data.head()}")
    print(f"Basic statistics:\n{data.describe()}")

    missing = data.isnull().sum()
    duplicate_count = data.duplicated().sum()

    print(f"Missing values: {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])

    print(f"Duplicate rows: {duplicate_count:,}")
    if duplicate_count > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"New shape: {data.shape[0]:,} rows")

    return data


def analyse_class_distribution(data):
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = data["Class"].value_counts().sort_index()
    class_percentages = data["Class"].value_counts(normalize=True).sort_index() * 100
    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    print(f"Legitimate: {class_counts[0]:,} ({class_percentages[0]:.4f}%)")
    print(f"Fraudulent: {class_counts[1]:,} ({class_percentages[1]:.4f}%)")
    print(f"Imbalance ratio: 1 : {class_counts[0] / class_counts[1]:.0f}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Class Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")

    for position, (count, percentage) in enumerate(zip(class_counts.values, class_percentages.values)):
        axes[0].text(position, count + count * 0.01, f"{count:,}\n({percentage:.3f}%)", ha="center", fontsize=10)

    axes[1].pie(class_counts.values, labels=labels, colors=colors, autopct="%1.3f%%", startangle=90, explode=(0, 0.1))
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    figure.suptitle("Class Imbalance in Dataset 3", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "01_class_distribution.png")

    return colors, labels


def analyse_time_features(data):
    print("\n" + "=" * 60)
    print("3. TIME FEATURE ANALYSIS")
    print("=" * 60)

    data["Hour"] = data["Time"] / 3600
    total_hours = data["Hour"].max()
    print(f"Time range: 0 - {data['Time'].max():,.0f} seconds ({total_hours:.1f} hours, {total_hours / 24:.1f} days)")

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].hist(data["Hour"], bins=48, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].set_title("All Transactions Over Time", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")

    fraud_hours = data[data["Class"] == 1]["Hour"]
    axes[1].hist(fraud_hours, bins=48, color="#e74c3c", edgecolor="black", alpha=0.7)
    axes[1].set_title("Fraudulent Transactions Over Time", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].set_xlabel("Time (Hours)")

    figure.suptitle("Transaction Volume Distribution Over About 48 Hours", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "02_time_distribution.png")

    data["HourBin"] = data["Hour"].astype(int)
    hourly_stats = data.groupby("HourBin").agg(total=("Class", "count"), fraud=("Class", "sum")).reset_index()
    hourly_stats["fraud_rate"] = hourly_stats["fraud"] / hourly_stats["total"] * 100

    figure, volume_axis = plt.subplots(figsize=(14, 5))
    rate_axis = volume_axis.twinx()
    volume_axis.bar(hourly_stats["HourBin"], hourly_stats["total"], color="#3498db", alpha=0.4, label="Total Transactions")
    rate_axis.plot(hourly_stats["HourBin"], hourly_stats["fraud_rate"], color="#e74c3c", marker="o", linewidth=2, label="Fraud Rate (%)")
    volume_axis.set_xlabel("Hour", fontsize=12)
    volume_axis.set_ylabel("Transaction Count", fontsize=12, color="#3498db")
    rate_axis.set_ylabel("Fraud Rate (%)", fontsize=12, color="#e74c3c")
    volume_axis.set_title("Transaction Volume vs Fraud Rate by Hour", fontsize=14, fontweight="bold")

    volume_handles, volume_labels = volume_axis.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    volume_axis.legend(volume_handles + rate_handles, volume_labels + rate_labels, loc="upper right")

    figure.tight_layout()
    save_fig(figure, "03_fraud_rate_by_hour.png")

    return total_hours


def analyse_amounts(data):
    print("\n" + "=" * 60)
    print("4. TRANSACTION AMOUNT ANALYSIS")
    print("=" * 60)

    legitimate_amounts = data[data["Class"] == 0]["Amount"]
    fraud_amounts = data[data["Class"] == 1]["Amount"]
    rows = [
        ("Mean", legitimate_amounts.mean(), fraud_amounts.mean()),
        ("Median", legitimate_amounts.median(), fraud_amounts.median()),
        ("Std Dev", legitimate_amounts.std(), fraud_amounts.std()),
        ("Min", legitimate_amounts.min(), fraud_amounts.min()),
        ("Max", legitimate_amounts.max(), fraud_amounts.max()),
        ("25th Percentile", legitimate_amounts.quantile(0.25), fraud_amounts.quantile(0.25)),
        ("75th Percentile", legitimate_amounts.quantile(0.75), fraud_amounts.quantile(0.75)),
    ]

    print(f"{'Metric':<20} {'Legitimate':>14} {'Fraudulent':>14}")
    print("-" * 50)
    for label, legitimate_value, fraud_value in rows:
        print(f"{label:<20} {legitimate_value:>14.2f} {fraud_value:>14.2f}")

    figure, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].hist(legitimate_amounts, bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axes[0].hist(fraud_amounts, bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axes[0].set_title("Amount Distribution (Full Range)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(legitimate_amounts[legitimate_amounts <= 500], bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axes[1].hist(fraud_amounts[fraud_amounts <= 500], bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axes[1].set_title("Amount Distribution (<= $500)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    figure.suptitle("Transaction Amount: Legitimate vs Fraudulent", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "04_amount_distribution.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(np.log1p(legitimate_amounts), bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axis.hist(np.log1p(fraud_amounts), bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axis.set_title("Log-Transformed Amount Distribution", fontsize=14, fontweight="bold")
    axis.set_xlabel("log(1 + Amount)")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    save_fig(figure, "05_amount_log_distribution.png")


def analyse_pca_features(data):
    print("\n" + "=" * 60)
    print("5. PCA FEATURE ANALYSIS")
    print("=" * 60)

    pca_columns = [f"V{number}" for number in range(1, 29)]
    legitimate_means = data[data["Class"] == 0][pca_columns].mean()
    fraud_means = data[data["Class"] == 1][pca_columns].mean()
    mean_differences = (fraud_means - legitimate_means).abs().sort_values(ascending=False)

    print("Top PCA features by mean difference:")
    for feature_name, difference in mean_differences.head(10).items():
        print(f"  {feature_name:<5} {difference:.4f}")

    top_features = mean_differences.head(6).index.tolist()
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for axis, feature_name in zip(axes, top_features):
        axis.hist(data[data["Class"] == 0][feature_name], bins=80, alpha=0.5, color="#2ecc71", label="Legitimate", density=True)
        axis.hist(data[data["Class"] == 1][feature_name], bins=80, alpha=0.5, color="#e74c3c", label="Fraudulent", density=True)
        axis.set_title(f"{feature_name} Distribution", fontsize=13, fontweight="bold")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Density")
        axis.legend(fontsize=9)

    figure.suptitle("Top PCA Features: Fraud vs Legitimate", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "06_top_pca_features.png")

    figure, axes = plt.subplots(4, 7, figsize=(28, 16))
    axes = axes.flatten()

    for axis, feature_name in zip(axes, pca_columns):
        boxplot = axis.boxplot(
            [data[data["Class"] == 0][feature_name].values, data[data["Class"] == 1][feature_name].values],
            labels=["0", "1"],
            patch_artist=True,
            widths=0.6,
        )
        boxplot["boxes"][0].set_facecolor("#2ecc71")
        boxplot["boxes"][1].set_facecolor("#e74c3c")
        axis.set_title(feature_name, fontsize=11, fontweight="bold")
        axis.tick_params(labelsize=8)

    figure.suptitle("V1-V28 Feature Distributions by Class", fontsize=18, fontweight="bold", y=1.01)
    figure.tight_layout()
    save_fig(figure, "07_all_v_features_boxplots.png")

    return pca_columns


def analyse_correlations(data, pca_columns):
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS")
    print("=" * 60)

    feature_columns = pca_columns + ["Amount"]
    correlations = data[feature_columns + ["Class"]].corr()["Class"].drop("Class").sort_values()

    print("Most negative correlations:")
    for feature_name, correlation in correlations.head(5).items():
        print(f"  {feature_name:<8} {correlation:+.4f}")

    print("Most positive correlations:")
    for feature_name, correlation in correlations.tail(5).items():
        print(f"  {feature_name:<8} {correlation:+.4f}")

    figure, axis = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if value < 0 else "#2ecc71" for value in correlations.values]
    axis.barh(correlations.index, correlations.values, color=colors, edgecolor="black", linewidth=0.5)
    axis.set_xlabel("Pearson Correlation with Class", fontsize=12)
    axis.set_title("Feature Correlation with Fraud Label", fontsize=14, fontweight="bold")
    axis.axvline(x=0, color="black", linewidth=0.8)
    figure.tight_layout()
    save_fig(figure, "08_feature_correlation_with_class.png")

    correlation_matrix = data[feature_columns + ["Class"]].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    figure, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    figure.tight_layout()
    save_fig(figure, "09_correlation_heatmap.png")


def prepare_features(data):
    print("\n" + "=" * 60)
    print("7. PREPROCESSING")
    print("=" * 60)

    data = data.drop(columns=["Hour", "HourBin"], errors="ignore")
    features = data.drop(columns=["Class"])
    target = data["Class"]

    print(f"Feature matrix: {features.shape}")
    print(f"Target vector: {target.shape} (fraud={target.sum():,}, legit={len(target) - target.sum():,})")

    return features, target


def print_split_summary(splits, features, target):
    hours = features["Time"] / 3600
    total_hours = hours.max()

    print("\n" + "=" * 60)
    print("8. TIME-BASED TRAIN / TEST SPLITS")
    print("=" * 60)
    print(f"Total time span: {total_hours:.1f} hours")
    print(f"Training starts at: 16h | Test window: 8h")
    print(f"Number of splits: {len(splits)}")
    print(f"{'Split':<8} {'Train Size':>12} {'Train Fraud':>13} {'Test Size':>11} {'Test Fraud':>12} {'Train Hours':>13} {'Test Hours':>12}")
    print("-" * 85)

    for split_number, (train_index, test_index) in enumerate(splits, start=1):
        train_fraud = target.loc[train_index].sum()
        test_fraud = target.loc[test_index].sum()
        train_hours = f"0-{hours.loc[train_index].max():.0f}h"
        test_hours = f"{hours.loc[test_index].min():.0f}-{hours.loc[test_index].max():.0f}h"
        print(f"{split_number:<8} {len(train_index):>12,} {train_fraud:>13,} {len(test_index):>11,} {test_fraud:>12,} {train_hours:>13} {test_hours:>12}")

    figure, axis = plt.subplots(figsize=(14, 4))

    for split_index, (train_index, test_index) in enumerate(splits):
        train_end = hours.loc[train_index].max()
        test_start = hours.loc[test_index].min()
        test_end = hours.loc[test_index].max()
        axis.barh(split_index, train_end, height=0.5, color="#3498db", alpha=0.7, label="Train" if split_index == 0 else "")
        axis.barh(split_index, test_end - test_start, left=test_start, height=0.5, color="#e74c3c", alpha=0.7, label="Test" if split_index == 0 else "")

    axis.set_yticks(range(len(splits)))
    axis.set_yticklabels([f"Split {number}" for number in range(1, len(splits) + 1)])
    axis.set_xlabel("Time (Hours)", fontsize=12)
    axis.set_title("Expanding Window Time-Based Train/Test Splits", fontsize=14, fontweight="bold")
    axis.legend(loc="lower right")
    axis.invert_yaxis()
    figure.tight_layout()
    save_fig(figure, "10_time_based_splits.png")


def demonstrate_smote(splits, features, target, colors, labels):
    print("\n" + "=" * 60)
    print("9. FEATURE SCALING AND SMOTE CHECK")
    print("=" * 60)

    model_features = features.drop(columns=["Time"])
    train_index, test_index = splits[0]
    train_raw = model_features.loc[train_index]
    test_raw = model_features.loc[test_index]
    train_target = target.loc[train_index]
    test_target = target.loc[test_index]
    feature_names = model_features.columns.tolist()
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(scaler.fit_transform(train_raw), columns=feature_names, index=train_raw.index)
    test_scaled = pd.DataFrame(scaler.transform(test_raw), columns=feature_names, index=test_raw.index)

    print(f"Train before SMOTE: {train_scaled.shape} (fraud={train_target.sum()}, legit={len(train_target) - train_target.sum()})")

    resampler = SMOTE(random_state=RANDOM_STATE)
    train_resampled, target_resampled = resampler.fit_resample(train_scaled, train_target)

    print(f"Train after SMOTE:  {train_resampled.shape} (fraud={target_resampled.sum()}, legit={len(target_resampled) - target_resampled.sum()})")
    print(f"Test untouched:      {test_scaled.shape} (fraud={test_target.sum()}, legit={len(test_target) - test_target.sum()})")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    before_counts = train_target.value_counts().sort_index()
    after_counts = pd.Series(target_resampled).value_counts().sort_index()

    axes[0].bar(labels, before_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Training Set Before SMOTE", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")

    axes[1].bar(labels, after_counts.values, color=colors, edgecolor="black")
    axes[1].set_title("Training Set After SMOTE", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")

    for axis, counts in zip(axes, [before_counts, after_counts]):
        for position, count in enumerate(counts.values):
            axis.text(position, count + count * 0.01, f"{count:,}", ha="center", fontsize=10)

    figure.suptitle("SMOTE Resampling Effect on Class Balance", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "11_smote_effect.png")


def print_preprocessing_summary(data, total_hours, split_count):
    print("\n" + "=" * 60)
    print("10. PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Dataset: Kaggle Credit Card Fraud")
    print(f"Transactions: {len(data):,}")
    print(f"Fraudulent: {data['Class'].sum():,} ({data['Class'].mean() * 100:.4f}%)")
    print(f"Legitimate: {(data['Class'] == 0).sum():,} ({(1 - data['Class'].mean()) * 100:.4f}%)")
    print(f"Time span: about {total_hours:.0f} hours")
    print("Features: V1-V28 + Amount")
    print("Scaling: StandardScaler")
    print("Class balancing: SMOTE on training splits only")
    print(f"Validation: expanding-window time split ({split_count} splits)")
    print(f"Results folder: {os.path.abspath(RESULTS_DIR)}")


def run_preprocessing():
    data = load_and_clean()
    colors, labels = analyse_class_distribution(data)
    total_hours = analyse_time_features(data)
    analyse_amounts(data)
    pca_columns = analyse_pca_features(data)
    analyse_correlations(data, pca_columns)
    features, target = prepare_features(data)
    splits = create_time_splits(features)
    print_split_summary(splits, features, target)
    demonstrate_smote(splits, features, target, colors, labels)
    print_preprocessing_summary(data, total_hours, len(splits))

    return data, features, target, splits, total_hours