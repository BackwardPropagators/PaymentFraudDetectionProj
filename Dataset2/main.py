import os

import kagglehub
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42
N_FOLDS = 10
RESULTS_DIR = "results"

HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.3
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(figure, name):
    path = os.path.join(RESULTS_DIR, name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {path}")


def load_data():
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    data = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nelgiriyewithana/credit-card-fraud-detection-dataset-2023",
        "creditcard_2023.csv",
    )

    if "id" in data.columns:
        data = data.drop(columns=["id"])
        print("Dropped id column")

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
    print(f"Ratio: 1 : {class_counts[0] / class_counts[1]:.2f}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Class Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")

    for position, (count, percentage) in enumerate(zip(class_counts.values, class_percentages.values)):
        axes[0].text(position, count + count * 0.01, f"{count:,}\n({percentage:.3f}%)", ha="center", fontsize=10)

    axes[1].pie(class_counts.values, labels=labels, colors=colors, autopct="%1.3f%%", startangle=90)
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    figure.suptitle("Class Distribution in Dataset 2", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "01_class_distribution.png")

    return colors, labels


def analyse_amounts(data):
    print("\n" + "=" * 60)
    print("3. TRANSACTION AMOUNT ANALYSIS")
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
    save_fig(figure, "02_amount_distribution.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(np.log1p(legitimate_amounts), bins=80, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
    axis.hist(np.log1p(fraud_amounts), bins=80, alpha=0.6, color="#e74c3c", label="Fraudulent", density=True)
    axis.set_title("Log-Transformed Amount Distribution", fontsize=14, fontweight="bold")
    axis.set_xlabel("log(1 + Amount)")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    save_fig(figure, "03_amount_log_distribution.png")


def analyse_pca_features(data):
    print("\n" + "=" * 60)
    print("4. PCA FEATURE ANALYSIS")
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
    save_fig(figure, "04_top_pca_features.png")

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
    save_fig(figure, "05_all_v_features_boxplots.png")

    return pca_columns


def analyse_correlations(data, pca_columns):
    print("\n" + "=" * 60)
    print("5. CORRELATION ANALYSIS")
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
    save_fig(figure, "06_feature_correlation_with_class.png")

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
    save_fig(figure, "07_correlation_heatmap.png")


def prepare_features(data):
    print("\n" + "=" * 60)
    print("6. PREPROCESSING")
    print("=" * 60)

    features = data.drop(columns=["Class"])
    target = data["Class"]

    print(f"Feature matrix: {features.shape}")
    print(f"Target vector: {target.shape} (fraud={target.sum():,}, legit={len(target) - target.sum():,})")

    return features, target


def create_splits(features, target):
    print("\n" + "=" * 60)
    print("7. STRATIFIED K-FOLD SETUP")
    print("=" * 60)

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(features, target))

    print(f"Folds: {N_FOLDS}")
    print(f"{'Fold':<6} {'Train Size':>12} {'Train Fraud':>13} {'Test Size':>11} {'Test Fraud':>12}")
    print("-" * 60)

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        train_fraud = target.iloc[train_index].sum()
        test_fraud = target.iloc[test_index].sum()
        print(f"{fold_number:<6} {len(train_index):>12,} {train_fraud:>13,} {len(test_index):>11,} {test_fraud:>12,}")

    return splits


def scale_split(features, target, train_index, test_index):
    feature_names = features.columns.tolist()
    train_raw = features.iloc[train_index]
    test_raw = features.iloc[test_index]
    train_target = target.iloc[train_index]
    test_target = target.iloc[test_index]
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(scaler.fit_transform(train_raw), columns=feature_names, index=train_raw.index)
    test_scaled = pd.DataFrame(scaler.transform(test_raw), columns=feature_names, index=test_raw.index)

    return train_scaled, train_target, test_scaled, test_target, scaler


def demonstrate_scaling(splits, features, target, colors, labels):
    print("\n" + "=" * 60)
    print("8. SCALING CHECK")
    print("=" * 60)

    train_index, test_index = splits[0]
    train_scaled, train_target, test_scaled, test_target, _ = scale_split(features, target, train_index, test_index)

    print(f"Train shape: {train_scaled.shape} (fraud={train_target.sum():,}, legit={len(train_target) - train_target.sum():,})")
    print(f"Test shape:  {test_scaled.shape} (fraud={test_target.sum():,}, legit={len(test_target) - test_target.sum():,})")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    train_counts = train_target.value_counts().sort_index()
    test_counts = test_target.value_counts().sort_index()

    axes[0].bar(labels, train_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Training Set (Fold 1)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")

    axes[1].bar(labels, test_counts.values, color=colors, edgecolor="black")
    axes[1].set_title("Test Set (Fold 1)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")

    for axis, counts in zip(axes, [train_counts, test_counts]):
        for position, count in enumerate(counts.values):
            axis.text(position, count + count * 0.01, f"{count:,}", ha="center", fontsize=10)

    figure.suptitle("Class Balance Check (Fold 1)", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, "08_class_balance_verification.png")


def print_preprocessing_summary(data, split_count):
    print("\n" + "=" * 60)
    print("9. PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Dataset: Kaggle Credit Card Fraud 2023")
    print(f"Transactions: {len(data):,}")
    print(f"Fraudulent: {data['Class'].sum():,} ({data['Class'].mean() * 100:.4f}%)")
    print(f"Legitimate: {(data['Class'] == 0).sum():,} ({(1 - data['Class'].mean()) * 100:.4f}%)")
    print(f"Features: V1-V28 + Amount")
    print(f"Scaling: StandardScaler")
    print(f"Splits: {split_count}-fold stratified cross-validation")
    print(f"Results folder: {os.path.abspath(RESULTS_DIR)}")


def compute_metrics(true_target, predictions, probabilities):
    return {
        "precision": precision_score(true_target, predictions, zero_division=0),
        "recall": recall_score(true_target, predictions, zero_division=0),
        "f1": f1_score(true_target, predictions, zero_division=0),
        "mcc": matthews_corrcoef(true_target, predictions),
        "auc_roc": roc_auc_score(true_target, probabilities),
    }


def print_classification_report(true_target, predictions):
    print(
        classification_report(
            true_target,
            predictions,
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
        )
    )


def print_summary_table(results, model_name):
    print(f"\n{model_name} Summary Across All Folds")
    print(f"{'Metric':<12} {'Mean':>10} {'Std':>10}")
    print("-" * 34)

    for metric in ["precision", "recall", "f1", "mcc", "auc_roc"]:
        print(f"{metric:<12} {results[metric].mean():>10.4f} {results[metric].std():>10.4f}")


def plot_confusion_matrices(splits, target, predictions_by_fold, model_name, filename):
    shown_folds = min(5, len(splits))
    figure, axes = plt.subplots(1, shown_folds, figsize=(5 * shown_folds, 4))
    if shown_folds == 1:
        axes = [axes]

    for fold_offset in range(shown_folds):
        test_index = splits[fold_offset][1]
        test_target = target.iloc[test_index]
        predictions = predictions_by_fold[fold_offset + 1]["predictions"]
        matrix = confusion_matrix(test_target, predictions)

        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[fold_offset], xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[fold_offset].set_title(f"Fold {fold_offset + 1}", fontsize=12, fontweight="bold")
        axes[fold_offset].set_ylabel("Actual")
        axes[fold_offset].set_xlabel("Predicted")

    figure.suptitle(f"{model_name} - Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, filename)


def plot_roc_curves(splits, target, predictions_by_fold, model_name, filename):
    figure, axis = plt.subplots(figsize=(8, 6))

    for fold_number, split in enumerate(splits, start=1):
        test_index = split[1]
        test_target = target.iloc[test_index]
        probabilities = predictions_by_fold[fold_number]["probabilities"]
        false_positive_rate, true_positive_rate, _ = roc_curve(test_target, probabilities)
        auc_value = roc_auc_score(test_target, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label=f"Fold {fold_number} (AUC={auc_value:.4f})", alpha=0.7)

    axis.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    axis.set_xlabel("False Positive Rate", fontsize=12)
    axis.set_ylabel("True Positive Rate", fontsize=12)
    axis.set_title(f"{model_name} - ROC Curves", fontsize=14, fontweight="bold")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    save_fig(figure, filename)


def plot_metrics_bars(results, splits, model_name, filename):
    shown_folds = min(5, len(splits))
    figure, axis = plt.subplots(figsize=(12, 5))
    metrics = ["precision", "recall", "f1", "mcc", "auc_roc"]
    positions = np.arange(shown_folds)
    width = 0.15
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for metric_offset, metric in enumerate(metrics):
        axis.bar(positions + metric_offset * width, results[metric].values[:shown_folds], width, label=metric.upper().replace("_", "-"), color=colors[metric_offset])

    axis.set_xlabel("Fold", fontsize=12)
    axis.set_ylabel("Score", fontsize=12)
    axis.set_title(f"{model_name} - Metrics by Fold", fontsize=14, fontweight="bold")
    axis.set_xticks(positions + width * 2)
    axis.set_xticklabels([f"Fold {fold_number}" for fold_number in range(1, shown_folds + 1)])
    axis.legend(loc="lower right")
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    save_fig(figure, filename)


class FraudDetectorNN(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


def set_seeds():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def train_nn(train_features, train_target, input_dim):
    set_seeds()

    model = FraudDetectorNN(input_dim).to(DEVICE)
    loss_function = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    feature_values = train_features.values if hasattr(train_features, "values") else np.array(train_features)
    target_values = train_target.values if hasattr(train_target, "values") else np.array(train_target)
    dataset = TensorDataset(torch.FloatTensor(feature_values), torch.FloatTensor(target_values))
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        sample_count = 0

        for batch_features, batch_target in loader:
            batch_features = batch_features.to(DEVICE)
            batch_target = batch_target.to(DEVICE)
            optimiser.zero_grad()
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_target)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch_features)
            sample_count += len(batch_features)

        average_loss = total_loss / sample_count
        losses.append(average_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1:>3}/{EPOCHS} | Loss: {average_loss:.6f}")

    return model, losses


def predict_nn(model, test_features):
    model.eval()
    with torch.no_grad():
        feature_values = test_features.values if hasattr(test_features, "values") else np.array(test_features)
        probabilities = model(torch.FloatTensor(feature_values).to(DEVICE)).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


def train_and_evaluate_nn(splits, features, target):
    print("\n" + "=" * 60)
    print("10. FEED-FORWARD NEURAL NETWORK")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    input_dim = features.shape[1]
    results = []
    predictions_by_fold = {}
    losses_by_fold = {}

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        print(f"\nFold {fold_number}/{N_FOLDS}")
        train_features, train_target, test_features, test_target, _ = scale_split(features, target, train_index, test_index)
        print(f"Train: {len(train_features):,} samples | Test: {len(test_features):,} samples")

        model, losses = train_nn(train_features, train_target, input_dim)
        predictions, probabilities = predict_nn(model, test_features)
        metrics = compute_metrics(test_target, predictions, probabilities)
        metrics["fold"] = fold_number

        results.append(metrics)
        predictions_by_fold[fold_number] = {"predictions": predictions, "probabilities": probabilities}
        losses_by_fold[fold_number] = losses

        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
        print_classification_report(test_target, predictions)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(os.path.join(RESULTS_DIR, "nn_results.csv"), index=False)
    print_summary_table(results_frame, "Feed-Forward Neural Network")

    return results_frame, predictions_by_fold, losses_by_fold


def generate_nn_visualisations(splits, target, results, predictions_by_fold, losses_by_fold):
    print("\n" + "=" * 60)
    print("11. FEED-FORWARD NEURAL NETWORK VISUALS")
    print("=" * 60)

    figure, axis = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for fold_number, losses in losses_by_fold.items():
        axis.plot(range(1, len(losses) + 1), losses, label=f"Fold {fold_number}", color=colors[(fold_number - 1) % len(colors)], linewidth=1.5, alpha=0.7)

    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    axis.set_title("Feed-Forward Neural Network - Training Loss", fontsize=14, fontweight="bold")
    axis.legend(loc="upper right", fontsize=8, ncol=2)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    save_fig(figure, "09_nn_training_loss.png")

    plot_confusion_matrices(splits, target, predictions_by_fold, "Feed-Forward Neural Network", "10_nn_confusion_matrices.png")
    plot_roc_curves(splits, target, predictions_by_fold, "Feed-Forward Neural Network", "11_nn_roc_curves.png")
    plot_metrics_bars(results, splits, "Feed-Forward Neural Network", "12_nn_metrics_by_fold.png")


def main():
    data = load_data()
    colors, labels = analyse_class_distribution(data)
    analyse_amounts(data)
    pca_columns = analyse_pca_features(data)
    analyse_correlations(data, pca_columns)
    features, target = prepare_features(data)
    splits = create_splits(features, target)
    demonstrate_scaling(splits, features, target, colors, labels)
    print_preprocessing_summary(data, len(splits))
    results, predictions_by_fold, losses_by_fold = train_and_evaluate_nn(splits, features, target)
    generate_nn_visualisations(splits, target, results, predictions_by_fold, losses_by_fold)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()