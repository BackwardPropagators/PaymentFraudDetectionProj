import matplotlib
import numpy as np
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

from src.utils import save_fig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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
    print(f"\n{model_name} Summary Across All Splits")
    print(f"{'Metric':<12} {'Mean':>10} {'Std':>10}")
    print("-" * 34)

    for metric in ["precision", "recall", "f1", "mcc", "auc_roc"]:
        print(f"{metric:<12} {results[metric].mean():>10.4f} {results[metric].std():>10.4f}")


def plot_confusion_matrices(splits, target, predict_function, model_name, filename):
    figure, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for split_number, (train_index, test_index) in enumerate(splits, start=1):
        predictions = predict_function(train_index, test_index)
        test_target = target.loc[test_index]
        matrix = confusion_matrix(test_target, predictions)

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[split_number - 1],
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        axes[split_number - 1].set_title(f"Split {split_number}", fontsize=12, fontweight="bold")
        axes[split_number - 1].set_ylabel("Actual")
        axes[split_number - 1].set_xlabel("Predicted")

    figure.suptitle(f"{model_name} - Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    figure.tight_layout()
    save_fig(figure, filename)


def plot_roc_curves(splits, target, predict_proba_function, model_name, filename):
    figure, axis = plt.subplots(figsize=(8, 6))

    for split_number, (train_index, test_index) in enumerate(splits, start=1):
        probabilities = predict_proba_function(train_index, test_index)
        test_target = target.loc[test_index]
        false_positive_rate, true_positive_rate, _ = roc_curve(test_target, probabilities)
        auc_value = roc_auc_score(test_target, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label=f"Split {split_number} (AUC={auc_value:.4f})")

    axis.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    axis.set_xlabel("False Positive Rate", fontsize=12)
    axis.set_ylabel("True Positive Rate", fontsize=12)
    axis.set_title(f"{model_name} - ROC Curves", fontsize=14, fontweight="bold")
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_fig(figure, filename)


def plot_metrics_bars(results, splits, model_name, filename):
    figure, axis = plt.subplots(figsize=(10, 5))
    metrics = ["precision", "recall", "f1", "mcc", "auc_roc"]
    positions = np.arange(len(splits))
    width = 0.15
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for metric_index, metric in enumerate(metrics):
        axis.bar(
            positions + metric_index * width,
            results[metric].values,
            width,
            label=metric.upper().replace("_", "-"),
            color=colors[metric_index],
        )

    axis.set_xlabel("Split", fontsize=12)
    axis.set_ylabel("Score", fontsize=12)
    axis.set_title(f"{model_name} - Metrics by Split", fontsize=14, fontweight="bold")
    axis.set_xticks(positions + width * 2)
    axis.set_xticklabels([f"Split {number}" for number in range(1, len(splits) + 1)])
    axis.legend(loc="lower right")
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    save_fig(figure, filename)


def plot_feature_coefficients(coefficients, feature_names, model_name, split_label, filename, top_count=15):
    figure, axis = plt.subplots(figsize=(12, 6))
    sorted_index = np.argsort(np.abs(coefficients))[::-1]
    selected_index = sorted_index[:top_count]
    selected_coefficients = coefficients[selected_index]

    axis.barh(
        range(top_count),
        selected_coefficients,
        color=["#e74c3c" if value < 0 else "#2ecc71" for value in selected_coefficients],
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_yticks(range(top_count))
    axis.set_yticklabels([feature_names[index] for index in selected_index])
    axis.set_xlabel("Coefficient Value", fontsize=12)
    axis.set_title(f"{model_name} - Top {top_count} Feature Coefficients ({split_label})", fontsize=14, fontweight="bold")
    axis.invert_yaxis()
    figure.tight_layout()
    save_fig(figure, filename)