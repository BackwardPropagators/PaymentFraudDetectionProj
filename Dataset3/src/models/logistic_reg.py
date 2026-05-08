import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    compute_metrics,
    plot_confusion_matrices,
    plot_feature_coefficients,
    plot_metrics_bars,
    plot_roc_curves,
    print_classification_report,
    print_summary_table,
)
from src.utils import RANDOM_STATE, RESULTS_DIR, scale_and_resample


def build_model():
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def train_and_evaluate(splits, features, target):
    print("\n" + "=" * 60)
    print("11. LOGISTIC REGRESSION")
    print("=" * 60)

    model_features = features.drop(columns=["Time"])
    feature_names = model_features.columns.tolist()
    rows = []
    last_model = None

    for split_number, (train_index, test_index) in enumerate(splits, start=1):
        print(f"\nSplit {split_number}")
        train_features, train_target, test_features, test_target, _ = scale_and_resample(
            model_features,
            target,
            train_index,
            test_index,
        )

        print(f"Train: {len(train_features):,} samples after SMOTE | Test: {len(test_features):,} samples")

        model = build_model()
        model.fit(train_features, train_target)

        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)[:, 1]
        metrics = compute_metrics(test_target, predictions, probabilities)
        metrics["split"] = split_number
        rows.append(metrics)

        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
        print_classification_report(test_target, predictions)

        last_model = model

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "lr_results.csv"), index=False)
    print_summary_table(results, "Logistic Regression")

    return results, last_model, feature_names


def generate_visualisations(splits, features, target, results, last_model, feature_names):
    print("\n" + "=" * 60)
    print("12. LOGISTIC REGRESSION VISUALS")
    print("=" * 60)

    model_features = features.drop(columns=["Time"])

    def predict(train_index, test_index):
        train_features, train_target, test_features, _, _ = scale_and_resample(
            model_features,
            target,
            train_index,
            test_index,
        )
        model = build_model()
        model.fit(train_features, train_target)
        return model.predict(test_features)

    def predict_proba(train_index, test_index):
        train_features, train_target, test_features, _, _ = scale_and_resample(
            model_features,
            target,
            train_index,
            test_index,
        )
        model = build_model()
        model.fit(train_features, train_target)
        return model.predict_proba(test_features)[:, 1]

    plot_confusion_matrices(splits, target, predict, "Logistic Regression", "12_lr_confusion_matrices.png")
    plot_roc_curves(splits, target, predict_proba, "Logistic Regression", "13_lr_roc_curves.png")
    plot_metrics_bars(results, splits, "Logistic Regression", "14_lr_metrics_by_split.png")
    plot_feature_coefficients(
        last_model.coef_[0],
        feature_names,
        "Logistic Regression",
        f"Split {len(splits)}",
        "15_lr_feature_coefficients.png",
    )

    print("\nLogistic Regression complete.")