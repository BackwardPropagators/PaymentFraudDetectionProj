import os

import kagglehub
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

matplotlib.use("Agg")


RANDOM_STATE = 42
SAMPLE_SIZE = 100_000


def load_data():
    dataset_path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")
    data_path = os.path.join(dataset_path, "PS_20174392719_1491204439457_log.csv")

    print(f"Dataset path: {dataset_path}")
    data = pd.read_csv(data_path)
    sample_size = min(SAMPLE_SIZE, len(data))
    return data.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)


def clean_data(data):
    missing = data.isnull().sum()
    duplicate_count = data.duplicated().sum()

    print(f"Missing values: {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])

    print(f"Duplicate rows: {duplicate_count:,}")
    if duplicate_count > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"New shape: {data.shape[0]:,} rows")

    return data.drop(columns=["nameOrig", "nameDest"], errors="ignore")


def encode_payment_type(data):
    if "type" not in data.columns:
        data["type_encoded"] = 0
        return data

    encoder = LabelEncoder()
    data = data.copy()
    data["type_encoded"] = encoder.fit_transform(data["type"])

    print("Payment type encoding:")
    for code, name in enumerate(encoder.classes_):
        print(f"  {name}: {code}")

    return data.drop(columns=["type"])


def add_features(data):
    amount_mean = data["amount"].mean()
    amount_std = data["amount"].std()
    average_amount = amount_mean + 1

    data = data.assign(
        balance_change_orig=data["newbalanceOrig"] - data["oldbalanceOrg"],
        balance_change_dest=data["newbalanceDest"] - data["oldbalanceDest"],
        amount_to_balance_ratio=data["amount"] / (data["oldbalanceOrg"] + 1),
        amount_to_dest_ratio=data["amount"] / (data["oldbalanceDest"] + 1),
    )

    data = data.assign(
        abs_balance_change_orig=np.abs(data["balance_change_orig"]),
        abs_balance_change_dest=np.abs(data["balance_change_dest"]),
        is_large_transaction=(data["amount"] > amount_mean + 2 * amount_std).astype(int),
        balance_mismatch_orig=(
            np.abs(data["newbalanceOrig"] - (data["oldbalanceOrg"] - data["amount"])) > 0.01
        ).astype(int),
        is_zero_balance_after=(data["newbalanceOrig"] == 0).astype(int),
        type_amount_interaction=data["type_encoded"] * np.log1p(data["amount"]),
        amount_change_interaction=data["amount"] * data["balance_change_orig"],
        log_amount=np.log1p(data["amount"]),
        log_oldbalance_orig=np.log1p(data["oldbalanceOrg"]),
        log_oldbalance_dest=np.log1p(data["oldbalanceDest"]),
        amount_to_avg_ratio=data["amount"] / average_amount,
    )

    return data


def split_data(data):
    features = data.drop(columns=["isFraud"])
    target = data["isFraud"].astype(int)

    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print("Target distribution:")
    print(f"  Non-fraud: {(target == 0).sum():,} ({(target == 0).mean() * 100:.4f}%)")
    print(f"  Fraud:     {(target == 1).sum():,} ({(target == 1).mean() * 100:.4f}%)")

    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def scale_numeric_columns(train_features, test_features):
    numeric_columns = train_features.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()

    scaled_train = train_features.copy()
    scaled_test = test_features.copy()
    scaled_train[numeric_columns] = scaler.fit_transform(train_features[numeric_columns])
    scaled_test[numeric_columns] = scaler.transform(test_features[numeric_columns])

    return scaled_train, scaled_test


def build_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def show_feature_importance(model, feature_names):
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("Top 10 most important features:")
    print(importance.head(10).to_string(index=False))


def show_metrics(test_target, predictions, probabilities):
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions, zero_division=0)
    recall = recall_score(test_target, predictions, zero_division=0)
    f1 = f1_score(test_target, predictions, zero_division=0)
    auc_roc = roc_auc_score(test_target, probabilities)

    print("Performance metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc_roc:.4f}")


def main():
    data = load_data()
    data = clean_data(data)
    data = encode_payment_type(data)
    data = add_features(data)

    train_features, test_features, train_target, test_target = split_data(data)
    train_features, test_features = scale_numeric_columns(train_features, test_features)

    print(f"Training set: {train_features.shape[0]:,} samples")
    print(f"Test set: {test_features.shape[0]:,} samples")

    model = build_model()
    model.fit(train_features, train_target)

    show_feature_importance(model, train_features.columns)

    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    show_metrics(test_target, predictions, probabilities)


if __name__ == "__main__":
    main()