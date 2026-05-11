import os

import matplotlib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "DataSet3.csv")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(figure, name):
    path = os.path.join(RESULTS_DIR, name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {path}")


def create_time_splits(features, test_window=8, min_train_hours=16):
    hours = features["Time"] / 3600
    total_hours = hours.max()
    splits = []
    test_start = min_train_hours

    while test_start < total_hours:
        test_end = min(test_start + test_window, total_hours + 1)
        train_mask = hours < test_start
        test_mask = (hours >= test_start) & (hours < test_end)
        train_index = features[train_mask].index.tolist()
        test_index = features[test_mask].index.tolist()

        if train_index and test_index:
            splits.append((train_index, test_index))

        test_start += test_window

    return splits


def scale_and_resample(model_features, target, train_index, test_index):
    feature_names = model_features.columns.tolist()
    train_raw = model_features.loc[train_index]
    test_raw = model_features.loc[test_index]
    train_target = target.loc[train_index]
    test_target = target.loc[test_index]
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_raw),
        columns=feature_names,
        index=train_raw.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_raw),
        columns=feature_names,
        index=test_raw.index,
    )

    resampler = SMOTE(random_state=RANDOM_STATE)
    train_resampled, target_resampled = resampler.fit_resample(train_scaled, train_target)

    return train_resampled, target_resampled, test_scaled, test_target, scaler