"""
Utility helpers: SMOTE resampling, time-based splitting, figure saving, and shared constants.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── Shared constants ──
RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "DataSet3.csv")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save a matplotlib figure to the results directory."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def create_time_splits(X, test_window=8, min_train_hours=16):
    """
    Create expanding-window time-based train/test splits.

    The dataset covers ~48 hours (0–172,792 seconds).
    We use an expanding-window approach:
       - Minimum training window: first 16 hours
       - Test window: 8 hours each
       - Train always precedes test (no temporal leakage)

    Returns a list of (train_idx, test_idx) tuples.
    """
    hours = X["Time"] / 3600
    total_hours = hours.max()

    splits = []
    test_start = min_train_hours

    while test_start < total_hours:
        test_end = min(test_start + test_window, total_hours + 1)
        train_mask = hours < test_start
        test_mask = (hours >= test_start) & (hours < test_end)

        train_idx = X[train_mask].index.tolist()
        test_idx = X[test_mask].index.tolist()

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        test_start += test_window

    return splits


def scale_and_resample(X_features, y, train_idx, test_idx):
    """
    Scale features (fit on train only) and apply SMOTE to the training set.

    Why after scaling? SMOTE relies on Euclidean distance to find nearest neighbours.
    If features aren't scaled, the distance calculation is dominated by whichever feature
    has the largest magnitude — the synthetic points would cluster along the Amount axis
    and ignore the V-features.

    Returns: X_train_res, y_train_res, X_test_scaled, y_test, scaler
    """
    feature_names = X_features.columns.tolist()

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

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, y_train_res, X_test_scaled, y_test, scaler
