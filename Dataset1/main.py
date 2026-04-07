import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score, recall_score, precision_score)

path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")

print("Path to dataset files:", path)

#Loading the Dataset
df = pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"))

SAMPLE_SIZE = 100000  # Start with 10,000 rows (adjust as needed: 5000, 10000, 20000)
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

df.head()

#Cleaning dataset
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

# There are no missing rows or duplicates, therefore null/missing/duplicate cleaning not needed

#Cleaning dataset

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

# There are no missing rows or duplicates, therefore null/missing/duplicate cleaning not needed

# drop identifiers(nameOrig, nameDest)
df = df.drop(columns=['nameOrig', 'nameDest'])

df.head()

# encoding "type"
if 'type' in df.columns:
    print(f"Original 'type' column values: {df['type'].unique()}")
    print(f"Value counts:\n{df['type'].value_counts()}")
    
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])
    print(f"\nLabel encoding applied:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"  {category} → {i}")

    df = df.drop(columns=['type'])
    print(f"\nDropped original 'type' column")
else:
    print("No 'type' column found for encoding")

df.head()


#Feture Engineering
df = df.assign(
    balance_change_orig=df['newbalanceOrig'] - df['oldbalanceOrg']
)

df = df.assign(
    balance_change_dest=df['newbalanceDest'] - df['oldbalanceDest']
)

df = df.assign(
    amount_to_balance_ratio=df['amount'] / (df['oldbalanceOrg'] + 1)
)

df = df.assign(
    amount_to_dest_ratio=df['amount'] / (df['oldbalanceDest'] + 1)
)

df = df.assign(
    abs_balance_change_orig=np.abs(df['balance_change_orig']),
    abs_balance_change_dest=np.abs(df.get('balance_change_dest', 0))
)

amount_mean = df['amount'].mean()
amount_std = df['amount'].std()
df = df.assign(
    is_large_transaction=(df['amount'] > amount_mean + 2 * amount_std).astype(int)
)

df = df.assign(
    balance_mismatch_orig=(
        np.abs(df['newbalanceOrig'] - 
               (df['oldbalanceOrg'] - df['amount'])) > 0.01
    ).astype(int)
)

df = df.assign(
    is_zero_balance_after=(df['newbalanceOrig'] == 0).astype(int)
)

df = df.assign(
    type_amount_interaction=df['type_encoded'] * np.log1p(df['amount'])
)

df = df.assign(
    amount_change_interaction=df['amount'] * df['balance_change_orig']
)

df = df.assign(
    log_amount=np.log1p(df['amount']),
    log_oldbalance_orig=np.log1p(df['oldbalanceOrg']),
    log_oldbalance_dest=np.log1p(df.get('oldbalanceDest', 0))
)

avg_transaction_amount = df['amount'].mean()
df = df.assign(
    amount_to_avg_ratio=df['amount'] / (avg_transaction_amount + 1)
)

df.head()

# Features and Targets
X = df.drop(columns=['isFraud'])
Y = df['isFraud']

Y = Y.astype(int)

print(f"Features shape: {X.shape}")
print(f"Target shape: {Y.shape}")
print(f"\nTarget distribution:")
print(f"  Non-Fraud (0): {(Y == 0).sum():,} ({(Y == 0).mean() * 100:.4f}%)")
print(f"  Fraud (1):     {(Y == 1).sum():,} ({(Y == 1).mean() * 100:.4f}%)")


# Data Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42,
    stratify=Y  # Preserves the fraud ratio in both sets
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"\nTest set: {X_test.shape[0]:,} samples")


# Handeling scaler
scaler = RobustScaler() #StandardScaler()

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

#training random forest model
random_forest_params = {
    'n_estimators': 100,  # T = number of trees
    'max_depth': None,      # Maximum depth of each tree
    'min_samples_split': 2,   # Minimum samples required to split a node
    'min_samples_leaf': 1,    # Minimum samples required at a leaf node
    'max_features': 'sqrt',   # Number of features to consider for best split
    'random_state': 42,
    'n_jobs': -1,         # Use all CPU cores
    # 'class_weight': 'balanced'  # Handle imbalance automatically
}

print("Baseline Random Forest Parameters:")
print(f"  n_estimators: {random_forest_params['n_estimators']}")
print(f"  max_depth: {random_forest_params['max_depth']}")
print(f"  min_samples_split: {random_forest_params['min_samples_split']}")
print(f"  min_samples_leaf: {random_forest_params['min_samples_leaf']}")
print(f"  class_weight: None (no balancing)")
print(f"  random_state: {random_forest_params['random_state']}")

rf_model = RandomForestClassifier(**random_forest_params)
rf_model.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

#Testing Model 
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nBaseline Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
