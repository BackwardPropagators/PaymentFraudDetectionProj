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

path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")

print("Path to dataset files:", path)

#Loading the Dataset
df = pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"))

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
