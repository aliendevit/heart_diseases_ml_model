# src/preprocess.py
# place that:

# Loads heart.csv

# Creates age_group

# One-hot encodes age_group

# Returns X, y, and list of feature columns
import pandas as pd
from pathlib import Path

DEFAULT_TARGET_COL = "target"  # change if your label column has another name

def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw heart dataset from a CSV file.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical 'age_group' column based on 'age'.
    """
    bins = [0, 40, 50, 60, 100]
    labels = ["<=40", "41-50", "51-60", ">60"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def encode_features(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL):
    """
    Apply feature engineering and return:
    - X: feature matrix (DataFrame)
    - y: target Series
    - feature_cols: list of feature column names

    Steps:
    1) Add age_group
    2) One-hot encode age_group with pd.get_dummies
    3) Split into X, y
    """
    df = add_age_group(df)

    # One-hot encode age_group
    df_encoded = pd.get_dummies(df, columns=["age_group"], drop_first=True)

    if target_col not in df_encoded.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns.")

    feature_cols = [col for col in df_encoded.columns if col != target_col]

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    return X, y, feature_cols


def load_and_preprocess(csv_path: str | Path, target_col: str = DEFAULT_TARGET_COL):
    """
    Convenience function:
    1) Load raw data
    2) Encode features and split into X, y
    """
    df = load_raw_data(csv_path)
    X, y, feature_cols = encode_features(df, target_col=target_col)
    return df, X, y, feature_cols
