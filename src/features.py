import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TARGET_COL, TEST_SIZE, STRATIFY_SPLIT

"""
    Derive an hour-of-day feature from `Time` (seconds elapsed since
    the first transaction in the dataset). This is the one piece of
    interpretable feature engineering available, since V1-V28 are
    anonymized PCA components.
    """
def add_time_of_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    return df

"""Split a dataframe into X (features) and y (target)."""
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

"""
    Stratified train/test split on the target column.

    Stratification matters here specifically because fraud is ~0.17%
    of the data — a non-stratified split risks producing a test set
    with too few (or even zero) fraud examples, which would make
    evaluation metrics meaningless.
    """
def train_test_split_stratified(df: pd.DataFrame, test_size=TEST_SIZE):
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY_SPLIT else None,
    )