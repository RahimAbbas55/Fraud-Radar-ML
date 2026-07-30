import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COL

def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw credit card transaction dataset.
 
    Raises a clear error if the file is missing, rather than letting
    pandas raise a generic FileNotFoundError with no context.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from https://www.kaggle.com/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in the data/ directory. "
            "See README.md 'Setup' section for details."
        )
 
    df = pd.read_csv(path)
    return df
 
 
def validate_schema(df: pd.DataFrame) -> None:
    """
    Sanity-check the loaded dataframe against known-good expectations
    for this dataset. Fails loudly and early if something is off,
    rather than letting a malformed dataset silently corrupt downstream
    model training.
    """
    expected_cols = {"Time", "Amount", TARGET_COL} | {f"V{i}" for i in range(1, 29)}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
 
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values — expected none for this source.")
 
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected values in target column '{TARGET_COL}'.")
 
"""Return normalized class distribution (fraud vs. non-fraud)."""
def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].value_counts(normalize=True)
 
"""Convenience wrapper: load, validate, and return the dataframe."""
def load_and_validate(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    validate_schema(df)
    return df
 