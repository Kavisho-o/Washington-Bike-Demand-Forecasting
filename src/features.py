import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding for month and weekday.
    """
    # If cyclical columns already present, don't overwrite
    cyclical_cols = {"month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"}
    if cyclical_cols.issubset(set(df.columns)):
        return df

    # Try to compute from explicit `month`/`weekday` columns if available
    if "month" in df.columns and "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "weekday" in df.columns and "dayofweek_sin" not in df.columns:
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    return df


def prepare_features(df: pd.DataFrame):
    """
    Separate features and target.
    """
    X = df.drop("cnt", axis=1)
    y = df["cnt"]
    return X, y