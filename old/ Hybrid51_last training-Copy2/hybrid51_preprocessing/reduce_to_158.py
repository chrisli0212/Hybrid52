"""
Reduce high-dimensional feature arrays (e.g. 270/327-d) to Hybrid51 158-d by column name matching.

Use when your pipeline produces more features than Hybrid51 expects; columns that match
FEATURE_SCHEMA names are copied into the correct 158-d slots; missing columns are filled with 0.
"""

import numpy as np
import pandas as pd
from typing import List, Union

from .hybrid51_158_schema import FEATURE_DIM, get_expected_158_column_names


def reduce_to_158(
    X: Union[np.ndarray, pd.DataFrame],
    columns: List[str] = None,
    fill_missing: float = 0.0,
) -> np.ndarray:
    """
    Reduce features to 158-d using column name alignment.

    - If X is DataFrame, columns are taken from X.columns (columns arg ignored).
    - If X is ndarray, columns must be provided (length = X.shape[-1]).
    - Expected 158 names come from Hybrid51 FEATURE_SCHEMA. For each expected name,
      if it exists in the provided columns, the value is copied; otherwise fill_missing.

    Returns:
        np.ndarray of shape (..., 158). Same leading shape as X.
    """
    expected = get_expected_158_column_names()
    col_to_idx = {name: i for i, name in enumerate(expected)}

    if isinstance(X, pd.DataFrame):
        columns = list(X.columns)
        X = X.values
    else:
        X = np.asarray(X)
        if columns is None:
            raise ValueError("columns must be provided when X is not a DataFrame")
    if len(columns) != X.shape[-1]:
        raise ValueError(f"columns length {len(columns)} != X last dim {X.shape[-1]}")

    out = np.full(X.shape[:-1] + (FEATURE_DIM,), fill_missing, dtype=X.dtype)
    for i, name in enumerate(columns):
        if name in col_to_idx:
            out[..., col_to_idx[name]] = X[..., i]
    return out


def reduce_df_to_158(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    fill_missing: float = 0.0,
) -> pd.DataFrame:
    """
    Reduce a DataFrame to 158 columns by name matching. Returns a new DataFrame
    with only the 158 Hybrid51 columns (in order); missing columns are added with fill_missing.
    """
    cols = feature_columns if feature_columns is not None else list(df.columns)
    if feature_columns is not None:
        sub = df[cols]
    else:
        sub = df
    arr = reduce_to_158(sub, columns=cols, fill_missing=fill_missing)
    expected = get_expected_158_column_names()
    return pd.DataFrame(arr, index=df.index, columns=expected)
