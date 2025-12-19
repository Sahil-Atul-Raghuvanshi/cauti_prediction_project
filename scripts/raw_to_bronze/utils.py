import pandas as pd
import os

def drop_columns(df, columns):
    """
    Drop specified columns from a DataFrame safely.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        List of column names to drop

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns removed
    """
    return df.drop(columns=columns, errors="ignore")
