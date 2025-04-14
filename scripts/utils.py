"""
utils.py - Utility Functions for Data Mining Assignment

This module provides helper functions to facilitate data loading and preparation for analysis.
It includes functions to load ARFF data into a pandas DataFrame (with proper decoding of 
byte strings) and to split the data into features and target labels while optionally dropping 
duplicate rows.

Functions:
    load_arff_data(filepath): Loads an ARFF file and returns a DataFrame with decoded string columns.
    prepare_data(df, target_column="class", drop_duplicates=True): Splits the DataFrame into 
        features (X) and target labels (y), converting the target to integers and optionally 
        removing duplicate rows.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 14, 2025
"""
import pandas as pd
import numpy as np
from scipy.io import arff

# load data
def load_arff_data(filepath):
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")
    return df

# split features and labels
def prepare_data(df, target_column="class", drop_duplicates=True):
    # drop duplicates based only on feature columns
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=df.columns.difference([target_column]))
        after = len(df)
        print(f"🧹 Dropped {before - after} duplicates (excluding target column).")

    # convert features to integers
    X = df.drop(columns=[target_column]).astype(int)

    # convert target labels to binary: -1 → 0, 1 → 1
    y = df[target_column].astype(int)
    y = np.where(y == -1, 0, 1)

    # log class distribution
    print(f"Cleaned data shape: {df.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y
