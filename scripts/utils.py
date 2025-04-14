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
    if drop_duplicates:
        df = df.drop_duplicates()

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    return X, y
