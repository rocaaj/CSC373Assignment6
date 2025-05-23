"""
understand_data.py - Data Understanding and Reporting for CelebA Bald vs. Non-Bald Dataset

This script loads the dataset from an ARFF file, generates a comprehensive report of the dataset,
and saves the report to a text file. The report includes information about the shape of the data, 
column names, class distribution, duplicate counts, missing values, data types, and summary statistics.

Functions:
    load_arff(filepath): Loads an ARFF file into a pandas DataFrame and decodes byte strings.
    gen_report(df): Generates a detailed textual report summarizing the dataset.
    save_report(report, filepath): Saves the generated report to a specified .txt file.

Usage:
    When run as a script, it loads the dataset from the data directory, prints a sample report to the console,
    and saves the full report to the output directory.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 14, 2025
"""

import pandas as pd
from scipy.io import arff
import os

def load_arff(filepath):
    """
    Loads an ARFF file into a pandas DataFrame.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # decode byte strings if cat columns are in bytes
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")
    
    return df

def gen_report(df):
    """
    Generates a report of the dataset.
    """
    target_col = df.columns[-1]

    report = "Data description:\n" + f"Shape: {df.shape}\n"
    report += "Columns:\n" + ", ".join(df.columns) + "\n"
    report += f"Class Distribution ({target_col}):\n" + df[target_col].value_counts().to_string() + "\n"
    report += f"Duplicated rows: {df.duplicated().sum()}\n" + f"Number of rows with missing values: {df.isnull().any(axis=1).sum()}\n"
    report += f"Number of columns with missing values: {df.isnull().any(axis=0).sum()}\n\n" + f"{df.dtypes.to_string()}\n\n" + f"{df.describe().T.to_string()}\n"
    return report

def save_report(report, filepath):
    """
    Saves the report to a .txt file.
    """
    with open(filepath, "w") as file:
        file.write(report)

if __name__ == "__main__":
    arff_path = os.path.join("../data", "celeba_baldvsnonbald.arff")
    df = load_arff(arff_path)
    print("Dataset Loaded Successfully")

    # gen report
    report = gen_report(df)
    # console sanity check
    print(report)

    # save report
    report_path = os.path.join("../output", "data_report.txt")
    save_report(report, report_path)
    print(f"Report saved to {report_path}")
    print("Script ran successfully")
