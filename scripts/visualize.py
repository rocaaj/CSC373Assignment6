"""
visualize.py - Visualization for CelebA Bald vs. Non-Bald Dataset

This script:
- Loads the ARFF dataset.
- Prints a random sample of rows from the dataset.
- Displays and saves a bar chart of the class distribution.
- Helps to quickly inspect data quality and class imbalance.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 14, 2025
"""

import os
import pandas as pd
from scipy.io import arff
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for remote systems
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    """
    Loads an ARFF file into a pandas DataFrame.
    Decodes any byte strings in categorical columns.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    # Decode byte strings in object columns
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].str.decode("utf-8")
    return df

def sample_data(df, num_samples=5):
    """
    Prints a sample of random rows from the dataset.
    """
    sample = df.sample(num_samples, random_state=42)
    print("----- Random Sample of Data -----")
    print(sample)
    print("---------------------------------")

def plot_class_distribution(df, output_dir="../output"):
    """
    Plots and saves a bar chart of the target class distribution.
    Assumes the target column is named "class".
    """
    # Compute class counts
    class_counts = df['class'].value_counts()
    print("Unique classes and their frequencies:")
    print(class_counts)
    
    # Create the bar chart
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar", color=["skyblue", "salmon"])
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in CelebA Bald vs Non-Bald")
    plt.tight_layout()
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Class distribution plot saved to {output_path}\n")

if __name__ == "__main__":
    # Adjust the path as needed; here it's set relative to the current script location.
    data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
    df = load_data(data_path)
    print("Dataset loaded successfully.\n")
    
    # Display a random sample of the dataset.
    sample_data(df, num_samples=5)
    
    # Generate and save the class distribution plot.
    plot_class_distribution(df, output_dir=os.path.join("..", "output"))
    
    print("Visualization completed successfully.")
