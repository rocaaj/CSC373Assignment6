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