import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def generate_artificial_data(n_samples=10, n_features=39):
    """
    Generates a purely random, synthetic DataFrame matching the structure of the real dataset.
    """
    data = np.random.choice([-1, 1], size=(n_samples, n_features))
    columns = [f"att{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

def generate_smart_artificial_data(df, n_samples=10, perturb_fraction=0.1):
    """
    Samples real rows and perturbs a small number of features per row.
    """
    sample = df.sample(n=n_samples, replace=True).drop(columns=["class"]).reset_index(drop=True)
    
    for i in range(n_samples):
        n_perturb = int(perturb_fraction * sample.shape[1])
        cols_to_flip = np.random.choice(sample.columns, size=n_perturb, replace=False)
        for col in cols_to_flip:
            sample.at[i, col] = -1 if sample.at[i, col] == 1 else 1
    
    return sample

def run_prediction(pipeline_path, df, filename_prefix="predictions"):
    # load pipeline
    pipeline = joblib.load(pipeline_path)
    print("Pipeline loaded")

    # predict (-1 = anomaly, 1 = normal → flip to match ground truth convention)
    y_pred = pipeline.predict(df)
    y_pred = np.where(y_pred == -1, 1, -1)

    # attach predictions
    df["prediction"] = y_pred
    print("\nPredictions on Artificial Data:")
    print(df)

    # dynamic filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    output_path = os.path.join("..", "output", filename)

    # save predictions
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    pipeline_path = os.path.join("..", "output", "isolation_forest_pipeline.joblib")
    
    # generate mock data
    artificial_df = generate_artificial_data(n_samples=10)

    # or use smart generation
    smart_artificial_df = generate_smart_artificial_data(artificial_df, n_samples=10)

    run_prediction(pipeline_path, artificial_df, filename_prefix="random_artificial")
    run_prediction(pipeline_path, smart_artificial_df, filename_prefix="smart_artificial")
    print("Predictions made successfully")