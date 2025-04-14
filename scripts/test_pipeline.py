import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def generate_artificial_data(n_samples=10, n_features=39):
    """
    Generates a purely random, synthetic DataFrame matching the structure of the real dataset.
    """
    data = np.random.choice([0, 1], size=(n_samples, n_features))
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
            sample.at[i, col] = 0 if sample.at[i, col] == 1 else 1
    return sample

def evaluate_predictions(df, output_path):
    y_true = df.get("class")
    y_pred = df["prediction"]

    if y_true is not None:
        print("\nEvaluation Metrics:")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC Score: {auc:.4f}")

        # save classification metrics and plot
        report = classification_report(y_true, y_pred, output_dict=True)
        f1 = report["1"]["f1-score"] if "1" in report else 0

        # plot prediction counts
        plt.figure(figsize=(6, 4))
        df["prediction"].value_counts().sort_index().plot(kind="bar", color="skyblue")
        plt.title("Prediction Distribution")
        plt.xlabel("Predicted Class")
        plt.ylabel("Frequency")
        plt.xticks([0, 1], ["Non-Bald (0)", "Bald (1)"])
        plot_path = output_path.replace(".csv", "_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Prediction distribution plot saved to {plot_path}")

        return auc, f1
    else:
        print("No ground truth available for evaluation.")
        return None, None

def run_prediction(pipeline_path, df, filename_prefix="predictions"):
    pipeline = joblib.load(pipeline_path)
    print("Pipeline loaded")

    y_pred = pipeline.predict(df)
    y_pred = np.where(y_pred == -1, 1, 0)
    df["prediction"] = y_pred

    print("\nPredictions on Artificial Data:")
    print(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    output_path = os.path.join("..", "output", filename)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    evaluate_predictions(df, output_path)

if __name__ == "__main__":
    pipeline_path = os.path.join("..", "output", "isolation_forest_pipeline.joblib")
    artificial_df = generate_artificial_data(n_samples=10)

    try:
        real_data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
        from utils import load_arff_data
        real_df = load_arff_data(real_data_path)
        real_df = real_df.replace({-1: 0})
        smart_artificial_df = generate_smart_artificial_data(real_df, n_samples=10)
        smart_artificial_df["class"] = 0  # dummy labels for testing (optional)
        run_prediction(pipeline_path, smart_artificial_df, filename_prefix="smart_artificial")
    except Exception as e:
        print("Could not generate smart artificial data:", e)

    artificial_df["class"] = 0  # dummy labels
    run_prediction(pipeline_path, artificial_df, filename_prefix="random_artificial")
    print("Predictions made successfully")
