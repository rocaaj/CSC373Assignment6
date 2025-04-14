import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import utils


def evaluate(y_true, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def main():
    start = time.time()

    data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
    output_dir = os.path.join("..", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Data...")
    df = utils.load_arff_data(data_path)
    X, y = utils.prepare_data(df)
    X = X.astype(int)

    encoder = OrdinalEncoder(categories=[[0, 1]] * 39, handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = encoder.fit_transform(X)

    # First stage: Isolation Forest for high-confidence anomalies
    print("Running Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.0255, random_state=42)
    isolation_forest.fit(X_encoded)
    scores = isolation_forest.decision_function(X_encoded)

    threshold = np.percentile(scores, 8)
    anomaly_mask = scores < threshold

    X_anomalies = X_encoded[anomaly_mask]
    y_anomalies = y[anomaly_mask]
    print(f"Detected {len(X_anomalies)} anomalies using Isolation Forest")

    # Second stage: Linear SVM on the rest
    X_remaining = X_encoded[~anomaly_mask]
    y_remaining = y[~anomaly_mask]

    print("Training Linear SVC on remaining data...")
    svm = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)
    svm.fit(X_remaining, y_remaining)

    # Make predictions
    y_pred_if = np.ones_like(y)  # default: all bald
    y_pred_if[anomaly_mask] = 1  # IF predicts bald
    y_pred_if[~anomaly_mask] = svm.predict(X_remaining)

    print("\nEvaluation on Full Dataset:")
    evaluate(y, y_pred_if)

    auc = roc_auc_score(y, y_pred_if)
    print(f"AUC: {auc:.4f}")

    # Save both models and encoder
    model_path = os.path.join(output_dir, "hybrid_pipeline.joblib")
    joblib.dump((encoder, isolation_forest, svm), model_path)
    print(f"\n✅ Hybrid model saved to {model_path}")

    end = time.time()
    print(f"\nTotal Runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
