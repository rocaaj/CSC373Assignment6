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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import utils


def evaluate(y_true, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best F1 threshold: {best_threshold:.4f} | F1: {f1_scores[best_idx]:.4f}")

    # plot PR curve
    plt.plot(recall, precision, label=f"Best F1 = {f1_scores[best_idx]:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return best_threshold

def main():
    start = time.time()

    data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
    output_dir = os.path.join("..", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Data...")
    df = utils.load_arff_data(data_path)
    X, y = utils.prepare_data(df)
    X = X.astype(int)

    encoder = OrdinalEncoder(categories=[[-1, 1]] * 39)
    X_encoded = encoder.fit_transform(X)

    # First stage: Isolation Forest for high-confidence anomalies
    print("Running Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.015, random_state=42)
    isolation_forest.fit(X_encoded)
    scores = isolation_forest.decision_function(X_encoded)

    # Determine threshold from PR curve
    best_threshold = find_best_threshold(y, -scores)
    anomaly_mask = scores < -best_threshold

    X_anomalies = X_encoded[anomaly_mask]
    y_anomalies = y[anomaly_mask]
    print(f"Detected {len(X_anomalies)} anomalies using Isolation Forest")

    # Second stage: Calibrated Linear SVC on the rest
    X_remaining = X_encoded[~anomaly_mask]
    y_remaining = y[~anomaly_mask]

    print("Training Calibrated Linear SVC on remaining data...")
    base_svc = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)
    calibrated_svc = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)
    calibrated_svc.fit(X_remaining, y_remaining)

    # Predict scores and apply threshold tuning
    y_scores_remaining = calibrated_svc.predict_proba(X_remaining)[:, 1]
    best_svm_thresh = find_best_threshold(y_remaining, y_scores_remaining)
    svm_preds = (y_scores_remaining >= best_svm_thresh).astype(int)

    # Combine predictions
    y_pred_hybrid = np.zeros_like(y)
    y_pred_hybrid[anomaly_mask] = 1  # isolation forest says bald
    y_pred_hybrid[~anomaly_mask] = svm_preds

    print("\nEvaluation on Full Dataset:")
    evaluate(y, y_pred_hybrid)

    auc = roc_auc_score(y, y_pred_hybrid)
    print(f"AUC: {auc:.4f}")

    # Save models
    model_path = os.path.join(output_dir, "hybrid_pipeline.joblib")
    joblib.dump((encoder, isolation_forest, calibrated_svc), model_path)
    print(f"\n✅ Hybrid model saved to {model_path}")

    end = time.time()
    print(f"\nTotal Runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
