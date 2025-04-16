"""
iso_forest.py - Data Mining Pipeline for CelebA Bald vs. Non-Bald Anomaly Detection

This script implements the full data mining workflow for the CelebA dataset anomaly detection
assignment. The pipeline is based on an Isolation Forest and consists of the following steps:

    - Load the dataset from an ARFF file and prepare features and labels.
    - Evaluate a baseline dummy model that always predicts the majority class.
    - Perform k-fold cross-validation to assess the performance of the pipeline.
    - Train a final pipeline on the full dataset and predict on the same.
    - Evaluate the final predictions using confusion matrix, classification report, and ROC AUC score.
    - Save cross-validation results, evaluation plots, and the final trained pipeline for future predictions.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 14, 2025
"""

import os
import joblib
import numpy as np
import pandas as pd
import utils
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def build_pipeline(contamination, random_state=42):
    num_features = 39  # manually set to match dataset structure
    encoder = OrdinalEncoder(categories=[[-1, 1]] * num_features) # hard mapping to avoid flipped labels
    pipeline = Pipeline([
        ("encoder", encoder),
        ("model", IsolationForest(contamination=contamination, random_state=random_state))
    ])
    return pipeline

def evaluate(y_true, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def baseline_dummy_model(y_true):
    print("\nBaseline Dummy Model Evaluation:")
    y_pred = np.zeros_like(y_true)  # always predict non-bald (0)
    evaluate(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Baseline AUC: {auc:.4f}")
    return report, auc

def cross_validate_pipeline(contamination, X, y_true, output_dir, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nRunning {n_splits}-Fold Cross Validation...")

    results = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        print(f"\nFold {i+1}:")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_val = y_true.iloc[val_idx]

        unique_classes = np.unique(y_val)
        if len(unique_classes) < 2:
            print(f"Fold {i+1} contains only one class. Skipping evaluation.")
            continue

        start = time.time()
        pipeline = build_pipeline(contamination)
        pipeline.fit(X_train)
        end = time.time()

        # get decision scores
        scores = pipeline.decision_function(X_val)
        threshold = np.percentile(scores, 8)  # threshold tuning to catch more TPs
        y_pred = (scores < threshold).astype(int)

        evaluate(y_val, y_pred)

        report = classification_report(y_val, y_pred, output_dict=True)

        try:
            auc = roc_auc_score(y_val, y_pred)
        except ValueError:
            auc = np.nan
            print(f"Skipping AUC in Fold {i+1}: Only one class present in y_true.")

        results.append({
            "fold": i + 1,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1-score": report["1"]["f1-score"],
            "support": report["1"]["support"],
            "auc": auc,
            "runtime_sec": round(end - start, 2),
            "contamination": contamination,
            "random_state": 42
        })
        print(f"Completed {len(results)} valid folds (out of {n_splits})")

    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"cv_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nCross-validation results saved to {results_path}")

    # plot F1 scores per fold
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["fold"], results_df["f1-score"], color="skyblue")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.title("F1 Score per Fold")
    plt.ylim(0, 1)
    plt.xticks(results_df["fold"])
    plot_path = os.path.join(output_dir, f"cv_f1_scores_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"F1-score plot saved to {plot_path}")


def main():
    start_time = time.time()

    # paths
    data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
    output_dir = os.path.join("..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # load and prep data
    print("Loading Data...")
    df = utils.load_arff_data(data_path)
    X, y_true = utils.prepare_data(df)
    X = X.astype(int)

    # run baseline dummy
    print("\nBaseline Model Evaluation:")
    baseline_report, baseline_auc = baseline_dummy_model(y_true)
    print(f"Baseline AUC: {baseline_auc:.4f}")
    print("\nBaseline Report:")
    print(baseline_report)

    contamination = np.sum(y_true) / len(y_true)  # calculate contamination rate

    # run k-fold validation w logging
    print("\nK-Fold Cross Validation:")
    cross_validate_pipeline(contamination, pd.DataFrame(X), pd.Series(y_true), output_dir, n_splits=5)

    # train final pipeline on whole dataset
    print("\nTraining Final Pipeline on Full Data:")
    print(f"Contamination Rate: {contamination:.4f}")
    pipeline = build_pipeline(contamination)
    pipeline.fit(X)
    print("Pipeline fitted")

    # get anomaly scores (higher = more normal)
    scores = pipeline.decision_function(X)

    # lower threshold = label more samples as anomalies
    threshold = np.percentile(scores, 8)  # top 8% most anomalous
    y_pred = (scores < threshold).astype(int)  # 1 = bald (anomaly), 0 = non-bald

    print("\nFinal Evaluation on Full Data:")
    evaluate(y_true, y_pred)

    # save final pipeline
    print("\nSaving Final Pipeline:")
    model_path = os.path.join(output_dir, "isolation_forest_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Full pipeline saved to {model_path}")

    end_time = time.time()
    print(f"\nTotal Runtime: {end_time - start_time:.2f} seconds")

    plot_path = os.path.join(output_dir, f"roc_auc_curve.png")

    # compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Isolation Forest Anomaly Scores")
    plt.legend()

    # mark the currently selected threshold
    score_threshold = np.percentile(scores, 8)
    closest_idx = np.argmin(np.abs(thresholds - score_threshold))
    plt.scatter(fpr[closest_idx], tpr[closest_idx], color='red', label='Chosen Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()



if __name__ == "__main__":
    main()
