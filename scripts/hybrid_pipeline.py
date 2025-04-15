"""
hybrid_model.py - Hybrid Anomaly Detection Pipeline for CelebA Bald vs. Non-Bald Classification

This script implements a hybrid two-stage machine learning pipeline to detect anomalies 
(i.e., bald individuals) in the CelebA dataset. The model first applies an Isolation Forest 
to identify high-confidence anomalies, then refines the remaining predictions using a 
calibrated Linear Support Vector Machine (SVC).

Key Features:
    - Loads ARFF-formatted data and encodes categorical features using OrdinalEncoder.
    - Applies Isolation Forest for unsupervised anomaly detection and threshold tuning via F1 optimization.
    - Trains a Calibrated Linear SVC on the remaining "normal" instances with threshold tuning.
    - Combines predictions from both stages into a final hybrid prediction.
    - Uses a modular, reusable scikit-learn Pipeline with a custom HybridClassifier class.
    - Evaluates using confusion matrix, classification report, and ROC AUC score.
    - Saves the final pipeline and visual evaluation artifacts (ROC and PR curves).

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT  
Date: April 15, 2025
"""


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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import utils


# custom hybrid classifier combining iso forest and calibrated lin SVC
class HybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.015, svc_max_iter=5000):
        # define parameters
        self.contamination = contamination
        self.svc_max_iter = svc_max_iter

        # encoder to map categorical features to ordinal values
        self.encoder = OrdinalEncoder(categories=[[-1, 1]] * 39, handle_unknown="use_encoded_value", unknown_value=-1)

        # iso forest for detecting high-confidence anomalies
        self.iforest = IsolationForest(contamination=self.contamination, random_state=42)

        # calibrated line SVC for refining preds
        self.svc = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42, max_iter=self.svc_max_iter), method="sigmoid", cv=3)

    # fit the hybrid model on tain data
    def fit(self, X, y):
        X_encoded = self.encoder.fit_transform(X)
        self.iforest.fit(X_encoded)

        # compute anomaly scores
        scores = self.iforest.decision_function(X_encoded)

        # find best threshold compared to F1 score
        self.if_thresh = self._find_best_threshold(y, -scores)
        # ID high-confidence anomalies
        mask = scores < -self.if_thresh

        # fit svc on remaining data
        self.svc.fit(X_encoded[~mask], y[~mask])
        # find best threshold for svc
        self.svc_thresh = self._find_best_threshold(y[~mask], self.svc.predict_proba(X_encoded[~mask])[:, 1])
        return self

    # make preds using the hybrid approach
    def predict(self, X):
        X_encoded = self.encoder.transform(X)

        # get anomaly scores from iso forest
        scores = self.iforest.decision_function(X_encoded)

        # get calibrated SVC prediction probabilities for remaining samples
        mask = scores < -self.if_thresh
        y_scores_svc = self.svc.predict_proba(X_encoded[~mask])[:, 1]

        # init array and gen preds
        preds = np.zeros(X.shape[0], dtype=int)
        preds[mask] = 1
        preds[~mask] = (y_scores_svc >= self.svc_thresh).astype(int)
        return preds

    # return decision scores used in evaluation
    def predict_scores(self, X):
        X_encoded = self.encoder.transform(X)
        scores = self.iforest.decision_function(X_encoded)
        mask = scores < -self.if_thresh
        svc_scores = self.svc.predict_proba(X_encoded[~mask])[:, 1]
        out = np.zeros(X.shape[0], dtype=float)
        out[mask] = -scores[mask]
        out[~mask] = svc_scores
        return out

    # find the best F1 threshold given labels and scores
    def _find_best_threshold(self, y_true, scores):
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        return thresholds[np.argmax(f1_scores)]


# print confusion matrix and anomaly detection report
def evaluate(y_true, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# save metrics and plots to output directory
def save_evaluation_artifacts(y_true, y_scores, y_pred, output_dir):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_dir, "hybrid_model_classification_report.csv"))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()


# main pipeline
def main():
    start = time.time()

    data_path = os.path.join("..", "data", "celeba_baldvsnonbald.arff")
    output_dir = os.path.join("..", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Data...")
    df = utils.load_arff_data(data_path)
    X, y = utils.prepare_data(df)
    X = X.astype(int)

    # split data for eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training Hybrid Model via Pipeline...")
    pipeline = Pipeline([
        ("hybrid", HybridClassifier())
    ])
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on Test Data...")
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.named_steps["hybrid"].predict_scores(X_test)

    evaluate(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC: {auc:.4f}")

    save_evaluation_artifacts(y_test, y_scores, y_pred, output_dir)

    # save the final trained pipeline
    model_path = os.path.join(output_dir, "hybrid_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nHybrid model saved to {model_path}")

    end = time.time()
    print(f"\nTotal Runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
